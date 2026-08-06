# ============================================
# WBC Classification (CNN + Transformer + ECA)
# - No Augmentation
# - Class Weighted Loss (minority priority)
# - Cross-Dataset Evaluation
# ============================================

import os
import math
import time
import copy
from dataclasses import dataclass
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18

# -----------------------------
# 1) Reproducibility
# -----------------------------
def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------------
# 2) ECA Module (Efficient Channel Attention)
# Paper idea: lightweight 1D conv across channels
# -----------------------------
class ECABlock(nn.Module):
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        # Adaptive kernel sizing is common; keep it simple & stable.
        # You can tune k_size (3,5,7) depending on channels.
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        y = self.avg_pool(x)                  # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)   # (B, 1, C)
        y = self.conv(y)                      # (B, 1, C)
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1) # (B, C, 1, 1)
        return x * y

# -----------------------------
# 3) CNN Backbone (ResNet18) -> Feature Map
# Remove classifier and keep conv features
# -----------------------------
class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, pretrained: bool = True, out_channels: int = 512):
        super().__init__()
        m = resnet18(weights="DEFAULT" if pretrained else None)
        # Keep everything up to layer4 (feature map)
        self.stem = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4
        )
        self.out_channels = out_channels  # resnet18 layer4 outputs 512

    def forward(self, x):
        # (B, 3, H, W) -> (B, 512, H/32, W/32)
        return self.stem(x)

# -----------------------------
# 4) Transformer Encoder Block
# -----------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, N, D)
        h = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = h + self.drop1(x)

        h = x
        x = self.norm2(x)
        x = h + self.mlp(x)
        return x

# -----------------------------
# 5) Hybrid Model: CNN -> ECA -> Patch tokens -> Transformer -> Classifier
# -----------------------------
class WBC_ECA_CNNTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        img_size: int = 224,
        pretrained_cnn: bool = True,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        eca_k: int = 3
    ):
        super().__init__()
        self.backbone = ResNet18FeatureExtractor(pretrained=pretrained_cnn, out_channels=512)
        self.eca = ECABlock(channels=512, k_size=eca_k)

        # Feature map size depends on input; for 224, resnet18 -> 7x7
        # We convert (B, C, Hf, Wf) into tokens (B, N=Hf*Wf, C)
        self.proj = nn.Linear(512, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Max tokens for 224 -> 7*7 + 1 = 50, but keep flexible by learned pos embed per token count
        # We'll build pos embedding dynamically (interpolated) to support other image sizes.
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 49, embed_dim))  # (1, 50, D) baseline
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoder(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def _resize_pos_embed(self, pos_embed: torch.Tensor, target_tokens: int) -> torch.Tensor:
        """
        pos_embed: (1, 1+N0, D) where N0=49
        target_tokens: 1+N1
        We interpolate only patch tokens, keep cls token.
        """
        if pos_embed.shape[1] == target_tokens:
            return pos_embed

        cls_pos = pos_embed[:, :1, :]          # (1,1,D)
        patch_pos = pos_embed[:, 1:, :]        # (1,N0,D)

        N0 = patch_pos.shape[1]
        D = patch_pos.shape[2]
        # Assume square grid for baseline N0
        gs0 = int(math.sqrt(N0))
        patch_pos = patch_pos.reshape(1, gs0, gs0, D).permute(0, 3, 1, 2)  # (1,D,gs0,gs0)

        N1 = target_tokens - 1
        gs1 = int(math.sqrt(N1))
        patch_pos = F.interpolate(patch_pos, size=(gs1, gs1), mode="bilinear", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, gs1 * gs1, D)  # (1,N1,D)

        return torch.cat([cls_pos, patch_pos], dim=1)  # (1,1+N1,D)

    def forward(self, x):
        # CNN features
        f = self.backbone(x)     # (B,512,Hf,Wf)
        f = self.eca(f)          # ECA-guided channel attention

        B, C, Hf, Wf = f.shape
        tokens = f.flatten(2).transpose(1, 2)  # (B, N=Hf*Wf, C)
        tokens = self.proj(tokens)             # (B, N, D)

        cls = self.cls_token.expand(B, -1, -1) # (B,1,D)
        x = torch.cat([cls, tokens], dim=1)    # (B,1+N,D)

        # positional embedding (interpolated if needed)
        pos = self._resize_pos_embed(self.pos_embed, x.shape[1])
        x = x + pos
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]                     # (B,D)
        logits = self.head(cls_out)           # (B,num_classes)
        return logits

# -----------------------------
# 6) Class Weighted Loss
# -----------------------------
def compute_class_weights_from_folder(imagefolder_dataset: datasets.ImageFolder, num_classes: int) -> torch.Tensor:
    # counts by class index
    counts = torch.zeros(num_classes, dtype=torch.float)
    for _, y in imagefolder_dataset.samples:
        counts[y] += 1.0

    # Inverse-frequency weights (common for imbalance)
    # weight_c = total / (num_classes * count_c)
    total = counts.sum().clamp_min(1.0)
    weights = total / (num_classes * counts.clamp_min(1.0))

    # Optional: normalize (not necessary but often stable)
    weights = weights / weights.sum() * num_classes
    return weights

# -----------------------------
# 7) Metrics (Accuracy + Macro F1)
# -----------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int):
    model.eval()
    correct = 0
    total = 0

    # confusion matrix for macro-F1
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.numel()

        for t, p in zip(y.view(-1), pred.view(-1)):
            cm[t.long(), p.long()] += 1

    acc = correct / max(total, 1)

    # Macro F1
    f1s = []
    for c in range(num_classes):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1s.append(f1)

    macro_f1 = sum(f1s) / num_classes
    return acc, macro_f1, cm

# -----------------------------
# 8) Training Loop
# -----------------------------
def train_one_run(
    train_dir: str,
    test_dir: str,
    img_size: int = 224,
    batch_size: int = 32,
    lr: float = 3e-4,
    epochs: int = 20,
    num_workers: int = 4,
    pretrained_cnn: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    device = torch.device(device)

    # NO AUGMENTATION (per abstract)
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    test_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    test_ds  = datasets.ImageFolder(test_dir,  transform=test_tf)

    num_classes = len(train_ds.classes)
    assert len(test_ds.classes) == num_classes, (
        "Train/Test must have same class names & count. "
        "Ensure both folders use identical subfolder class names."
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # class weights from TRAIN distribution (imbalance handling)
    class_weights = compute_class_weights_from_folder(train_ds, num_classes).to(device)

    model = WBC_ECA_CNNTransformer(
        num_classes=num_classes,
        img_size=img_size,
        pretrained_cnn=pretrained_cnn,
        embed_dim=256,
        depth=4,
        num_heads=8,
        dropout=0.1,
        eca_k=3
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

    # simple scheduler (optional)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best = {"acc": 0.0, "f1": 0.0, "state": None}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            n += x.size(0)

        scheduler.step()

        train_loss = running_loss / max(n, 1)
        test_acc, test_f1, cm = evaluate(model, test_loader, device, num_classes)

        if test_f1 > best["f1"]:
            best["f1"] = test_f1
            best["acc"] = test_acc
            best["state"] = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch:02d}/{epochs} | loss={train_loss:.4f} | test_acc={test_acc:.4f} | test_macroF1={test_f1:.4f}")

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    print("\n===== BEST (by Macro-F1) =====")
    print(f"Best Test Accuracy: {best['acc']:.4f}")
    print(f"Best Test Macro-F1 : {best['f1']:.4f}")
    print("Classes:", train_ds.classes)

    return model, train_ds.classes

# -----------------------------
# 9) Example Main (Cross Dataset)
# Folder structure (ImageFolder):
# train_dir/
#   class1/ img1.jpg ...
#   class2/ ...
# test_dir/
#   class1/ ...
#   class2/ ...
# -----------------------------
if __name__ == "__main__":
    set_seed(42)

    # Example:
    # Train on Dataset-A, test on Dataset-B (cross dataset)
    TRAIN_DIR = r"/path/to/WBC_Dataset_A/train"
    TEST_DIR  = r"/path/to/WBC_Dataset_B/test"

    # If your dataset doesn't have train/test split, you can create folders accordingly.
    model, classes = train_one_run(
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        img_size=224,
        batch_size=32,
        lr=3e-4,
        epochs=20,
        num_workers=4,
        pretrained_cnn=True
    )

    # Save best model
    torch.save({"model_state": model.state_dict(), "classes": classes}, "wbc_eca_cnn_transformer.pth")
    print("\nSaved: wbc_eca_cnn_transformer.pth")
