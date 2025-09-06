# -*- coding: utf-8 -*-
"""
Hybrid CNN–ViT training script
- Replaces plain ResNet50 classifier with a Hybrid CNN–Transformer encoder
- Preserves your dataloaders, early stopping, metrics, and plots

If you previously ran the old script, this one can replace it directly.
"""

import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from typing import List

# --------------------------
# Paths (EDIT if needed)
# --------------------------
train_dir = r'D:\System Variables\Ongoing\White Blood cells\Train\Train'
val_dir   = r'D:\System Variables\Ongoing\White Blood cells\val'
test_dir  = r'D:\System Variables\Ongoing\White Blood cells\Train\Test'

# Model save directory
save_dir = r'D:\System Variables\Ongoing\White Blood cells\Save'
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, 'best_model_hybrid_cnn_vit.pth')

# --------------------------
# Hyperparameters (EDIT if needed)
# --------------------------
batch_size   = 32
num_epochs   = 200
patience     = 10
lr           = 2e-4
weight_decay = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# Transforms (use ImageNet mean/std for pretrained CNN)
# --------------------------
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    # You can uncomment simple augmentations if desired:
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# --------------------------
# Datasets and Loaders
# --------------------------
train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
val_dataset   = datasets.ImageFolder(val_dir,   transform=transform_eval)
test_dataset  = datasets.ImageFolder(test_dir,  transform=transform_eval)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)

num_classes = len(train_dataset.classes)

# --------------------------
# Class weights (computed once from train set for reproducibility)
# --------------------------
def compute_class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    # dataset.samples is a list of (path, class_index)
    labels = [cls for _, cls in dataset.samples]
    counts = Counter(labels)
    total  = len(labels)
    # Inverse frequency weighting normalized to sum=1 (or you can skip normalization)
    raw = [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
    weights = torch.tensor(raw, dtype=torch.float)
    # Normalize (optional): keeps the scale stable
    weights = weights / weights.sum()
    return weights

class_weights = compute_class_weights(train_dataset).to(device)

# --------------------------
# Hybrid CNN–ViT Model
#   - CNN backbone (ResNet18) → (B, C, H, W)
#   - 1x1 conv to project to d_model
#   - Flatten spatial to tokens (H*W, d_model) + CLS token
#   - Add learnable positional embeddings
#   - TransformerEncoder
#   - Classifier on CLS token
# --------------------------
class HybridCNNViT(nn.Module):
    def __init__(
        self,
        num_classes: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        backbone: str = "resnet18",
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # --- CNN backbone ---
        if backbone == "resnet18":
            cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            cnn_out_channels = 512
        elif backbone == "resnet34":
            cnn = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            cnn_out_channels = 512
        else:
            # Fallback
            cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            cnn_out_channels = 512

        # remove avgpool and fc, keep conv features (B, C, H, W)
        self.cnn_stem = nn.Sequential(
            cnn.conv1, cnn.bn1, cnn.relu, cnn.maxpool,
            cnn.layer1, cnn.layer2, cnn.layer3, cnn.layer4
        )
        if freeze_backbone:
            for p in self.cnn_stem.parameters():
                p.requires_grad = False

        # Project CNN channels → d_model for transformer
        self.proj = nn.Conv2d(cnn_out_channels, d_model, kernel_size=1)

        # we expect 224x224 -> after resnet18 layers: 7x7 features (typical)
        # but we’ll infer H,W dynamically for safety and create positional embeddings accordingly.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = None  # will init on first forward

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # (B, N, d_model)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        # init
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.constant_(self.head.bias, 0.0)

    def _build_pos_embed(self, n_tokens: int, d_model: int, device):
        # learnable positional embeddings: (1, N+1, d_model) incl. CLS
        pe = nn.Parameter(torch.zeros(1, n_tokens + 1, d_model, device=device))
        nn.init.trunc_normal_(pe, std=0.02)
        return pe

    def forward(self, x):
        # x: (B, 3, 224, 224)
        feats = self.cnn_stem(x)          # (B, C, H, W)
        feats = self.proj(feats)          # (B, d_model, H, W)
        B, D, H, W = feats.shape
        # to tokens
        tokens = feats.flatten(2).transpose(1, 2)  # (B, H*W, D)

        # CLS token
        cls_tok = self.cls_token.expand(B, -1, -1)  # (B, 1, D)

        # positional embeddings (create once lazily to match H*W)
        if (self.pos_embed is None) or (self.pos_embed.shape[1] != (H * W + 1)):
            self.pos_embed = self._build_pos_embed(H * W, D, x.device)

        x_seq = torch.cat([cls_tok, tokens], dim=1) + self.pos_embed  # (B, 1+HW, D)

        # Transformer
        x_seq = self.transformer(x_seq)  # (B, 1+HW, D)

        # CLS token representation
        cls_out = self.norm(x_seq[:, 0, :])  # (B, D)
        logits = self.head(cls_out)          # (B, num_classes)
        return logits

# --------------------------
# Instantiate model, loss, optimizer
# --------------------------
model = HybridCNNViT(
    num_classes=num_classes,
    d_model=256,
    nhead=8,
    num_layers=4,
    dim_feedforward=512,
    dropout=0.1,
    backbone="resnet18",
    freeze_backbone=False
).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# --------------------------
# Evaluation helper
# --------------------------
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_labels, all_preds

# --------------------------
# Training loop (with early stopping)
# --------------------------
best_val_loss = float('inf')
early_stopping_counter = 0

train_losses, val_losses = [], []
train_accs, val_accs = [], []

print(f"Classes: {train_dataset.classes}")
print("Class weights (normalized):")
for idx, w in enumerate(class_weights.tolist()):
    print(f"  {train_dataset.classes[idx]}: {w:.4f}")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train, total_train = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc  = correct_train / total_train
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)

    # Validation
    val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Early stopping and save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        torch.save(model.state_dict(), model_path)
        print(f"🎉 New best model saved at epoch {epoch+1} "
              f"with Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print("⏹️ Early stopping triggered.")
            break

# --------------------------
# Load best model
# --------------------------
model.load_state_dict(torch.load(model_path, map_location=device))

# --------------------------
# Final evaluation
# --------------------------
train_loss, train_acc, train_labels, train_preds = evaluate(model, train_loader, criterion)
test_loss,  test_acc,  test_labels,  test_preds  = evaluate(model, test_loader,  criterion)

# Confusion matrices
train_cm = confusion_matrix(train_labels, train_preds)
test_cm  = confusion_matrix(test_labels,  test_preds)

# --------------------------
# Plot confusion matrices and curves
# --------------------------
fig, axs = plt.subplots(1, 3, figsize=(20, 6))

ConfusionMatrixDisplay(train_cm, display_labels=train_dataset.classes).plot(ax=axs[0], xticks_rotation=45)
axs[0].set_title("Train Confusion Matrix")

ConfusionMatrixDisplay(test_cm, display_labels=test_dataset.classes).plot(ax=axs[1], xticks_rotation=45)
axs[1].set_title("Test Confusion Matrix")

axs[2].plot(train_losses, label='Train Loss')
axs[2].plot(val_losses,   label='Validation Loss')
axs[2].plot(train_accs,   label='Train Acc')
axs[2].plot(val_accs,     label='Validation Acc')
axs[2].set_title("Loss & Accuracy Curves")
axs[2].set_xlabel("Epoch")
axs[2].set_ylabel("Value")
axs[2].legend()

plt.tight_layout()
plt.show()

# --------------------------
# Print final results
# --------------------------
print(f"\n✅ Final Train Accuracy: {train_acc*100:.2f}%")
print(f"✅ Final Test  Accuracy: {test_acc*100:.2f}%")
print("\nNormalized Class Weights (sum = {:.2f}):".format(class_weights.sum().item()))
for idx, w in enumerate(class_weights.tolist()):
    print(f"Class '{train_dataset.classes[idx]}': {w:.4f}")
