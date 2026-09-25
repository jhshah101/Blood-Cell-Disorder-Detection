"""torchvision CNN baselines wrapped with the same input normalisation as the
proposed model, so that every row of the comparison tables shares one input
pipeline and differs only in the network."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models as tvm

from ..config import CNN_BASELINES
from .hybrid import InputNormalization


def _replace_classifier(name: str, m: nn.Module, num_classes: int) -> nn.Module:
    if name in ("resnet18", "resnet50", "googlenet"):
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "densenet121":
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    elif name in ("mobilenet_v2", "efficientnet_b1"):
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif name == "squeezenet1_0":
        m.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        m.num_classes = num_classes
    else:
        raise ValueError(f"Unsupported baseline {name!r}")
    return m


class CNNBaseline(nn.Module):
    def __init__(self, name: str, num_classes: int, pretrained: bool = True, normalization: str = "imagenet"):
        super().__init__()
        if name not in CNN_BASELINES:
            raise ValueError(f"baseline must be one of {CNN_BASELINES}")
        self.backbone_name = name
        self.input_norm = InputNormalization(normalization)
        ctor = getattr(tvm, name)
        kwargs = {}
        if pretrained:
            enum_name = {
                "resnet18": "ResNet18_Weights",
                "resnet50": "ResNet50_Weights",
                "densenet121": "DenseNet121_Weights",
                "mobilenet_v2": "MobileNet_V2_Weights",
                "googlenet": "GoogLeNet_Weights",
                "efficientnet_b1": "EfficientNet_B1_Weights",
                "squeezenet1_0": "SqueezeNet1_0_Weights",
            }[name]
            kwargs["weights"] = getattr(tvm, enum_name).IMAGENET1K_V1
        else:
            kwargs["weights"] = None
        if name == "googlenet":
            kwargs["aux_logits"] = False
            if not pretrained:
                kwargs["init_weights"] = True
        net = ctor(**kwargs)
        self.net = _replace_classifier(name, net, num_classes)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        out = self.net(self.input_norm(raw))
        if isinstance(out, tuple):  # GoogLeNet in training mode with aux logits
            out = out[0]
        return out
