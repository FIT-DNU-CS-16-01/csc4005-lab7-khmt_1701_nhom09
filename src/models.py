from __future__ import annotations

from torch import nn


def build_vit_teacher(num_classes: int = 5, model_name: str = "vit_b_16", dropout: float = 0.2):
    from torchvision.models import vit_b_16, vit_b_32

    if model_name == "vit_b_16":
        model = vit_b_16(weights=None)
    elif model_name == "vit_b_32":
        model = vit_b_32(weights=None)
    else:
        raise ValueError(f"Unsupported teacher model: {model_name}")

    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return model


def build_student(num_classes: int = 5, student_model: str = "mobilenet_v2", pretrained: bool = True):
    from torchvision.models import MobileNet_V2_Weights, ResNet18_Weights, mobilenet_v2, resnet18

    if student_model == "mobilenet_v2":
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if student_model == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported student model: {student_model}")


def load_teacher_checkpoint(checkpoint_path: str, num_classes: int = 5, model_name: str = "vit_b_16"):
    import torch

    model = build_vit_teacher(num_classes=num_classes, model_name=model_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    return model
