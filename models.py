import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    resnet34,
    ResNet34_Weights,
    resnet50,
    ResNet50_Weights,
    resnet101,
    ResNet101_Weights,
    vit_b_16,
    ViT_B_16_Weights,
)


def get_model_optimizer(
    backbone="resnet18",
    use_tanh=True,
    optimizer_type="sgdm",
    lr=1e-3,
    weight_decay=1e-2,
    checkpoint_path=None,
):
    if backbone.startswith("resnet"):
        if backbone == "resnet18":
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif backbone == "resnet34":
            model = resnet34(weights=ResNet34_Weights.DEFAULT)
        elif backbone == "resnet50":
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            model = resnet101(weights=ResNet101_Weights.DEFAULT)

        num_in = model.fc.in_features
        model.fc = (
            nn.Sequential(
                nn.Linear(in_features=num_in, out_features=1, bias=True), nn.Tanh()
            )
            if use_tanh
            else nn.Linear(in_features=num_in, out_features=1, bias=True)
        )
    else:
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

        num_in = model.heads.head.in_features
        model.heads = (
            nn.Sequential(
                nn.Linear(in_features=num_in, out_features=1, bias=True), nn.Tanh()
            )
            if use_tanh
            else nn.Linear(in_features=num_in, out_features=1, bias=True)
        )

    if optimizer_type == "sgdm":
        optimizer = optim.SGD(
            model.parameters(),
            momentum=0.9,
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_type == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer
