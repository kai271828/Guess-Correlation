import torch.nn as nn
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


def get_model(backbone="resnet18", use_tanh=True):
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

    return model
