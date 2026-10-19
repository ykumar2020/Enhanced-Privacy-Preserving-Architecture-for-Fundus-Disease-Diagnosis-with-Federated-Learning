from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision.models import VGG19_Weights, vgg19


def build_model(config: dict[str, Any], num_classes: int = 3) -> nn.Module:
    if config["architecture"].lower() != "vgg19":
        raise ValueError("The reproduction currently supports only the paper's VGG-19 backbone")
    weights = VGG19_Weights.IMAGENET1K_V1 if config.get("imagenet_weights", True) else None
    model = vgg19(weights=weights)
    if config.get("freeze_features", True):
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    if not isinstance(model.classifier[-1], nn.Linear):
        raise TypeError("Unexpected torchvision VGG-19 classifier")
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def trainable_parameters(model: nn.Module):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def model_size_bytes(model: nn.Module, trainable_only: bool = False) -> int:
    parameters = (
        parameter for parameter in model.parameters()
        if parameter.requires_grad or not trainable_only
    )
    return sum(parameter.numel() * parameter.element_size() for parameter in parameters)


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

