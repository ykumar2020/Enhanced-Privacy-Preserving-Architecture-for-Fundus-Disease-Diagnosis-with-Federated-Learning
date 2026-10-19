from __future__ import annotations

from torch import nn

from retina_fl.model import build_model


def test_vgg19_reproduction_freezes_features_and_emits_three_logits():
    model = build_model({"architecture": "vgg19", "imagenet_weights": False, "freeze_features": True})
    assert all(not parameter.requires_grad for parameter in model.features.parameters())
    assert isinstance(model.classifier[-1], nn.Linear)
    assert model.classifier[-1].out_features == 3

