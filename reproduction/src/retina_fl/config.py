from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required = {"project", "data", "preprocess", "model", "training", "federated"}
    missing = required - set(config or {})
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    split = config["data"]["split"]
    if len(split) != 3 or not np.isclose(sum(split), 1.0):
        raise ValueError("data.split must contain train/validation/test fractions summing to 1")
    if config["data"].get("task", "multiclass") not in {"multiclass", "multilabel"}:
        raise ValueError("data.task must be multiclass or multilabel")
    classes = config["data"].get("classes", [])
    if sorted(classes) != ["cataract", "glaucoma", "normal"]:
        raise ValueError("data.classes must contain normal, glaucoma, and cataract exactly once")
    return config


def save_resolved_config(config: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def config_digest(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device

