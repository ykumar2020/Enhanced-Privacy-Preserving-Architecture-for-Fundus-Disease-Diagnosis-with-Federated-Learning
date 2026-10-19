from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from torch import nn

from .data import make_loader
from .model import clone_state_dict, trainable_parameters


@dataclass
class LocalUpdate:
    client_id: str
    examples: int
    loss: float
    accuracy: float
    elapsed_seconds: float
    state: dict[str, torch.Tensor]


def train_epoch(model: nn.Module, loader, optimizer, device: torch.device, task: str) -> dict[str, float | int]:
    model.train()
    criterion = nn.BCEWithLogitsLoss() if task == "multilabel" else nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    examples = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        count = int(targets.shape[0])
        examples += count
        loss_sum += float(loss.detach().cpu()) * count
        if task == "multilabel":
            decisions = torch.sigmoid(logits) >= 0.55
            correct += int((decisions == targets.bool()).all(dim=1).sum().detach().cpu())
        else:
            correct += int((logits.argmax(dim=1) == targets).sum().detach().cpu())
    return {
        "loss": loss_sum / examples,
        "accuracy": correct / examples,
        "examples": examples,
    }


@torch.inference_mode()
def predict(model: nn.Module, loader, device: torch.device, task: str, threshold: float) -> tuple[pd.DataFrame, float]:
    model.eval()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images)
        probabilities = (torch.sigmoid(logits) if task == "multilabel" else torch.softmax(logits, dim=1)).cpu()
        targets = batch["target"].cpu()
        for index in range(len(targets)):
            target_value = targets[index]
            row = {
                "image_id": batch["image_id"][index],
                "patient_id": batch["patient_id"][index],
            }
            if task == "multilabel":
                row.update({f"target_{class_index}": int(value) for class_index, value in enumerate(target_value)})
                row.update({f"prediction_{class_index}": int(value >= threshold) for class_index, value in enumerate(probabilities[index])})
            else:
                row["target"] = int(target_value)
                row["prediction"] = int(probabilities[index].argmax())
            row.update({f"probability_{class_index}": float(value) for class_index, value in enumerate(probabilities[index])})
            rows.append(row)
    return pd.DataFrame(rows), time.perf_counter() - started


def aggregate_sample_weighted(updates: list[LocalUpdate]) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    if not updates:
        raise ValueError("Cannot aggregate an empty client update list")
    total_examples = sum(update.examples for update in updates)
    if total_examples <= 0:
        raise ValueError("Selected client updates contain no examples")
    weights = {update.client_id: update.examples / total_examples for update in updates}
    if abs(sum(weights.values()) - 1.0) > 1e-12:
        raise AssertionError("Aggregation weights do not sum to one")
    first = updates[0].state
    aggregated: dict[str, torch.Tensor] = {}
    for name, tensor in first.items():
        if tensor.is_floating_point():
            value = torch.zeros_like(tensor)
            for update in updates:
                value.add_(update.state[name], alpha=weights[update.client_id])
            aggregated[name] = value
        else:
            if any(not torch.equal(update.state[name], tensor) for update in updates[1:]):
                raise ValueError(f"Non-floating state {name!r} differs across clients")
            aggregated[name] = tensor.clone()
    return aggregated, weights


def train_local_update(
    global_model: nn.Module,
    client_frame: pd.DataFrame,
    client_id: str,
    config: dict[str, Any],
    device: torch.device,
    round_index: int,
) -> LocalUpdate:
    model = copy.deepcopy(global_model).to(device)
    training = config["training"]
    loader = make_loader(
        client_frame,
        config["preprocess"],
        str(config["data"].get("task", "multiclass")),
        list(config["data"]["classes"]),
        int(training["batch_size"]),
        train=True,
        num_workers=int(training["num_workers"]),
        seed=int(config["project"]["seed"]) + round_index,
    )
    optimizer = torch.optim.Adam(trainable_parameters(model), lr=float(training["learning_rate"]))
    started = time.perf_counter()
    epoch_metrics = None
    for _ in range(int(training["local_epochs"])):
        epoch_metrics = train_epoch(model, loader, optimizer, device, str(config["data"].get("task", "multiclass")))
    assert epoch_metrics is not None
    elapsed = time.perf_counter() - started
    return LocalUpdate(
        client_id=client_id,
        examples=int(epoch_metrics["examples"]),
        loss=float(epoch_metrics["loss"]),
        accuracy=float(epoch_metrics["accuracy"]),
        elapsed_seconds=elapsed,
        state=clone_state_dict(model),
    )


def select_clients(client_ids: list[str], clients_per_round: int, seed: int, round_index: int) -> list[str]:
    if clients_per_round > len(client_ids):
        raise ValueError("clients_per_round exceeds the number of available clients")
    generator = random.Random(seed + round_index)
    return sorted(generator.sample(client_ids, clients_per_round))

