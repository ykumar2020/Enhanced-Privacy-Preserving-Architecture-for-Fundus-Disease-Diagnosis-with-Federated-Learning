from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
import yaml

from .config import config_digest, resolve_device, save_resolved_config, seed_everything
from .data import make_loader
from .federated import (
    aggregate_sample_weighted,
    predict,
    select_clients,
    train_epoch,
    train_local_update,
)
from .manifest import audit_manifest
from .metrics import classification_metrics, multilabel_metrics
from .model import build_model, clone_state_dict, trainable_parameters


def _prepare_result_dir(config: dict[str, Any], mode: str, overwrite: bool) -> Path:
    directory = Path(config["data"]["output_dir"]).resolve() / mode
    if directory.exists():
        if not overwrite:
            raise FileExistsError(f"Run directory exists: {directory}. Use --overwrite deliberately to replace it.")
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    save_resolved_config(config, directory / "resolved_config.yaml")
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "config_sha256": config_digest(config),
    }
    (directory / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    return directory


def _load_audited_manifest(config: dict[str, Any], result_dir: Path) -> pd.DataFrame:
    source = Path(config["data"]["output_dir"]).resolve() / "manifest.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Prepared manifest not found: {source}. Run `retina-fl prepare` first.")
    frame = pd.read_csv(source, dtype={"patient_id": str, "client_id": str})
    audit = audit_manifest(frame)
    shutil.copy2(source, result_dir / "manifest.csv")
    (result_dir / "manifest_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return frame


def _state_bytes(state: dict[str, torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in state.values())


def _evaluate(model, frame: pd.DataFrame, config: dict[str, Any], device: torch.device):
    task = str(config["data"].get("task", "multiclass"))
    class_names = list(config["data"]["classes"])
    threshold = float(config["data"].get("decision_threshold", 0.5))
    loader = make_loader(
        frame,
        config["preprocess"],
        task,
        class_names,
        int(config["training"]["batch_size"]),
        train=False,
        num_workers=int(config["training"]["num_workers"]),
        seed=int(config["project"]["seed"]),
    )
    predictions, elapsed = predict(model, loader, device, task, threshold)
    probability_columns = [f"probability_{index}" for index in range(len(class_names))]
    if task == "multilabel":
        target_columns = [f"target_{index}" for index in range(len(class_names))]
        metrics = multilabel_metrics(
            predictions[target_columns].to_numpy(), predictions[probability_columns].to_numpy(), class_names, threshold
        )
    else:
        metrics = classification_metrics(
            predictions["target"].tolist(), predictions[probability_columns].to_numpy(), class_names
        )
    metrics["elapsed_seconds"] = elapsed
    return predictions, metrics


def _save_model(model, path: Path) -> str:
    torch.save(model.state_dict(), path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_centralized(config: dict[str, Any], overwrite: bool = False) -> Path:
    seed_everything(int(config["project"]["seed"]))
    result_dir = _prepare_result_dir(config, "centralized", overwrite)
    manifest = _load_audited_manifest(config, result_dir)
    device = resolve_device(config["training"]["device"])
    model = build_model(config["model"], len(config["data"]["classes"])).to(device)
    optimizer = torch.optim.Adam(trainable_parameters(model), lr=float(config["training"]["learning_rate"]))
    loader = make_loader(
        manifest[manifest["split"] == "train"], config["preprocess"],
        str(config["data"].get("task", "multiclass")),
        list(config["data"]["classes"]),
        int(config["training"]["batch_size"]), True,
        int(config["training"]["num_workers"]), int(config["project"]["seed"]),
    )
    rows = []
    started = time.perf_counter()
    for epoch in range(1, int(config["training"]["centralized_epochs"]) + 1):
        values = train_epoch(model, loader, optimizer, device, str(config["data"].get("task", "multiclass")))
        rows.append({"epoch": epoch, **values})
    training_seconds = time.perf_counter() - started
    predictions, metrics = _evaluate(model, manifest[manifest["split"] == "test"], config, device)
    pd.DataFrame(rows).to_csv(result_dir / "epoch_metrics.csv", index=False)
    predictions.to_csv(result_dir / "predictions.csv", index=False)
    metrics["training_seconds"] = training_seconds
    metrics["model_sha256"] = _save_model(model, result_dir / "model.pt")
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return result_dir


def run_local(config: dict[str, Any], overwrite: bool = False) -> Path:
    seed_everything(int(config["project"]["seed"]))
    result_dir = _prepare_result_dir(config, "local", overwrite)
    manifest = _load_audited_manifest(config, result_dir)
    device = resolve_device(config["training"]["device"])
    test_frame = manifest[manifest["split"] == "test"]
    all_predictions = []
    all_metrics: dict[str, Any] = {}
    for client_id in sorted(manifest["client_id"].dropna().unique()):
        model = build_model(config["model"], len(config["data"]["classes"])).to(device)
        optimizer = torch.optim.Adam(trainable_parameters(model), lr=float(config["training"]["learning_rate"]))
        client_frame = manifest[(manifest["split"] == "train") & (manifest["client_id"] == client_id)]
        loader = make_loader(
            client_frame, config["preprocess"], str(config["data"].get("task", "multiclass")),
            list(config["data"]["classes"]),
            int(config["training"]["batch_size"]), True,
            int(config["training"]["num_workers"]), int(config["project"]["seed"]),
        )
        started = time.perf_counter()
        for _ in range(int(config["training"]["centralized_epochs"])):
            train_epoch(model, loader, optimizer, device, str(config["data"].get("task", "multiclass")))
        predictions, metrics = _evaluate(model, test_frame, config, device)
        predictions.insert(0, "model_source", client_id)
        all_predictions.append(predictions)
        metrics["training_seconds"] = time.perf_counter() - started
        metrics["training_images"] = int(len(client_frame))
        metrics["model_sha256"] = _save_model(model, result_dir / f"{client_id}.pt")
        all_metrics[client_id] = metrics
    pd.concat(all_predictions, ignore_index=True).to_csv(result_dir / "predictions.csv", index=False)
    (result_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    return result_dir


def run_federated(config: dict[str, Any], overwrite: bool = False) -> Path:
    seed = int(config["project"]["seed"])
    seed_everything(seed)
    result_dir = _prepare_result_dir(config, "federated", overwrite)
    manifest = _load_audited_manifest(config, result_dir)
    device = resolve_device(config["training"]["device"])
    model = build_model(config["model"], len(config["data"]["classes"])).to(device)
    client_ids = sorted(manifest["client_id"].dropna().unique().tolist())
    rounds = int(config["training"]["rounds"])
    clients_per_round = int(config["training"]["clients_per_round"])
    schedule = []
    round_rows = []
    total_bytes = 0
    total_processed = 0
    started = time.perf_counter()
    for round_index in range(1, rounds + 1):
        selected = select_clients(client_ids, clients_per_round, seed, round_index)
        updates = []
        for client_id in selected:
            client_frame = manifest[(manifest["split"] == "train") & (manifest["client_id"] == client_id)]
            updates.append(train_local_update(model, client_frame, client_id, config, device, round_index))
        aggregated, weights = aggregate_sample_weighted(updates)
        model.load_state_dict(aggregated, strict=True)
        bytes_per_state = _state_bytes(aggregated)
        round_bytes = 2 * len(selected) * bytes_per_state
        total_bytes += round_bytes
        processed = sum(update.examples for update in updates)
        total_processed += processed
        validation_predictions, validation_metrics = _evaluate(
            model, manifest[manifest["split"] == "validation"], config, device
        )
        schedule.append({
            "round": round_index,
            "selected_clients": selected,
            "aggregation_weights": weights,
            "client_examples": {update.client_id: update.examples for update in updates},
        })
        round_rows.append({
            "round": round_index,
            "selected_clients": ",".join(selected),
            "processed_examples": processed,
            "transmitted_bytes": round_bytes,
            "mean_local_loss": float(np.mean([update.loss for update in updates])),
            "mean_local_accuracy": float(np.mean([update.accuracy for update in updates])),
            "validation_primary_accuracy": validation_metrics.get("accuracy", validation_metrics.get("exact_match_accuracy")),
            "validation_balanced_accuracy": validation_metrics.get("balanced_accuracy"),
            "validation_macro_f1": validation_metrics["macro_f1"],
        })
    training_seconds = time.perf_counter() - started
    predictions, metrics = _evaluate(model, manifest[manifest["split"] == "test"], config, device)
    predictions.to_csv(result_dir / "predictions.csv", index=False)
    pd.DataFrame(round_rows).to_csv(result_dir / "round_metrics.csv", index=False)
    (result_dir / "client_schedule.json").write_text(json.dumps(schedule, indent=2), encoding="utf-8")
    communication = {
        "definition": "full global state sent to and full local state returned by each selected client",
        "total_transmitted_bytes": total_bytes,
        "total_processed_examples": total_processed,
        "full_dataset_equivalent_passes": total_processed / len(manifest[manifest["split"] == "train"]),
        "training_seconds": training_seconds,
    }
    (result_dir / "communication.json").write_text(json.dumps(communication, indent=2), encoding="utf-8")
    metrics.update(communication)
    metrics["model_sha256"] = _save_model(model, result_dir / "model.pt")
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return result_dir

