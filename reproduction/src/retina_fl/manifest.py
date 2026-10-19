from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


LABELS = {"normal": 0, "glaucoma": 1, "cataract": 2}
ALIASES = {
    "patient_id": ["id", "patient id", "patient_id"],
    "left_file": ["left-fundus", "left fundus", "left_fundus"],
    "right_file": ["right-fundus", "right fundus", "right_fundus"],
    "left_keywords": ["left-diagnostic keywords", "left diagnostic keywords", "left_diagnostic_keywords"],
    "right_keywords": ["right-diagnostic keywords", "right diagnostic keywords", "right_diagnostic_keywords"],
}


def _column_key(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower().replace("_", " "))


def _resolve_columns(frame: pd.DataFrame) -> dict[str, str]:
    normalized = {_column_key(column): column for column in frame.columns}
    resolved: dict[str, str] = {}
    for field, aliases in ALIASES.items():
        match = next((normalized.get(_column_key(alias)) for alias in aliases if _column_key(alias) in normalized), None)
        if match is None:
            raise ValueError(f"ODIR annotation column for {field!r} was not found. Available: {list(frame.columns)}")
        resolved[field] = match
    return resolved


def _read_annotations(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"ODIR annotations not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("ODIR annotations must be .xlsx, .xls, or .csv")


def _build_from_original_per_eye(source: pd.DataFrame, image_dir: Path, exclude_multilabel: bool) -> tuple[list[dict[str, Any]], Counter[str]]:
    normalized = {_column_key(column): column for column in source.columns}
    id_column = normalized.get("id")
    target_columns = {name: normalized.get(name) for name in LABELS}
    if id_column is None or any(column is None for column in target_columns.values()):
        raise ValueError("Original per-eye CSV must contain ID, normal, glaucoma, and cataract columns")
    accepted: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    for source_row, row in source.iterrows():
        filename = str(row[id_column]).strip()
        targets = {name: int(row[column]) for name, column in target_columns.items()}
        active = [name for name, value in targets.items() if value == 1]
        if not active:
            exclusions["outside_target_endpoint"] += 1
            continue
        if exclude_multilabel and len(active) != 1:
            exclusions["multiple_target_labels"] += 1
            continue
        image_path = image_dir / filename
        if not image_path.is_file():
            exclusions["missing_image_file"] += 1
            continue
        stem = Path(filename).stem
        match = re.match(r"(.+?)_(left|right)$", stem, flags=re.IGNORECASE)
        patient_id = match.group(1) if match else stem
        eye = match.group(2).lower() if match else "unknown"
        accepted.append({
            "image_id": filename,
            "patient_id": patient_id,
            "eye": eye,
            "image_path": str(image_path),
            "label": "+".join(active),
            "label_index": LABELS[active[0]] if len(active) == 1 else -1,
            "target_normal": targets["normal"],
            "target_glaucoma": targets["glaucoma"],
            "target_cataract": targets["cataract"],
            "diagnostic_keywords": "not_available_in_original_per_eye_csv",
            "source_row": int(source_row),
        })
    return accepted, exclusions


def classify_eye(keyword_text: Any, strict: bool = True) -> tuple[str | None, str]:
    text = re.sub(r"\s+", " ", str(keyword_text or "").strip().lower())
    if not text or text == "nan":
        return None, "missing_keywords"
    candidates: set[str] = set()
    if "normal fundus" in text or text == "normal":
        candidates.add("normal")
    if "glaucoma" in text:
        if strict and any(term in text for term in ("suspect", "suspicious", "large cup")):
            return None, "ambiguous_glaucoma_keyword"
        candidates.add("glaucoma")
    if "cataract" in text:
        candidates.add("cataract")
    if len(candidates) == 1:
        return next(iter(candidates)), "accepted"
    if len(candidates) > 1:
        return None, "multiple_target_labels"
    return None, "outside_target_endpoint"


def _patient_signatures(frame: pd.DataFrame) -> pd.DataFrame:
    signatures = (
        frame.groupby("patient_id")["label"]
        .apply(lambda values: "+".join(sorted(set(values))))
        .rename("signature")
        .reset_index()
    )
    return signatures


def _safe_split(ids: list[str], labels: list[str], test_size: float, seed: int) -> tuple[list[str], list[str], bool]:
    counts = Counter(labels)
    stratify = labels if len(counts) > 1 and min(counts.values()) >= 2 else None
    left, right = train_test_split(ids, test_size=test_size, random_state=seed, stratify=stratify)
    return list(left), list(right), stratify is not None


def assign_patient_splits(frame: pd.DataFrame, fractions: list[float], seed: int) -> tuple[pd.DataFrame, list[str]]:
    signatures = _patient_signatures(frame)
    patient_ids = signatures["patient_id"].astype(str).tolist()
    labels = signatures["signature"].tolist()
    train_val, test, stratified_test = _safe_split(patient_ids, labels, fractions[2], seed)
    remaining = signatures.set_index("patient_id").loc[train_val].reset_index()
    validation_fraction = fractions[1] / (fractions[0] + fractions[1])
    train, validation, stratified_validation = _safe_split(
        remaining["patient_id"].astype(str).tolist(),
        remaining["signature"].tolist(),
        validation_fraction,
        seed + 1,
    )
    split_map = {patient: "train" for patient in train}
    split_map.update({patient: "validation" for patient in validation})
    split_map.update({patient: "test" for patient in test})
    output = frame.copy()
    output["split"] = output["patient_id"].map(split_map)
    warnings = []
    if not stratified_test:
        warnings.append("test split was group-random because at least one patient label signature was too rare to stratify")
    if not stratified_validation:
        warnings.append("validation split was group-random because at least one patient label signature was too rare to stratify")
    return output, warnings


def assign_clients(frame: pd.DataFrame, num_clients: int, seed: int) -> pd.DataFrame:
    output = frame.copy()
    output["client_id"] = pd.NA
    train = output[output["split"] == "train"]
    patient_rows = train.groupby("patient_id").agg(image_count=("image_id", "size"), signature=("label", lambda x: "+".join(sorted(set(x))))).reset_index()
    patient_rows["tie_break"] = patient_rows["patient_id"].map(
        lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    )
    patient_rows = patient_rows.sort_values(["image_count", "signature", "tie_break"], ascending=[False, True, True])
    client_load = [0] * num_clients
    client_class_load = [Counter() for _ in range(num_clients)]
    assignments: dict[str, str] = {}
    for row in patient_rows.itertuples(index=False):
        choices = sorted(
            range(num_clients),
            key=lambda index: (client_class_load[index][row.signature], client_load[index], index),
        )
        selected = choices[0]
        assignments[str(row.patient_id)] = f"client_{selected:02d}"
        client_load[selected] += int(row.image_count)
        client_class_load[selected][row.signature] += int(row.image_count)
    train_mask = output["split"] == "train"
    output.loc[train_mask, "client_id"] = output.loc[train_mask, "patient_id"].map(assignments)
    return output


def build_manifest(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = config["data"]
    if not data.get("patient_separated", True):
        raise ValueError(
            "This reproduction requires patient-separated splits. The manuscript does not provide "
            "the historical patient assignments needed to reproduce an unsafe image-level split exactly."
        )
    annotations_path = Path(data["annotations"]).resolve()
    image_dir = Path(data["image_dir"]).resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"ODIR image directory not found: {image_dir}")
    source = _read_annotations(annotations_path)
    if data.get("label_source") == "original_per_eye_csv":
        accepted, exclusions = _build_from_original_per_eye(source, image_dir, bool(data.get("exclude_multilabel", False)))
    else:
        columns = _resolve_columns(source)
        accepted = []
        exclusions = Counter()
        for source_row, row in source.iterrows():
            patient_id = str(row[columns["patient_id"]]).strip()
            for eye in ("left", "right"):
                filename = str(row[columns[f"{eye}_file"]]).strip()
                label, reason = classify_eye(row[columns[f"{eye}_keywords"]], bool(data.get("strict_eye_labels", True)))
                if label is None:
                    exclusions[reason] += 1
                    continue
                image_path = image_dir / filename
                if not image_path.is_file():
                    exclusions["missing_image_file"] += 1
                    continue
                targets = {name: int(name == label) for name in LABELS}
                accepted.append({
                    "image_id": filename,
                    "patient_id": patient_id,
                    "eye": eye,
                    "image_path": str(image_path),
                    "label": label,
                    "label_index": LABELS[label],
                    "target_normal": targets["normal"],
                    "target_glaucoma": targets["glaucoma"],
                    "target_cataract": targets["cataract"],
                    "diagnostic_keywords": str(row[columns[f"{eye}_keywords"]]),
                    "source_row": int(source_row),
                })
    manifest = pd.DataFrame(accepted)
    if manifest.empty:
        raise ValueError("No eligible normal/glaucoma/cataract images were found")
    if str(data.get("task", "multiclass")) == "multiclass":
        class_to_index = {name: index for index, name in enumerate(data["classes"])}
        manifest["label_index"] = manifest["label"].map(class_to_index)
        if manifest["label_index"].isna().any():
            unknown = sorted(manifest.loc[manifest["label_index"].isna(), "label"].unique())
            raise ValueError(f"Manifest contains labels absent from data.classes: {unknown}")
        manifest["label_index"] = manifest["label_index"].astype(int)
    duplicates = manifest[manifest.duplicated("image_id", keep=False)]
    if not duplicates.empty:
        raise ValueError(f"Duplicate accepted image filenames: {duplicates['image_id'].unique().tolist()[:10]}")
    manifest, warnings = assign_patient_splits(manifest, list(data["split"]), int(config["project"]["seed"]))
    manifest = assign_clients(manifest, int(data["num_clients"]), int(config["project"]["seed"]))
    manifest = manifest.sort_values(["split", "patient_id", "eye"]).reset_index(drop=True)
    report = {
        "annotation_rows": int(len(source)),
        "eligible_images": int(len(manifest)),
        "eligible_patients": int(manifest["patient_id"].nunique()),
        "target_3402_count_matched": bool(len(manifest) == 3402),
        "patient_separated": True,
        "class_positive_counts": {name: int(manifest[f"target_{name}"].sum()) for name in LABELS},
        "label_combination_counts": manifest["label"].value_counts().sort_index().to_dict(),
        "split_image_counts": manifest["split"].value_counts().to_dict(),
        "split_patient_counts": manifest.groupby("split")["patient_id"].nunique().to_dict(),
        "client_image_counts": manifest.dropna(subset=["client_id"])["client_id"].value_counts().sort_index().to_dict(),
        "excluded_eye_records": dict(sorted(exclusions.items())),
        "warnings": warnings + ([] if len(manifest) == 3402 else ["strict eye-level selection did not yield the paper's undocumented 3,402-image subset"]),
    }
    return manifest, report


def audit_manifest(frame: pd.DataFrame, require_files: bool = True) -> dict[str, Any]:
    required = {
        "image_id", "patient_id", "image_path", "label", "label_index", "split", "client_id",
        "target_normal", "target_glaucoma", "target_cataract",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    errors: list[str] = []
    duplicated = frame[frame.duplicated("image_id", keep=False)]["image_id"].unique().tolist()
    if duplicated:
        errors.append(f"duplicate image IDs: {duplicated[:10]}")
    split_counts = frame.groupby("patient_id")["split"].nunique()
    leaking_splits = split_counts[split_counts > 1].index.tolist()
    if leaking_splits:
        errors.append(f"patients cross data splits: {leaking_splits[:10]}")
    train = frame[frame["split"] == "train"].dropna(subset=["client_id"])
    client_counts = train.groupby("patient_id")["client_id"].nunique()
    leaking_clients = client_counts[client_counts > 1].index.tolist()
    if leaking_clients:
        errors.append(f"patients cross federated clients: {leaking_clients[:10]}")
    if frame[frame["split"] == "train"]["client_id"].isna().any():
        errors.append("training images without a client assignment")
    if frame[frame["split"] != "train"]["client_id"].notna().any():
        errors.append("validation/test images have client assignments")
    if require_files:
        missing_files = [path for path in frame["image_path"] if not Path(path).is_file()]
        if missing_files:
            errors.append(f"missing image files: {missing_files[:10]}")
    report = {
        "ok": not errors,
        "errors": errors,
        "images": int(len(frame)),
        "patients": int(frame["patient_id"].nunique()),
        "manifest_sha256": hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest(),
        "split_images": frame["split"].value_counts().to_dict(),
        "split_patients": frame.groupby("split")["patient_id"].nunique().to_dict(),
        "class_positive_counts": {name: int(frame[f"target_{name}"].sum()) for name in LABELS},
        "label_combination_counts": frame["label"].value_counts().to_dict(),
    }
    if errors:
        raise ValueError("Manifest audit failed: " + "; ".join(errors))
    return report


def write_manifest_artifacts(manifest: pd.DataFrame, report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    (output_dir / "selection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    audit = audit_manifest(manifest)
    (output_dir / "manifest_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

