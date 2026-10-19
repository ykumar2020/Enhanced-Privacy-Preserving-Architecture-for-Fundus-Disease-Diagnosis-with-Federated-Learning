from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


KAGGLE_HANDLE = "andrewmvd/ocular-disease-recognition-odir5k"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def locate_odir(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    annotation_candidates = [
        path for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".xlsx", ".xls", ".csv"}
        and ("annotation" in path.name.lower() or path.name.lower() == "full_df.csv")
    ]
    annotation_candidates.sort(
        key=lambda path: (
            "training_annotations" not in path.name.lower(),
            path.suffix.lower() != ".xlsx",
            len(path.parts),
            str(path),
        )
    )
    if not annotation_candidates:
        raise FileNotFoundError(f"No ODIR annotation workbook/CSV found under {root}")

    image_counts = Counter(
        path.parent for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_counts:
        raise FileNotFoundError(f"No fundus images found under {root}")

    def image_rank(item: tuple[Path, int]) -> tuple[int, int, int, str]:
        directory, count = item
        name = directory.name.lower().replace("_", " ")
        return (
            int("training images" in name),
            int(name == "raw"),
            count,
            str(directory),
        )

    image_dir, image_count = max(image_counts.items(), key=image_rank)
    if image_count < 1000:
        raise ValueError(f"Largest image directory contains only {image_count} files: {image_dir}")

    return {
        "dataset_root": str(root),
        "annotations": str(annotation_candidates[0]),
        "image_dir": str(image_dir),
        "image_count": int(image_count),
        "annotation_candidates": [str(path) for path in annotation_candidates],
    }


def write_runtime_config(
    template_path: str | Path,
    location: dict[str, Any],
    destination: str | Path,
    run_root: str | Path,
) -> Path:
    config = yaml.safe_load(Path(template_path).read_text(encoding="utf-8"))
    config["data"]["annotations"] = location["annotations"]
    config["data"]["image_dir"] = location["image_dir"]
    config["data"]["output_dir"] = str(Path(run_root).resolve())
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return destination


def acquire_odir(
    destination: str | Path,
    template_path: str | Path,
    config_path: str | Path,
    run_root: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("Install the Colab extra first: pip install -e '.[colab]'") from exc

    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        kagglehub.dataset_download(
            KAGGLE_HANDLE,
            output_dir=str(destination),
            force_download=force,
        )
    ).resolve()
    location = locate_odir(downloaded)
    location.update({"kaggle_handle": KAGGLE_HANDLE, "download_path": str(downloaded)})
    runtime_config = write_runtime_config(template_path, location, config_path, run_root)
    location["runtime_config"] = str(runtime_config)
    evidence_path = runtime_config.parent / "dataset_location.json"
    evidence_path.write_text(json.dumps(location, indent=2), encoding="utf-8")
    return location
