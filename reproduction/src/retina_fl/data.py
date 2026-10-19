from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF


def rgb_clahe(image: np.ndarray, clip_limit: float, grid_size: int, mode: str = "lab_l") -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    if mode == "rgb_channels":
        return np.stack([clahe.apply(image[:, :, channel]) for channel in range(3)], axis=2)
    if mode not in {"lab_l", "original_lab_l"}:
        raise ValueError(f"Unknown CLAHE mode: {mode}")
    conversion_input = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if mode == "original_lab_l" else image
    lab = cv2.cvtColor(conversion_input, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def crop_fundus(image: np.ndarray, threshold: int) -> np.ndarray:
    mask = image.max(axis=2) > threshold
    rows, columns = np.where(mask)
    if not len(rows):
        raise ValueError("Image contains no pixels above the configured background threshold")
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(columns.min()), int(columns.max()) + 1
    cropped = image[top:bottom, left:right]
    height, width = cropped.shape[:2]
    side = max(height, width)
    canvas = np.zeros((side, side, 3), dtype=cropped.dtype)
    y = (side - height) // 2
    x = (side - width) // 2
    canvas[y:y + height, x:x + width] = cropped
    return canvas


class FundusDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, preprocessing: dict[str, Any], task: str, class_names: list[str], train: bool = False):
        self.frame = frame.reset_index(drop=True).copy()
        self.preprocessing = preprocessing
        self.task = task
        self.class_names = class_names
        self.train = train

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        bgr = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Unreadable image: {row['image_path']}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = rgb_clahe(
            rgb,
            float(self.preprocessing["clahe_clip_limit"]),
            int(self.preprocessing["clahe_grid_size"]),
            str(self.preprocessing.get("clahe_mode", "lab_l")),
        )
        rgb = crop_fundus(rgb, int(self.preprocessing["crop_background_threshold"]))
        image = Image.fromarray(rgb).resize(
            (int(self.preprocessing["image_size"]), int(self.preprocessing["image_size"])),
            Image.Resampling.BILINEAR,
        )
        if self.train and torch.rand(()) < float(self.preprocessing["horizontal_flip_probability"]):
            image = TF.hflip(image)
        if self.train and torch.rand(()) < float(self.preprocessing["vertical_flip_probability"]):
            image = TF.vflip(image)
        tensor = TF.normalize(
            TF.to_tensor(image),
            self.preprocessing.get("imagenet_mean", [0.485, 0.456, 0.406]),
            self.preprocessing.get("imagenet_std", [0.229, 0.224, 0.225]),
        )
        if self.task == "multilabel":
            target = torch.tensor(
                [row[f"target_{name}"] for name in self.class_names],
                dtype=torch.float32,
            )
        else:
            target = torch.tensor(int(row["label_index"]), dtype=torch.long)
        return {
            "image": tensor,
            "target": target,
            "image_id": str(row["image_id"]),
            "patient_id": str(row["patient_id"]),
        }


def make_loader(
    frame: pd.DataFrame,
    preprocessing: dict[str, Any],
    task: str,
    class_names: list[str],
    batch_size: int,
    train: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        FundusDataset(frame, preprocessing, task=task, class_names=class_names, train=train),
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        drop_last=False,
    )

