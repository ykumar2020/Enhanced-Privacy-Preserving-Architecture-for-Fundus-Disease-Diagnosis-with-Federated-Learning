from __future__ import annotations

import numpy as np

from retina_fl.data import crop_fundus, rgb_clahe


def test_crop_fundus_removes_background_and_returns_square():
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[10:30, 15:45] = [30, 100, 180]
    cropped = crop_fundus(image, threshold=10)
    assert cropped.shape == (30, 30, 3)
    assert cropped.max() == 180


def test_rgb_clahe_preserves_shape_and_dtype():
    image = np.tile(np.arange(32, dtype=np.uint8), (32, 1))
    rgb = np.stack([image, image, image], axis=2)
    enhanced = rgb_clahe(rgb, clip_limit=2.0, grid_size=8)
    assert enhanced.shape == rgb.shape
    assert enhanced.dtype == np.uint8

