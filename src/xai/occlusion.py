from __future__ import annotations

from typing import Callable

import numpy as np


def occlusion_sensitivity(
    image_01: np.ndarray,
    predict_proba: Callable[[np.ndarray], np.ndarray],
    class_index: int,
    patch: int = 24,
    stride: int = 12,
    fill: float = 0.0,
) -> np.ndarray:
    """Compute a coarse occlusion sensitivity heatmap.

    The heatmap value indicates how much the probability for class_index drops
    when a patch is occluded.
    """

    H, W, C = image_01.shape
    base_p = float(predict_proba(image_01[None, ...])[0, class_index])

    heat = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y1 = min(y + patch, H)
            x1 = min(x + patch, W)
            occluded = image_01.copy()
            occluded[y:y1, x:x1, :] = fill
            p = float(predict_proba(occluded[None, ...])[0, class_index])
            delta = base_p - p
            heat[y:y1, x:x1] += delta
            counts[y:y1, x:x1] += 1.0

    heat = heat / np.maximum(counts, 1.0)
    heat -= heat.min()
    denom = heat.max() + 1e-8
    heat = heat / denom
    return heat


def occlusion_overlay(image_01: np.ndarray, heat_01: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    overlay = image_01.copy().astype("float32")
    overlay[..., 0] = np.clip(overlay[..., 0] + alpha * heat_01, 0, 1)
    return overlay
