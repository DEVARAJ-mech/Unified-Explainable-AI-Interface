from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf


@dataclass
class ShapResult:
    """Container for SHAP explanation outputs."""

    heatmap_01: np.ndarray  # HxW in [0, 1]
    raw_values: np.ndarray  # HxWxC (signed)


def _safe_import_shap():
    try:
        import shap  # type: ignore

        return shap
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "SHAP is not available. Install dependencies with: pip install -r requirements.txt "
            "(ensuring the 'shap' package installs correctly)."
        ) from e


def _normalize_01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - np.min(x)
    denom = np.max(x) + eps
    return x / denom


def explain_shap(
    model: tf.keras.Model,
    x_01: np.ndarray,
    class_index: int,
    background: Optional[np.ndarray] = None,
    nsamples: int = 50,
) -> ShapResult:
    """Compute a simple SHAP-based attribution heatmap for an image-like input.

    Notes:
    - This uses SHAP's GradientExplainer which works for TF/Keras models.
    - A minimal background is used by default (all-zeros image). For better results,
      provide a small batch of representative background samples.

    Args:
        model: TF/Keras model returning class probabilities or logits.
        x_01: HxWxC float32 array in [0, 1].
        class_index: target class for attribution.
        background: optional background batch BxHxWxC.
        nsamples: sampling budget for the explainer.

    Returns:
        ShapResult with a normalized HxW heatmap and raw signed SHAP values.
    """

    shap = _safe_import_shap()

    x = np.asarray(x_01, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected HxWxC input, got shape={x.shape}")

    xin = x[None, ...]

    if background is None:
        background = np.zeros_like(xin)
    else:
        background = np.asarray(background, dtype=np.float32)
        if background.ndim != 4:
            raise ValueError("background must be a 4D batch BxHxWxC")

    # Ensure model is callable in eager mode.
    _ = model(tf.convert_to_tensor(background[:1]))

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(xin, nsamples=nsamples)

    # shap_values is either a list (per class) or a single array.
    if isinstance(shap_values, list):
        if class_index < 0 or class_index >= len(shap_values):
            class_index = int(np.argmax(model(xin).numpy()[0]))
        sv = np.asarray(shap_values[class_index][0], dtype=np.float32)
    else:
        sv = np.asarray(shap_values[0], dtype=np.float32)

    # Aggregate channel attributions to a 2D heatmap.
    heat = np.mean(np.abs(sv), axis=-1)
    heat_01 = _normalize_01(heat)

    return ShapResult(heatmap_01=heat_01, raw_values=sv)
