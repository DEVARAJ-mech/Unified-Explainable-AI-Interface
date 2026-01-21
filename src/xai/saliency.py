from __future__ import annotations

import numpy as np
import tensorflow as tf


def saliency_map(model: tf.keras.Model, image_01: np.ndarray, class_index: int) -> np.ndarray:
    """Vanilla gradient saliency (max over channels), returned as HxW in [0,1]."""

    x = tf.convert_to_tensor(image_01[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        preds = model(x)
        loss = preds[:, class_index]

    grads = tape.gradient(loss, x)[0]  # HxWxC
    sal = tf.reduce_max(tf.abs(grads), axis=-1)
    sal = sal - tf.reduce_min(sal)
    sal = sal / (tf.reduce_max(sal) + 1e-8)
    return sal.numpy().astype("float32")


def saliency_overlay(image_01: np.ndarray, saliency_01: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay saliency into red channel; returns HxWx3 float32 in [0,1]."""

    overlay = image_01.copy().astype("float32")
    overlay[..., 0] = np.clip(overlay[..., 0] + alpha * saliency_01, 0, 1)
    return overlay
