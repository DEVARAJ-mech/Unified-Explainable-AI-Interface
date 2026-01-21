from __future__ import annotations

from pathlib import Path

import tensorflow as tf


def _build_demo_model() -> tf.keras.Model:
    base = tf.keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling="avg",
    )
    x = tf.keras.layers.Dropout(0.2)(base.output)
    out = tf.keras.layers.Dense(2, activation="softmax", name="lung_head")(x)
    model = tf.keras.Model(base.input, out, name="lung_densenet_demo")
    # The head is not trained in this repo. If you provide trained weights, they will be loaded.
    return model


def load_lung_demo_model(assets_dir: str) -> tf.keras.Model:
    """Load a lung X-ray classifier.

    If you have trained weights, place them at:
      assets/lung_densenet_demo.weights.h5

    Otherwise, the app falls back to an ImageNet-pretrained DenseNet backbone
    with an untrained 2-class head (demo-only).
    """

    weights_path = Path(assets_dir) / "lung_densenet_demo.weights.h5"
    model = _build_demo_model()
    if weights_path.exists():
        model.load_weights(str(weights_path))
    return model
