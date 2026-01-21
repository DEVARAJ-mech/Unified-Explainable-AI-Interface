from __future__ import annotations

from pathlib import Path

import tensorflow as tf


def load_deepfake_model(assets_dir: str) -> tf.keras.Model:
    """Load the provided SavedModel for deepfake audio detection.

    The original project stored the model under Streamlit/saved_model/model.
    In this merged repo we keep it under assets/saved_model/model.
    """

    model_path = Path(assets_dir) / "saved_model" / "model"
    if not model_path.exists():
        raise FileNotFoundError(f"Deepfake model not found at: {model_path}")

    model = tf.keras.models.load_model(str(model_path))
    return model
