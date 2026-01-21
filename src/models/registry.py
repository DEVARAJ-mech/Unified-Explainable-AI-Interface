from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    modality: str  # 'audio'|'image'|'csv'
    num_classes: int
    class_names: List[str]
    load_fn: Callable[[], tf.keras.Model]
    preprocess: Callable[[np.ndarray], np.ndarray]


def _identity(x: np.ndarray) -> np.ndarray:
    return x


def build_registry(assets_dir: str) -> Dict[str, ModelSpec]:
    from .deepfake_mobilenet import load_deepfake_model
    from .lung_densenet_demo import load_lung_demo_model

    return {
        "deepfake_mobilenet": ModelSpec(
            key="deepfake_mobilenet",
            display_name="Deepfake Audio (MobileNet on Mel-Spectrogram)",
            modality="audio",
            num_classes=2,
            class_names=["real", "fake"],
            load_fn=lambda: load_deepfake_model(assets_dir),
            preprocess=_identity,
        ),
        "lung_densenet_demo": ModelSpec(
            key="lung_densenet_demo",
            display_name="Lung X-Ray (DenseNet121 demo)",
            modality="image",
            num_classes=2,
            class_names=["benign", "malignant"],
            load_fn=lambda: load_lung_demo_model(assets_dir),
            preprocess=_identity,
        ),
    }
