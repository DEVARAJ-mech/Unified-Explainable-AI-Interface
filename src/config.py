from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Compatibility:
    models_by_modality: Dict[str, List[str]]
    xai_by_modality: Dict[str, List[str]]


def default_compatibility() -> Compatibility:
    return Compatibility(
        models_by_modality={
            "audio": ["deepfake_mobilenet"],
            "image": ["lung_densenet_demo"],
            "csv": [],
        },
        xai_by_modality={
            "audio": ["lime", "gradcam", "shap", "saliency", "occlusion"],
            "image": ["lime", "gradcam", "shap", "saliency", "occlusion"],
            "csv": ["lime"],
        },
    )
