from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries


def explain_lime(
    image_01: np.ndarray,
    predict_proba: Callable[[np.ndarray], np.ndarray],
    class_index: int,
    num_samples: int = 1000,
    num_features: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (overlay_image, mask) for LIME.

    image_01: HxWxC float32 in [0,1]
    predict_proba: function that accepts NxHxWxC in [0,1] and returns NxK probabilities
    """

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_01.astype("float64"),
        classifier_fn=predict_proba,
        hide_color=0,
        num_samples=num_samples,
    )

    temp, mask = explanation.get_image_and_mask(
        label=class_index,
        positive_only=False,
        num_features=num_features,
        hide_rest=True,
    )

    overlay = mark_boundaries(temp, mask)
    return overlay, mask
