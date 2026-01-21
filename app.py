from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st
import tensorflow as tf
import keras

from src.config import default_compatibility
from src.io_utils import audio_to_mel_spectrogram_image, load_image_bytes
from src.models.registry import ModelSpec, build_registry
from src.xai.gradcam import apply_heatmap, gradcam_heatmap
from src.xai.lime_xai import explain_lime
from src.xai.occlusion import occlusion_overlay, occlusion_sensitivity
from src.xai.saliency import saliency_map, saliency_overlay
from src.xai.shap_xai import explain_shap

ASSETS_DIR = str(Path(__file__).parent / "assets")


@st.cache_resource(show_spinner=False)
def load_model_cached(_spec: ModelSpec) -> keras.Model:
    # NOTE: underscore avoids Streamlit hashing ModelSpec
    return _spec.load_fn()


def predict_proba(model: keras.Model, x_01: np.ndarray) -> np.ndarray:
    """Return probabilities for NxHxWxC."""
    x = tf.convert_to_tensor(x_01, dtype=tf.float32)
    preds = model(x)
    preds = np.asarray(preds)

    # Defensive normalization in case the model does not end with softmax.
    if preds.ndim == 2:
        s = preds.sum(axis=1, keepdims=True)
        if np.all(s > 0) and not np.allclose(s, 1.0, atol=1e-3):
            preds = preds / s
    return preds


def top_pred(preds: np.ndarray, class_names: List[str]) -> Tuple[int, float, Dict[str, float]]:
    idx = int(np.argmax(preds))
    prob = float(preds[idx])
    dist = {class_names[i]: float(preds[i]) for i in range(len(class_names))}
    return idx, prob, dist


def _is_wav(name: str) -> bool:
    return Path(name).suffix.lower() == ".wav"


def _is_image(name: str) -> bool:
    return Path(name).suffix.lower() in {".png", ".jpg", ".jpeg"}


def _filter_specs_for_task_and_modality(
    registry: Dict[str, ModelSpec],
    model_keys_for_modality: List[str],
    task: str,
) -> List[ModelSpec]:
    """
    Task filtering without changing ModelSpec:
    - deepfake_audio -> keys that start with 'deepfake'
    - lung_xray      -> keys that start with 'lung'
    """
    if task == "deepfake_audio":
        allowed_prefixes = ("deepfake", "audio", "for")  # tolerant
    elif task == "lung_xray":
        allowed_prefixes = ("lung", "chex", "xray")
    else:
        allowed_prefixes = ()

    specs: List[ModelSpec] = []
    for k in model_keys_for_modality:
        if k not in registry:
            continue
        lk = k.lower()
        if allowed_prefixes and not lk.startswith(allowed_prefixes):
            continue
        specs.append(registry[k])
    return specs


def _sanitize_heatmap_2d(hm: np.ndarray) -> np.ndarray:
    """
    Ensure heatmap is (H, W) for apply_heatmap().

    Handles cases like:
    - (H, W, C)          -> mean over channels
    - (1, H, W)          -> de-batch
    - (1, H, W, C)       -> de-batch then mean over channels
    - any other >2D case -> reduce until 2D
    """
    hm = np.asarray(hm)
    if hm.ndim == 4:
        hm = hm[0]
    if hm.ndim == 3 and hm.shape[0] == 1:
        hm = hm[0]
    if hm.ndim == 3:
        hm = hm.mean(axis=-1)
    while hm.ndim > 2:
        hm = hm.mean(axis=-1)
    return hm


def main() -> None:
    st.set_page_config(
        page_title="Multi-Modal Classification + XAI", layout="wide")

    st.title("Multi-Modal Classification and Explainability")
    st.caption(
        "Single interface refactoring Deepfake Audio Detector and Lung Cancer XAI. "
        "Audio is converted to mel-spectrograms and treated as images for XAI."
    )

    compat = default_compatibility()
    registry = build_registry(ASSETS_DIR)

    TASK_LABEL_TO_KEY = {
        "Deepfake Audio (FoR / spectrogram)": "deepfake_audio",
        "Lung X-Ray (CheXpert)": "lung_xray",
    }

    with st.sidebar:
        st.header("Input")
        uploaded_files = st.file_uploader(
            "Upload audio (.wav) or images (.png/.jpg). You can select multiple files.",
            type=["wav", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )

        st.header("Settings")
        show_debug = st.toggle("Show debug", value=False)

    if not uploaded_files:
        st.info("Upload one or more files to start.")
        st.stop()

    tab_single, tab_compare = st.tabs(["Run", "Compare XAI"])

    def render_xai(
        method: str,
        container,
        *,
        model: keras.Model,
        x_01: np.ndarray,
        class_idx: int,
    ) -> None:
        """Compute + render one XAI method into a Streamlit container."""
        # Common predict function for XAI.
        def proba_fn(batch):
            return predict_proba(model, batch)

        if method == "gradcam":
            with container:
                with st.spinner("Computing Grad-CAM..."):
                    heat = gradcam_heatmap(model, x_01, class_idx)
                    overlay = apply_heatmap(x_01, heat)
                st.write("Grad-CAM")
                st.image((overlay * 255).astype(np.uint8),
                         use_container_width=True)

        elif method == "shap":
            with container:
                with st.spinner("Computing SHAP (GradientExplainer)..."):
                    res = explain_shap(model, x_01, class_idx)
                    hm = _sanitize_heatmap_2d(
                        res.heatmap_01)  # OPTIONAL SAFETY FIX
                    overlay = apply_heatmap(x_01, hm)
                st.write("SHAP (GradientExplainer)")
                st.image((overlay * 255).astype(np.uint8),
                         use_container_width=True)

        elif method == "saliency":
            with container:
                with st.spinner("Computing Saliency..."):
                    sal = saliency_map(model, x_01, class_idx)
                    overlay = saliency_overlay(x_01, sal)
                st.write("Saliency (vanilla gradients)")
                st.image((overlay * 255).astype(np.uint8),
                         use_container_width=True)

        elif method == "occlusion":
            with container:
                with st.spinner("Computing Occlusion sensitivity..."):
                    heat = occlusion_sensitivity(x_01, proba_fn, class_idx)
                    overlay = occlusion_overlay(x_01, heat)
                st.write("Occlusion sensitivity")
                st.image((overlay * 255).astype(np.uint8),
                         use_container_width=True)

        elif method == "lime":
            with container:
                with st.spinner("Computing LIME..."):
                    overlay, _mask = explain_lime(x_01, proba_fn, class_idx)
                st.write("LIME")
                st.image((overlay * 255).astype(np.uint8),
                         use_container_width=True)

        else:
            with container:
                st.error(f"Unknown XAI method: {method}")

    # -----------------------------
    # RUN TAB (single XAI per file)
    # -----------------------------
    with tab_single:
        st.subheader("Run (one model + one XAI per file)")
        st.caption(
            "Multiple uploads are processed independently. "
            "For image inputs, you must specify the dataset/task (X-ray vs spectrogram image) "
            "to avoid mixing models."
        )

        for i, uploaded in enumerate(uploaded_files):
            file_bytes = uploaded.getvalue()
            suffix = Path(uploaded.name).suffix.lower()

            # Determine modality from extension + preprocess
            if suffix == ".wav":
                prep = audio_to_mel_spectrogram_image(file_bytes)
                modality = "audio"
            elif suffix in {".png", ".jpg", ".jpeg"}:
                prep = load_image_bytes(file_bytes)
                modality = "image"
            else:
                st.error(f"[{uploaded.name}] Unsupported file type: {suffix}")
                continue

            widget_prefix = f"run_{i}_{uploaded.name}"

            with st.expander(f"{i+1}. {uploaded.name}", expanded=(i == 0)):
                # Task selection:
                # - WAV => forced deepfake_audio (because it is truly audio)
                # - Image => user chooses deepfake_audio (spectrogram image) vs lung_xray
                if modality == "audio":
                    task_key = "deepfake_audio"
                    task_label = "Deepfake Audio (FoR / spectrogram)"
                    st.selectbox(
                        "Dataset / Task",
                        options=list(TASK_LABEL_TO_KEY.keys()),
                        index=list(TASK_LABEL_TO_KEY.keys()).index(task_label),
                        disabled=True,
                        key=f"{widget_prefix}_task",
                    )
                else:
                    default_label = "Lung X-Ray (CheXpert)"
                    task_label = st.selectbox(
                        "Dataset / Task",
                        options=list(TASK_LABEL_TO_KEY.keys()),
                        index=list(TASK_LABEL_TO_KEY.keys()
                                   ).index(default_label),
                        key=f"{widget_prefix}_task",
                    )
                    task_key = TASK_LABEL_TO_KEY[task_label]

                # Filter models and XAI by modality, then by task (lightweight prefix-based filter)
                model_keys = compat.models_by_modality.get(modality, [])
                xai_keys = compat.xai_by_modality.get(modality, [])

                specs = _filter_specs_for_task_and_modality(
                    registry, model_keys, task_key)
                if not specs:
                    st.error(
                        f"No models configured for modality='{modality}' and task='{task_key}'. "
                        "Check src/config.py or model registry keys."
                    )
                    continue

                # Layout
                col_left, col_right = st.columns([1, 1], gap="large")

                with col_left:
                    st.subheader("Input")
                    st.write(f"Detected modality: **{modality}**")
                    st.write(f"Selected task: **{task_key}**")
                    st.image(prep.display_image, use_container_width=True)
                    if modality == "audio":
                        st.audio(file_bytes, format="audio/wav")

                with col_right:
                    st.subheader("Classification")

                    selected_model_name = st.selectbox(
                        "Model",
                        options=[s.display_name for s in specs],
                        index=0,
                        key=f"{widget_prefix}_model",
                    )
                    selected_spec = next(
                        s for s in specs if s.display_name == selected_model_name)
                    model = load_model_cached(selected_spec)

                    x_01 = selected_spec.preprocess(prep.model_input)
                    preds = predict_proba(model, x_01[None, ...])[0]
                    class_idx, class_prob, dist = top_pred(
                        preds, selected_spec.class_names)

                    st.metric(
                        "Prediction",
                        f"{selected_spec.class_names[class_idx]}",
                        f"{class_prob:.3f}",
                    )
                    st.write("Probability distribution")
                    st.json(dist)

                    if selected_spec.key == "lung_densenet_demo":
                        weights_path = Path(ASSETS_DIR) / \
                            "lung_densenet_demo.weights.h5"
                        if not weights_path.exists():
                            st.warning(
                                "The lung model is running in demo mode (ImageNet backbone + untrained 2-class head). "
                                "To use a trained lung classifier, add weights at assets/lung_densenet_demo.weights.h5."
                            )

                st.subheader("Explainability (XAI)")
                selected_xai = st.selectbox(
                    "Method",
                    options=xai_keys,
                    index=0,
                    key=f"{widget_prefix}_xai",
                )
                st.caption(
                    "XAI methods are filtered by a simple modality-to-method dictionary.")
                render_xai(
                    selected_xai,
                    st.container(),
                    model=model,
                    x_01=x_01,
                    class_idx=class_idx,
                )

                if show_debug:
                    st.subheader("Debug")
                    st.write("Tensor shape", x_01.shape)
                    st.write("Model name", getattr(model, "name", "<unnamed>"))

    # -----------------------------
    # COMPARE TAB (multi XAI per file)
    # -----------------------------
    with tab_compare:
        st.subheader("Compare XAI (multiple methods per file)")
        st.write(
            "For each uploaded file, select a model and multiple XAI methods. "
            "Methods are filtered automatically based on the detected modality."
        )

        for i, uploaded in enumerate(uploaded_files):
            file_bytes = uploaded.getvalue()
            suffix = Path(uploaded.name).suffix.lower()

            if suffix == ".wav":
                prep = audio_to_mel_spectrogram_image(file_bytes)
                modality = "audio"
            elif suffix in {".png", ".jpg", ".jpeg"}:
                prep = load_image_bytes(file_bytes)
                modality = "image"
            else:
                st.error(f"[{uploaded.name}] Unsupported file type: {suffix}")
                continue

            widget_prefix = f"cmp_{i}_{uploaded.name}"

            with st.expander(f"{i+1}. {uploaded.name}", expanded=(i == 0)):
                # Task selection
                if modality == "audio":
                    task_key = "deepfake_audio"
                    task_label = "Deepfake Audio (FoR / spectrogram)"
                    st.selectbox(
                        "Dataset / Task",
                        options=list(TASK_LABEL_TO_KEY.keys()),
                        index=list(TASK_LABEL_TO_KEY.keys()).index(task_label),
                        disabled=True,
                        key=f"{widget_prefix}_task",
                    )
                else:
                    default_label = "Lung X-Ray (CheXpert)"
                    task_label = st.selectbox(
                        "Dataset / Task",
                        options=list(TASK_LABEL_TO_KEY.keys()),
                        index=list(TASK_LABEL_TO_KEY.keys()
                                   ).index(default_label),
                        key=f"{widget_prefix}_task",
                    )
                    task_key = TASK_LABEL_TO_KEY[task_label]

                model_keys = compat.models_by_modality.get(modality, [])
                xai_keys = compat.xai_by_modality.get(modality, [])

                specs = _filter_specs_for_task_and_modality(
                    registry, model_keys, task_key)
                if not specs:
                    st.error(
                        f"No models configured for modality='{modality}' and task='{task_key}'. "
                        "Check src/config.py or model registry keys."
                    )
                    continue

                col_left, col_right = st.columns([1, 1], gap="large")

                with col_left:
                    st.subheader("Input")
                    st.write(f"Detected modality: **{modality}**")
                    st.write(f"Selected task: **{task_key}**")
                    st.image(prep.display_image, use_container_width=True)
                    if modality == "audio":
                        st.audio(file_bytes, format="audio/wav")

                with col_right:
                    st.subheader("Classification")

                    selected_model_name = st.selectbox(
                        "Model",
                        options=[s.display_name for s in specs],
                        index=0,
                        key=f"{widget_prefix}_model",
                    )
                    selected_spec = next(
                        s for s in specs if s.display_name == selected_model_name)
                    model = load_model_cached(selected_spec)

                    x_01 = selected_spec.preprocess(prep.model_input)
                    preds = predict_proba(model, x_01[None, ...])[0]
                    class_idx, class_prob, dist = top_pred(
                        preds, selected_spec.class_names)

                    st.metric(
                        "Prediction",
                        f"{selected_spec.class_names[class_idx]}",
                        f"{class_prob:.3f}",
                    )
                    st.write("Probability distribution")
                    st.json(dist)

                st.subheader("Comparison")
                selected_methods = st.multiselect(
                    "Methods to compare",
                    options=xai_keys,
                    default=[m for m in ["gradcam", "shap", "lime"]
                             if m in xai_keys],
                    key=f"{widget_prefix}_methods",
                )

                if not selected_methods:
                    st.info("Select at least one method to compare.")
                else:
                    cols = st.columns(len(selected_methods))
                    for c, m in zip(cols, selected_methods):
                        render_xai(
                            m,
                            c,
                            model=model,
                            x_01=x_01,
                            class_idx=class_idx,
                        )

                if show_debug:
                    st.subheader("Debug")
                    st.write("Tensor shape", x_01.shape)
                    st.write("Model name", getattr(model, "name", "<unnamed>"))


if __name__ == "__main__":
    main()
