# Short Technical Report

> Template — replace placeholders with your project-specific details.

## 1. Project overview
This project integrates two prior XAI codebases into a single interface that supports:
- Audio deepfake detection on `.wav` files
- Chest X-ray (lung cancer) classification on images
- Multiple XAI methods with an explicit comparison workflow

## 2. Key design and integration decisions
### 2.1 Unified data handling
- **Audio:** waveform → mel-spectrogram → 3-channel image tensor (HxWx3, normalized to [0,1])
- **Images:** resized and normalized to the same tensor contract

Rationale: a shared tensor contract enables reuse of image-based explainability methods across modalities.

### 2.2 Model registry
Models are exposed via a small registry (`src/models/registry.py`) that defines for each model:
- Display name
- Loader function
- Preprocessing function
- Class names

Rationale: adding a new model requires minimal code changes and no UI rewiring.

### 2.3 XAI module layout
XAI methods are implemented as separate modules under `src/xai/` and take:
- the preprocessed input tensor
- a prediction function (for perturbation-based methods)
- the target class index

## 3. Models included
### 3.1 Deepfake audio detection
- Model: SavedModel shipped in `assets/saved_model/model`
- Output: binary classification (e.g., real vs fake)

### 3.2 Lung X-ray detection
- Model: DenseNet-based demo implementation, with an explicit hook for adding trained weights
- Weights path: `assets/lung_densenet_demo.weights.h5`

## 4. XAI methods included
Required:
- **Grad-CAM**
- **LIME**
- **SHAP** (GradientExplainer)

Additional:
- Saliency (vanilla gradients)
- Occlusion sensitivity

## 5. Compatibility filtering
The UI filters XAI methods based on modality using a simple dictionary (`src/config.py`).
- The check is intentionally lightweight and “anticipatory” for future modalities (e.g., CSV).
- Current implementation treats audio as image-like after spectrogram conversion, so the filter is non-restrictive for audio.

## 6. Improvements over the original repositories
Examples (replace with what you actually did):
- Unified Streamlit UI with consistent user flow
- Shared preprocessing and standardized tensor contract
- Central model registry and pluggable XAI modules
- Dedicated comparison tab to visualize multiple XAI outputs side-by-side

## 7. Limitations and future work
- Provide trained lung model weights (or integrate additional pretrained checkpoints)
- Add modality-specific XAI methods (e.g., waveform-specific audio attribution)
- Add dataset handling beyond single-file input (folders, CSV, batch inference)
