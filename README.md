# Unified Multi-Modal Classification + XAI (Audio + Image)

This project refactors and combines two prior works into **one Streamlit interface**:

- **Deepfake Audio Detection with XAI** (audio → mel-spectrogram → image classifier)
- **Lung Cancer XAI** (chest X-ray classification + explanations)

The unified app can:

- Accept **.wav** audio files (converted to mel-spectrogram images)
- Accept **.png/.jpg** image files
- Run a **configurable set of models** per modality
- Run multiple **XAI methods** (**Grad-CAM, LIME, SHAP** required; plus Saliency and Occlusion) filtered by a **simple compatibility dictionary**
- Provide a dedicated **comparison tab** to view multiple explanations side-by-side

> Note: Training new models is optional. This repo ships the original deepfake SavedModel. The lung model is a **demo** classifier unless you provide trained weights.

---

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py
```

---

## Team

Fill this section before submission:

- **TD group:** (e.g., CDOF1)
- **Members:**
  - Name 1
  - Name 2
  - ...

---

## What was refactored

### Single interface
- One Streamlit app (`app.py`) with a unified upload flow and shared XAI pipeline.

### Modular code
- `src/io_utils.py`: input loading + audio→spectrogram conversion
- `src/models/*`: model loading (deepfake model + lung demo model)
- `src/xai/*`: explanation methods (Grad-CAM, LIME, Saliency, Occlusion)
- `src/config.py`: compatibility dictionary used to filter available models and XAI methods

---

## Compatibility filtering (as requested)

The compatibility checks are **front-end filters only** (no heavy validation). They exist to ensure the UI only offers XAI methods that make sense for the input modality.

See `src/config.py`:

- For **audio**: currently treated as an **image** (mel-spectrogram), so all image XAI methods are available
- For **image**: all the included image XAI methods are available
- For **csv**: placeholder entry (easy to extend later)

To extend:
1. Add a new model loader under `src/models/`
2. Register it in `src/models/registry.py`
3. Update modality compatibility in `src/config.py`

---

## Models

### 1) Deepfake audio model (provided)
The original SavedModel is included at:

```
assets/saved_model/model
```

The app uses it when the input is `.wav`.

### 2) Lung X-ray model (demo by default)
`src/models/lung_densenet_demo.py` builds a DenseNet121 backbone (ImageNet weights) with a 2-class head:

- Classes: `benign`, `malignant`

To use a real lung cancer classifier, provide trained weights at:

```
assets/lung_densenet_demo.weights.h5
```

The app will auto-load them at startup.

---

## XAI methods

- **Grad-CAM**: last convolutional layer auto-detected
- **LIME**: superpixel-based explanation over model probabilities
- **SHAP**: `shap.GradientExplainer` heatmap aggregated over channels
- **Saliency**: vanilla gradients w.r.t. the predicted class
- **Occlusion**: patch-based probability drop mapping

All methods operate on **HxWx3 float images in [0,1]**.

---

## Project structure

```
Unified-MultiModal-XAI/
  app.py
  requirements.txt
  assets/
    saved_model/                       # deepfake model
    DEEPFAKE_README.md
    LUNG_README.md
  src/
    config.py
    io_utils.py
    models/
      registry.py
      deepfake_mobilenet.py
      lung_densenet_demo.py
    xai/
      gradcam.py
      lime_xai.py
      shap_xai.py
      saliency.py
      occlusion.py
```

---

## Notes / limitations

- The lung classifier is **not trained** in this repository (by design, per requirements). It is a plug-in point for your trained weights.
- Audio is treated as an image after mel-spectrogram conversion, so XAI compatibility imposes **no constraints** at present.

---

## Generative AI Usage Statement

This project may be refactored and extended using Generative AI tools.

Declare the following in your final submission (edit to match your team’s real usage):

- **Tools/models used:** ChatGPT (OpenAI)
- **Purpose:** repository refactoring, integration design, UI implementation (Streamlit), and documentation drafting


