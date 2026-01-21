from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PreprocessedInput:
    modality: str  # 'audio'|'image'|'csv'
    display_image: Image.Image  # what we show in UI
    model_input: np.ndarray  # HxWxC float32 in [0,1]


def ensure_dir(path: str | os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_image_bytes(file_bytes: bytes, target_size: Tuple[int, int] = (224, 224)) -> PreprocessedInput:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    disp = img.copy()
    img = img.resize(target_size)
    arr = np.asarray(img).astype("float32") / 255.0
    return PreprocessedInput(modality="image", display_image=disp, model_input=arr)


def audio_to_mel_spectrogram_image(
    wav_bytes: bytes,
    target_size: Tuple[int, int] = (224, 224),
    sr: int | None = None,
    n_mels: int = 128,
) -> PreprocessedInput:
    """Convert audio bytes to a mel spectrogram and return as image-like tensor."""

    # librosa expects a path-like or file-like. We use soundfile via librosa.load on BytesIO.
    y, sr_out = librosa.load(io.BytesIO(wav_bytes), sr=sr)

    ms = librosa.feature.melspectrogram(y=y, sr=sr_out, n_mels=n_mels)
    log_ms = librosa.power_to_db(ms, ref=np.max)

    # Render to an RGB image using matplotlib (kept deterministic for reproducibility).
    fig = plt.figure(figsize=(3, 3), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    librosa.display.specshow(log_ms, sr=sr_out, x_axis=None, y_axis=None, ax=ax)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert("RGB")
    disp = img.copy()
    img = img.resize(target_size)
    arr = np.asarray(img).astype("float32") / 255.0

    return PreprocessedInput(modality="audio", display_image=disp, model_input=arr)
