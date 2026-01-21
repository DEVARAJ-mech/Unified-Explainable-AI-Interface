import numpy as np
import tensorflow as tf
import keras


def apply_heatmap(image_01: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Blend a heatmap onto an image in [0,1].

    Accepts heatmap shapes:
      - (H, W)
      - (H, W, C)  -> reduced to (H, W)
      - (1, H, W) / (1, H, W, C) -> de-batched then reduced
      - Any other 3D -> reduced on last axis (defensive)
    Also resizes to image spatial size if needed.
    """
    img = np.asarray(image_01, dtype=np.float32)
    hm = np.asarray(heatmap, dtype=np.float32)

    # De-batch if needed
    if hm.ndim == 4:
        hm = hm[0]
    if hm.ndim == 3 and hm.shape[0] == 1:
        hm = hm[0]

    # Reduce channels if needed
    if hm.ndim == 3:
        hm = hm.mean(axis=-1)

    # Ensure 2D
    while hm.ndim > 2:
        hm = hm.mean(axis=-1)

    H, W = img.shape[:2]

    # Resize heatmap if needed
    if hm.shape != (H, W):
        hm_tf = tf.convert_to_tensor(hm[..., None], dtype=tf.float32)
        hm_tf = tf.image.resize(hm_tf, (H, W), method="bilinear")
        hm = tf.squeeze(hm_tf, axis=-1).numpy()

    # Normalize heatmap to [0,1]
    hm_min, hm_max = float(np.min(hm)), float(np.max(hm))
    if hm_max > hm_min:
        hm = (hm - hm_min) / (hm_max - hm_min)
    else:
        hm = np.zeros((H, W), dtype=np.float32)

    # Ensure image is 3-channel
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)

    overlay = img.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + alpha * hm, 0, 1)
    overlay[..., 1] = np.clip(overlay[..., 1] + alpha * hm, 0, 1)
    overlay[..., 2] = np.clip(overlay[..., 2] + alpha * hm, 0, 1)
    return overlay


def _find_last_conv_layer_name(model: keras.Model) -> str:
    """Find the last convolution-like layer name."""
    for layer in reversed(model.layers):
        cls = layer.__class__.__name__.lower()
        if "conv" in cls:
            return layer.name
    raise ValueError(
        "Grad-CAM: could not find a convolutional layer in the model.")


def gradcam_heatmap(
    model: keras.Model,
    image_01: np.ndarray,
    class_idx: int,
    last_conv_layer_name: str | None = None,
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for a single image (H,W,C) in [0,1].

    Returns:
      heatmap: (H, W) float32 in [0,1]
    """
    if image_01.ndim != 3:
        raise ValueError(
            f"gradcam_heatmap expects (H,W,C) input, got shape={image_01.shape}")

    if last_conv_layer_name is None:
        last_conv_layer_name = _find_last_conv_layer_name(model)

    last_conv_layer = model.get_layer(last_conv_layer_name)

    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output],
    )

    x = tf.convert_to_tensor(image_01[None, ...], dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        score = preds[:, class_idx]

    grads = tape.gradient(score, conv_out)
    if grads is None:
        raise ValueError(
            "Grad-CAM: gradients are None. This can happen if the selected conv layer "
            "is not connected to the output, or the graph is non-differentiable."
        )

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (channels,)
    conv_out = conv_out[0]  # (h,w,channels)

    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)  # (h,w)
    heatmap = tf.nn.relu(heatmap)

    max_val = tf.reduce_max(heatmap)
    heatmap = tf.where(max_val > 0, heatmap / max_val, tf.zeros_like(heatmap))
    heatmap = heatmap.numpy()

    # Resize to input size
    H, W = image_01.shape[:2]
    heatmap_tf = tf.convert_to_tensor(heatmap[..., None], dtype=tf.float32)
    heatmap_tf = tf.image.resize(heatmap_tf, (H, W), method="bilinear")
    heatmap = tf.squeeze(heatmap_tf, axis=-1).numpy()

    return np.clip(heatmap, 0.0, 1.0).astype(np.float32)
