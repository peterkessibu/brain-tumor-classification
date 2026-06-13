"""
services/saliency_service.py — Gradient-based saliency map generation.

SRP: sole responsibility is to compute and save saliency maps.
DRY: path resolution for uploads vs. samples is unified in one helper.
"""
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from config import SALIENCY_DIR


def _resolve_output_path(input_file, is_upload: bool) -> Path:
    """
    Determine the saliency-map save path and, for uploads, persist
    the raw file to disk so OpenCV can read it back later.

    Parameters
    ----------
    input_file : UploadedFile | str | Path
    is_upload  : bool

    Returns
    -------
    Path
        Absolute path where the saliency map will be written.
    """
    if is_upload:
        filename = input_file.name
        dest = SALIENCY_DIR / filename
        dest.write_bytes(input_file.getbuffer())
    else:
        filename = Path(input_file).name

    return SALIENCY_DIR / filename


def generate_saliency_map(
    model,
    img_array: np.ndarray,
    class_index: int,
    img_size: tuple[int, int],
    input_file,
    is_upload: bool = True,
) -> str | None:
    """
    Compute a gradient saliency map and save it as an image.

    Returns
    -------
    str | None
        String path to the saved saliency-map image, or None on failure.
    """
    try:
        # ── Gradient computation ───────────────────────────────────────────
        with tf.GradientTape() as tape:
            img_tensor = tf.convert_to_tensor(img_array)
            tape.watch(img_tensor)
            predictions = model(img_tensor)
            target_class = predictions[:, class_index]

        gradients = tape.gradient(target_class, img_tensor)
        gradients = tf.math.abs(gradients)
        gradients = tf.reduce_max(gradients, axis=-1)
        gradients = gradients.numpy().squeeze()
        gradients = cv2.resize(gradients, img_size)

        # ── Circular brain mask ────────────────────────────────────────────
        center = (gradients.shape[0] // 2, gradients.shape[1] // 2)
        radius = min(center) - 10
        y, x = np.ogrid[: gradients.shape[0], : gradients.shape[1]]
        mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2

        gradients = gradients * mask

        brain_grads = gradients[mask]
        if brain_grads.max() > brain_grads.min():
            brain_grads = (brain_grads - brain_grads.min()) / (
                brain_grads.max() - brain_grads.min()
            )
        gradients[mask] = brain_grads

        # ── Threshold + blur ───────────────────────────────────────────────
        threshold = np.percentile(gradients[mask], 80)
        gradients[gradients < threshold] = 0
        gradients = cv2.GaussianBlur(gradients, (11, 11), 0)

        # ── Heatmap overlay ────────────────────────────────────────────────
        heatmap = cv2.applyColorMap(np.uint8(255 * gradients), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = cv2.resize(heatmap, img_size)

        original_img = img_to_array(load_img(input_file, target_size=img_size))
        superimposed = (heatmap * 0.7 + original_img * 0.3).astype(np.uint8)

        # ── Persist ────────────────────────────────────────────────────────
        saliency_path = _resolve_output_path(input_file, is_upload)
        cv2.imwrite(str(saliency_path), cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))

        return str(saliency_path)

    except Exception as exc:
        st.error(f"Error generating saliency map: {exc}")
        return None
