"""
services/image_service.py — Image loading and pre-processing.

SRP: one responsibility — turn a file (uploaded or path) into a
     normalised numpy array ready for model inference.
"""
from pathlib import Path

import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img


def load_and_preprocess(source, img_size: tuple[int, int]) -> np.ndarray:
    """
    Load an image from an uploaded file object or a filesystem path,
    resize it, and normalise pixel values to [0, 1].

    Parameters
    ----------
    source : UploadedFile | str | Path
        A Streamlit UploadedFile, a string path, or a Path object.
    img_size : tuple[int, int]
        Target (width, height) for resizing.

    Returns
    -------
    np.ndarray
        Array of shape (1, height, width, 3) with values in [0, 1].
    """
    img = load_img(source, target_size=img_size)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = arr / 255.0
    return arr
