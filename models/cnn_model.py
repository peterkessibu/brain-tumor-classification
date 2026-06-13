"""
models/cnn_model.py — Custom CNN model loader.

SRP: this module's only job is to load and run the CNN.
"""
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np

from config import CNN_MODEL_PATH, IMG_SIZE_CNN
from models.base_model import BrainTumorModel


class CNNModel(BrainTumorModel):
    """Wraps the custom CNN (.h5) trained for brain-tumour classification."""

    img_size: tuple[int, int] = IMG_SIZE_CNN

    def __init__(self) -> None:
        self._model = None

    def load(self) -> None:
        if not CNN_MODEL_PATH.exists():
            st.error(f"CNN model file not found: {CNN_MODEL_PATH}")
            st.stop()
        try:
            self._model = load_model(str(CNN_MODEL_PATH))
        except Exception as exc:
            st.error(f"Error loading Custom CNN model: {exc}")
            st.stop()

    def predict(self, img_array: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("CNNModel.load() must be called before predict().")
        return self._model.predict(img_array)
