"""
models/xception_model.py — Transfer-learning Xception model loader.

SRP: this module's only job is to build, load weights for, and run Xception.
"""
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
import numpy as np

from config import XCEPTION_MODEL_PATH, IMG_SIZE_XCEPTION, IMG_SHAPE_XCEPTION
from models.base_model import BrainTumorModel


class XceptionModel(BrainTumorModel):
    """Wraps the Xception transfer-learning model (.weights.h5)."""

    img_size: tuple[int, int] = IMG_SIZE_XCEPTION

    def __init__(self) -> None:
        self._model = None

    def load(self) -> None:
        if not XCEPTION_MODEL_PATH.exists():
            st.error(f"Xception model file not found: {XCEPTION_MODEL_PATH}")
            st.stop()
        try:
            base = tf.keras.applications.Xception(
                include_top=False,
                weights="imagenet",
                input_shape=IMG_SHAPE_XCEPTION,
                pooling="max",
            )
            model = Sequential([
                base,
                Flatten(),
                Dropout(rate=0.3),
                Dense(128, activation="relu"),
                Dropout(rate=0.25),
                Dense(4, activation="softmax"),
            ])
            model.build((None,) + IMG_SHAPE_XCEPTION)
            model.compile(
                Adamax(learning_rate=0.001),
                loss="categorical_crossentropy",
                metrics=["accuracy", Precision(), Recall()],
            )
            model.load_weights(str(XCEPTION_MODEL_PATH))
            self._model = model
        except Exception as exc:
            st.error(f"Error loading Xception model: {exc}")
            st.stop()

    def predict(self, img_array: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("XceptionModel.load() must be called before predict().")
        return self._model.predict(img_array)
