"""
models/base_model.py — Abstract contract for all brain-tumor classifiers.

OCP: adding a new model = new file only, zero changes to app.py.
LSP: every subclass is a drop-in replacement.
"""
from abc import ABC, abstractmethod
import numpy as np


class BrainTumorModel(ABC):
    """Abstract base class every model implementation must satisfy."""

    @property
    @abstractmethod
    def img_size(self) -> tuple[int, int]:
        """Return (width, height) expected by this model."""
        ...

    @abstractmethod
    def load(self) -> None:
        """Load weights from disk into memory."""
        ...

    @abstractmethod
    def predict(self, img_array: np.ndarray) -> np.ndarray:
        """
        Run inference.

        Parameters
        ----------
        img_array : np.ndarray
            Pre-processed image array with shape (1, H, W, 3), values in [0, 1].

        Returns
        -------
        np.ndarray
            Raw probability array of shape (1, num_classes).
        """
        ...
