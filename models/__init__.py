"""models/__init__.py"""
from .cnn_model import CNNModel
from .xception_model import XceptionModel

MODEL_REGISTRY = {
    "Custom CNN": CNNModel,
    "Transfer Learning - Xception": XceptionModel,
}
