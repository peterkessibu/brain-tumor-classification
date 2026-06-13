"""
config.py — Single source of truth for all constants, paths, and env setup.
"""
import os
from pathlib import Path
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
SALIENCY_DIR = BASE_DIR / "saliency_maps"
SAMPLES_DIR  = BASE_DIR / "public" / "samples"
CNN_MODEL_PATH     = BASE_DIR / "cnn_model.h5"
XCEPTION_MODEL_PATH = BASE_DIR / "xception_model.weights.h5"

# Ensure output directories exist at import time
SALIENCY_DIR.mkdir(exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# ── Model settings ─────────────────────────────────────────────────────────
IMG_SIZE_CNN      = (224, 224)
IMG_SIZE_XCEPTION = (299, 299)
IMG_SHAPE_XCEPTION = (299, 299, 3)

LABELS = ["Glioma", "Meningioma", "No tumor", "Pituitary"]

# ── Gemini generation config ───────────────────────────────────────────────
GENERATION_CONFIG = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
GEMINI_MODEL_NAME = "gemini-3.1"


def get_genai_model() -> genai.GenerativeModel:
    """
    Initialise and return the Gemini GenerativeModel.
    Calls st.stop() if the API key is missing or init fails,
    so callers never receive None.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found. Please set it in the .env file.")
        st.stop()

    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(
            model_name=GEMINI_MODEL_NAME,
            generation_config=GENERATION_CONFIG,
        )
    except Exception as exc:
        st.error(f"Failed to initialise Generative AI model: {exc}")
        st.stop()
