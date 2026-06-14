"""
config.py — Single source of truth for all constants, paths, and env setup.
"""
import os
from pathlib import Path
import streamlit as st
from google import genai
from google.genai import types
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
GEMINI_MODEL_NAME = "gemini-flash-latest"


def get_generate_content_config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level="HIGH",
        ),
        tools=[
            types.Tool(google_search=types.GoogleSearch()),
        ],
    )


def get_genai_client() -> genai.Client:
    """
    Initialise and return the Gemini Client.
    Calls st.stop() if the API key is missing or init fails,
    so callers never receive None.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        api_key = st.secrets.get("GEMINI_API_KEY", None)

    if not api_key:
        st.error("GEMINI_API_KEY not found. Please set it in the .env file or Streamlit secrets.")
        st.stop()

    try:
        return genai.Client(api_key=api_key)
    except Exception as exc:
        st.error(f"Failed to initialise Generative AI client: {exc}")
        st.stop()
