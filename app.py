"""
app.py — Streamlit entry point.

This file is intentionally thin: it only wires together the modules
from config/, models/, services/, and ui/. No business logic lives here.
"""
import streamlit as st

from config import LABELS, get_genai_model
from models import MODEL_REGISTRY
from services.image_service import load_and_preprocess
from services.saliency_service import generate_saliency_map
from services.explanation_service import ExplanationService
from ui.gallery import render_gallery
from ui.results import render_image_comparison, render_result_card, render_probability_chart

import numpy as np

# ── Page setup ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Brain Tumor Classification", layout="wide")
st.title("Brain Tumor Classification")

# ── Dependency injection ────────────────────────────────────────────────────
# get_genai_model() handles error/stop internally; ExplanationService
# receives the model object — it never reaches out to global state.
explanation_service = ExplanationService(genai_model=get_genai_model())

# ── Sample gallery ─────────────────────────────────────────────────────────
render_gallery()

if st.session_state.get("selected_image"):
    st.success(
        f"Analysing selected image: "
        f"{st.session_state.selected_image.split('/')[-1].split(chr(92))[-1]}. "
        "Please wait for results before selecting another image."
    )

# ── File upload ────────────────────────────────────────────────────────────
st.write("Upload an image of a brain MRI scan to classify.")
uploaded_file = st.file_uploader("Choose an image…", type=["jpg", "jpeg", "png"])

# ── Main pipeline ──────────────────────────────────────────────────────────
selected_sample = st.session_state.get("selected_image")
input_image = uploaded_file if uploaded_file is not None else selected_sample

if input_image is None:
    st.stop()

is_upload = uploaded_file is not None

# ── Model selection ────────────────────────────────────────────────────────
selected_model_name = st.radio("Select Model", list(MODEL_REGISTRY.keys()))
model = MODEL_REGISTRY[selected_model_name]()
model.load()

# ── Pre-process image ──────────────────────────────────────────────────────
try:
    img_array = load_and_preprocess(input_image, model.img_size)
except Exception as exc:
    st.error(f"Error processing image: {exc}")
    st.stop()

# ── Inference ──────────────────────────────────────────────────────────────
with st.spinner("Analysing image…"):
    try:
        prediction = model.predict(img_array)
    except Exception as exc:
        st.error(f"Error during model prediction: {exc}")
        st.stop()

class_index = int(np.argmax(prediction[0]))
result      = LABELS[class_index]
confidence  = float(prediction[0][class_index])

st.write(f"**Predicted Class**: {result}")
st.write("**Predictions:**")
for label, prob in zip(LABELS, prediction[0]):
    st.write(f"{label}: {prob:.4f}")

# ── Saliency map ───────────────────────────────────────────────────────────
saliency_map_path = generate_saliency_map(
    model=model._model,          # underlying keras model needed for GradientTape
    img_array=img_array,
    class_index=class_index,
    img_size=model.img_size,
    input_file=input_image,
    is_upload=is_upload,
)

if saliency_map_path is None:
    st.error("Saliency map generation failed.")
    st.stop()

# ── Results UI ─────────────────────────────────────────────────────────────
render_image_comparison(input_image, saliency_map_path, is_upload)
render_result_card(result, confidence)
render_probability_chart(prediction[0], result)

# ── AI explanation ─────────────────────────────────────────────────────────
with st.spinner("Generating expert explanation…"):
    explanation = explanation_service.explain(saliency_map_path, result, confidence)

if explanation.startswith("Error"):
    st.error(explanation)
else:
    st.write("## Explanation:")
    st.write(explanation)