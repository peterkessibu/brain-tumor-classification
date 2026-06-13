"""
ui/results.py — Classification result rendering components.

SRP: renders the result card, saliency comparison, and probability chart.
     No business logic lives here.
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from config import LABELS


def render_image_comparison(input_source, saliency_map_path: str, is_upload: bool) -> None:
    """Side-by-side display of the original image and its saliency map."""
    col1, col2 = st.columns(2)
    with col1:
        caption = "Uploaded Image" if is_upload else "Selected Sample"
        st.image(input_source, caption=caption, use_column_width=True)
    with col2:
        st.image(saliency_map_path, caption="Saliency Map", use_column_width=True)


def render_result_card(result: str, confidence: float) -> None:
    """Render the black card showing predicted class and confidence."""
    st.write("## Classification Results")
    st.markdown(
        f"""
        <div style="background-color:#000000;color:#ffffff;padding:30px;border-radius:15px;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div style="flex:1;text-align:center;">
                    <h3 style="color:#ffffff;margin-bottom:10px;font-size:20px;">Prediction</h3>
                    <p style="font-size:36px;font-weight:800;color:#FF0000;margin:0;">{result}</p>
                </div>
                <div style="width:2px;height:80px;background-color:#ffffff;margin:0 20px;"></div>
                <div style="flex:1;text-align:center;">
                    <h3 style="color:#ffffff;margin-bottom:10px;font-size:20px;">Confidence</h3>
                    <p style="font-size:36px;font-weight:800;color:#2196F3;margin:0;">{confidence:.4%}</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_chart(probabilities: np.ndarray, predicted_label: str) -> None:
    """Render a horizontal bar chart of per-class probabilities."""
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_labels = [LABELS[i] for i in sorted_indices]
    sorted_probs  = probabilities[sorted_indices]

    fig = go.Figure(
        go.Bar(
            x=sorted_probs,
            y=sorted_labels,
            orientation="h",
            marker_color=[
                "red" if label == predicted_label else "blue"
                for label in sorted_labels
            ],
        )
    )
    fig.update_layout(
        title="Probabilities for each class",
        xaxis_title="Probability",
        yaxis_title="Class",
        height=400,
        width=500,
        yaxis=dict(autorange="reversed"),
    )
    for i, prob in enumerate(sorted_probs):
        fig.add_annotation(
            x=prob, y=i,
            text=f"{prob:.4%}",
            showarrow=False,
            xanchor="left",
            xshift=5,
        )
    st.plotly_chart(fig)
