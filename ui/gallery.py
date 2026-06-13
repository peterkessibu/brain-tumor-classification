"""
ui/gallery.py — Sample MRI scan gallery component.

SRP: sole responsibility is rendering the image gallery and returning
     the user's selection via st.session_state.
"""
from pathlib import Path

import PIL.Image
import streamlit as st

from config import SAMPLES_DIR


def render_gallery() -> str | None:
    """
    Display a 4-column gallery of sample MRI scans.

    Initialises ``st.session_state.selected_image`` on first call.
    Sets it when the user clicks a thumbnail button.

    Returns
    -------
    str | None
        Path of the currently selected sample image, or None.
    """
    if "selected_image" not in st.session_state:
        st.session_state.selected_image = None

    st.write("## Sample MRI Scans")
    st.write("Click on any sample image to analyse it, or upload your own below.")

    sample_images = _load_sample_images()
    if not sample_images:
        st.warning(f"No sample images found in {SAMPLES_DIR}.")
        return None

    cols = st.columns(4)
    for idx, img_data in enumerate(sample_images):
        col = cols[idx % 4]
        with col:
            try:
                img = PIL.Image.open(img_data["path"])
                img.thumbnail((200, 200))

                if st.button(
                    label="",
                    key=f"sample_{idx}",
                    help=f"Click to analyse {img_data['name']}",
                ):
                    st.session_state.selected_image = img_data["path"]

                st.image(
                    img,
                    caption=Path(img_data["name"]).stem,
                    use_column_width=True,
                )
            except Exception as exc:
                st.error(f"Error loading image {img_data['name']}: {exc}")

    return st.session_state.selected_image


# ── Private helpers ────────────────────────────────────────────────────────

def _load_sample_images() -> list[dict]:
    """Return a list of {path, name} dicts for every image in SAMPLES_DIR."""
    if not SAMPLES_DIR.exists():
        return []

    return [
        {"path": str(p), "name": p.name}
        for p in SAMPLES_DIR.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
