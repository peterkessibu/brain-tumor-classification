# Streamlit Community Cloud Deployment

This app uses **tensorflow-cpu 2.21.0**, which supports **Python 3.10–3.13**. Streamlit Cloud defaults to Python 3.12+.

## 1. Python version

Streamlit Community Cloud **does not** read `runtime.txt` or `.python-version`. Set Python in the dashboard only.

1. Open [share.streamlit.io](https://share.streamlit.io) and select your app.
2. Go to **Settings** (gear icon) → **Advanced settings**.
3. Set **Python version** to **3.12** or **3.13**.
4. Save and reboot the app.

If build logs still show an incompatible Python version:

1. Delete the app from the Streamlit dashboard.
2. Redeploy from GitHub.
3. Select a supported Python version in Advanced settings **during initial deployment**.

Older TensorFlow pins (e.g. `tensorflow-cpu==2.16.2`) fail on Python 3.13+:

```
ERROR: No matching distribution found for tensorflow-cpu==2.16.2
```

## 2. Secrets (required for AI explanations)

In **Settings → Secrets**, add:

```toml
GEMINI_API_KEY = "your-api-key-here"
```

The app loads this key in `config.py` via `st.secrets`. Without it, the app stops at startup.

## 3. Repository layout

| File | Purpose |
|------|---------|
| `app.py` | Main entry point |
| `requirements.txt` | Python dependencies |
| `packages.txt` | System packages for OpenCV (`ffmpeg`, `libsm6`, `libxext6`) |
| `cnn_model.h5` | Custom CNN weights (Git LFS) |
| `xception_model.weights.h5` | Xception fine-tuned weights (Git LFS) |

## 4. Model weights

Both model files are tracked with **Git LFS**. After cloning:

```bash
git lfs install
git lfs pull
```

If `xception_model.weights.h5` is missing or empty, re-export it from the Colab notebook (`BrainTumorClassification (1).ipynb`):

```python
model.save_weights("xception_model.weights.h5")
```

Then commit and push:

```bash
git add xception_model.weights.h5
git commit -m "Add Xception model weights"
git push
```

## 5. Deploy checklist

After pushing changes:

- [ ] Build log shows `Using Python 3.12.x environment`
- [ ] `tensorflow-cpu==2.21.0` installs without error
- [ ] App loads gallery and file upload UI
- [ ] Sample image inference works (CNN and/or Xception)
- [ ] Gemini explanation generates (secrets configured)

## 6. Local development

Use Python 3.10–3.13 to match production:

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run app.py
```

Set `GEMINI_API_KEY` in a `.env` file for local runs.
