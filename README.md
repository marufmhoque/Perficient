# Perficient
Interview Building Challenge

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy to Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app** → choose repo `marufmhoque/Perficient`, branch `main`, main file `streamlit_app.py`.
3. Deploy. Any push to `main` redeploys automatically.

Secrets (API keys, etc.) go in the app's **Settings → Secrets** on Streamlit Cloud, or locally in `.streamlit/secrets.toml` (git-ignored).
