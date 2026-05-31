"""
Predictor module — handles text prediction and URL scraping.
Falls back to a demo mode when the trained model files are unavailable.
"""
import re
import hashlib
import numpy as np

# ---------------------------------------------------------------------------
# Load pre-trained model & vectorizer (cached at module level)
# ---------------------------------------------------------------------------
_model = None
_vectorizer = None
_demo_mode = False

try:
    import joblib
    _model = joblib.load("gbc_model.pkl")
    _vectorizer = joblib.load("vectorizer.pkl")
except Exception:
    _demo_mode = True


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def _clean_text(text: str) -> str:
