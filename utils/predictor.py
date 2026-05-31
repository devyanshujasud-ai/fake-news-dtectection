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
    """Basic text cleaning: lowercase, strip URLs, special chars."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)          # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)                # keep letters only
    text = re.sub(r"\s+", " ", text).strip()                # collapse whitespace
    return text


# ---------------------------------------------------------------------------
# Demo mode prediction (deterministic based on text hash)
# ---------------------------------------------------------------------------
_FAKE_KEYWORDS = [
    "shocking", "unbelievable", "secret", "conspiracy", "coverup",
    "hoax", "exposed", "they don't want you to know", "breaking",
    "urgent", "bombshell", "leaked", "banned", "censored",
    "miracle", "cure", "aliens", "illuminati", "deep state",
]

def _demo_predict(text: str) -> dict:
    """Generate a realistic-looking prediction based on text content."""
    cleaned = _clean_text(text)
    if not cleaned:
        return {"label": "UNKNOWN", "confidence": 0.0,
                "probabilities": {"REAL": 0.0, "FAKE": 0.0}}

    # Count fake-sounding keywords
    lower = text.lower()
