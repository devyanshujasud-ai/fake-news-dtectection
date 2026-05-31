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
    fake_score = sum(1 for kw in _FAKE_KEYWORDS if kw in lower)

    # Use hash for determinism + slight randomness feel
    h = int(hashlib.md5(cleaned.encode()).hexdigest()[:8], 16)
    base_fake = 0.25 + (h % 1000) / 2500  # 0.25 – 0.65 baseline

    # Adjust by keyword hits
    base_fake += fake_score * 0.07
    base_fake = min(base_fake, 0.98)

    # Longer, well-structured texts lean toward "REAL"
    word_count = len(cleaned.split())
    if word_count > 150:
        base_fake -= 0.15
    elif word_count > 80:
        base_fake -= 0.08
    elif word_count < 20:
        base_fake += 0.1

    base_fake = max(0.04, min(0.96, base_fake))

    fake_prob = round(base_fake, 4)
    real_prob = round(1 - fake_prob, 4)

    if real_prob >= fake_prob:
        label = "REAL"
        confidence = real_prob
    else:
        label = "FAKE"
        confidence = fake_prob

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": {
            "FAKE": fake_prob,
            "REAL": real_prob,
        },
    }


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------
def predict_news(text: str) -> dict:
    """
    Predict whether *text* is REAL or FAKE.

    Returns
    -------
    dict with keys:
        label   – "REAL" | "FAKE"
        confidence – float 0-1 (probability of the predicted class)
        probabilities – dict {"REAL": float, "FAKE": float}
    """
    if _demo_mode:
        return _demo_predict(text)

    cleaned = _clean_text(text)
    if not cleaned:
        return {"label": "UNKNOWN", "confidence": 0.0,
                "probabilities": {"REAL": 0.0, "FAKE": 0.0}}

    vec = _vectorizer.transform([cleaned])
    pred = _model.predict(vec)[0]
    proba = _model.predict_proba(vec)[0]

    label = "REAL" if pred == 1 else "FAKE"
    confidence = float(np.max(proba))

    # Gradient Boosting classes_ order: [0, 1] → [FAKE, REAL]
    return {
        "label": label,
        "confidence": confidence,
        "probabilities": {
            "FAKE": float(proba[0]),
            "REAL": float(proba[1]),
        },
    }


# ---------------------------------------------------------------------------
# URL-based prediction
# ---------------------------------------------------------------------------
def predict_from_url(url: str) -> dict:
    """
    Download the article at *url*, extract its text with newspaper3k,
    then predict.

    Returns the same dict as predict_news, plus:
        title   – extracted article title
        text    – extracted article body (first 5000 chars)
        authors – list of author names
    """
    try:
        from newspaper import Article

        article = Article(url)
        article.download()
        article.parse()

        title = article.title or ""
        body = article.text or ""
        authors = article.authors or []

        if not body.strip():
            return {
                "label": "ERROR",
                "confidence": 0.0,
                "probabilities": {"REAL": 0.0, "FAKE": 0.0},
                "title": title,
                "text": "",
                "authors": authors,
                "error": "Could not extract article text from this URL.",
            }

        result = predict_news(body[:5000])
        result["title"] = title
        result["text"] = body[:5000]
        result["authors"] = authors
        return result

    except Exception as e:
        # In demo mode, return a simulated URL result
        if _demo_mode:
            result = _demo_predict(f"article from {url}")
            result["title"] = f"Article from {url.split('/')[2] if '/' in url else url}"
            result["text"] = (
                "This is a simulated article extraction for demo purposes. "
                "In production, the newspaper3k library would download and parse "
                "the actual article content from the provided URL."
            )
            result["authors"] = ["Demo Author"]
            return result

        return {
            "label": "ERROR",
            "confidence": 0.0,
            "probabilities": {"REAL": 0.0, "FAKE": 0.0},
            "title": "",
            "text": "",
            "authors": [],
            "error": str(e),
        }


def is_demo_mode() -> bool:
    """Return whether the predictor is running in demo mode."""
    return _demo_mode
