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
