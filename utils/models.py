"""
Models module — train, compare, and evaluate multiple ML classifiers.
"""
import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
)

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def dataset_available() -> bool:
    """Check whether True.csv and Fake.csv exist in data/."""
    return (
        os.path.isfile(os.path.join(DATA_DIR, "True.csv"))
        and os.path.isfile(os.path.join(DATA_DIR, "Fake.csv"))
    )


def load_dataset() -> pd.DataFrame:
    """
    Load and merge True.csv + Fake.csv.
    Returns a DataFrame with columns: text, label (1=Real, 0=Fake).
    """
    true_df = pd.read_csv(os.path.join(DATA_DIR, "True.csv"))
