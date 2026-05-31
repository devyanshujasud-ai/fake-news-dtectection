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
    fake_df = pd.read_csv(os.path.join(DATA_DIR, "Fake.csv"))

    true_df["label"] = 1
    fake_df["label"] = 0

    df = pd.concat([true_df, fake_df], ignore_index=True)

    # Combine title + text for richer features
    if "title" in df.columns and "text" in df.columns:
        df["combined"] = df["title"].fillna("") + " " + df["text"].fillna("")
    elif "text" in df.columns:
        df["combined"] = df["text"].fillna("")
    else:
        raise ValueError("Dataset must contain a 'text' column.")

    df["combined"] = df["combined"].apply(_clean)
    df = df[df["combined"].str.len() > 10]  # drop near-empty rows
    return df[["combined", "label"]].rename(columns={"combined": "text"})


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    ),
}


# ---------------------------------------------------------------------------
# Train & evaluate
# ---------------------------------------------------------------------------
def train_and_compare(
    df: pd.DataFrame | None = None,
    test_size: float = 0.2,
    max_features: int = 5000,
    selected_models: list[str] | None = None,
    progress_callback=None,
):
    """
    Train multiple models and return evaluation results.

    Parameters
    ----------
    df : DataFrame with columns text, label. If None, load_dataset() is called.
    test_size : fraction held out for testing.
    max_features : TF-IDF vocabulary cap.
    selected_models : list of model names to train (keys of MODELS dict).
    progress_callback : optional callable(step, total, message).

    Returns
    -------
    dict  – {
        "vectorizer": fitted TfidfVectorizer,
        "X_test", "y_test": test arrays,
        "results": {
            model_name: {
                "model": fitted estimator,
                "accuracy", "precision", "recall", "f1": floats,
                "confusion_matrix": 2×2 ndarray,
                "roc_fpr", "roc_tpr", "roc_auc": arrays / float,
                "classification_report": str,
                "y_pred": array,
            }
        }
    }
    """
    if df is None:
        df = load_dataset()

    models_to_train = {
        k: v for k, v in MODELS.items()
        if selected_models is None or k in selected_models
    }
    total_steps = len(models_to_train) + 1  # +1 for vectorisation

    # --- Vectorise --------------------------------------------------------
    if progress_callback:
        progress_callback(0, total_steps, "Vectorizing text with TF-IDF…")

    tfidf = TfidfVectorizer(max_features=max_features, stop_words="english")
    X = tfidf.fit_transform(df["text"])
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

