"""
preprocess.py - Text preprocessing pipeline for AI Text Detection
"""

import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    """
    Clean and normalize a single text string.
    - Lowercase
    - Remove URLs
    - Remove excessive whitespace
    - Keep punctuation (useful for stylometric features)
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)           # remove URLs
    text = re.sub(r"\s+", " ", text).strip()             # normalize whitespace
    return text


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cleaning to all texts in the dataframe.
    Returns a new dataframe with an added 'clean_text' column.
    """
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 10].reset_index(drop=True)  # drop near-empty
    print(f"[preprocess] {len(df)} samples after cleaning.")
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split dataframe into train and test sets.

    Returns:
        X_train, X_test, y_train, y_test (all as lists/arrays)
    """
    X = df["clean_text"].tolist()
    y = df["label"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[preprocess] Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test