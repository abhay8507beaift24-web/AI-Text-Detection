"""
features.py - Feature engineering for AI Text Detection

Combines:
1. TF-IDF n-gram features (lexical)
2. Stylometric / hand-crafted features (statistical)

Stylometric features capture writing style differences between humans and AI:
- Average word length
- Sentence length variance
- Punctuation density
- Type-token ratio (lexical diversity)
- Contraction usage (humans use more)
- First-person pronoun usage
- Passive voice indicators
- Hedge word usage (AI tends to over-hedge)
"""

import re
import string
import numpy as np
import pandas as pd
from typing import List
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import hstack, issparse


# --- Stylometric feature transformer ---

CONTRACTIONS = re.compile(
    r"\b(i'm|i've|i'll|i'd|you're|you've|you'll|you'd|he's|she's|it's|"
    r"we're|we've|we'll|we'd|they're|they've|they'll|they'd|isn't|aren't|"
    r"wasn't|weren't|don't|doesn't|didn't|can't|couldn't|won't|wouldn't|"
    r"shouldn't|haven't|hasn't|hadn't|I'm|I've|I'll|I'd)\b"
)

FIRST_PERSON = re.compile(r"\b(i|me|my|mine|myself|we|our|ours|ourselves)\b", re.IGNORECASE)

HEDGE_WORDS = re.compile(
    r"\b(furthermore|additionally|it is important|it is worth|it should be noted|"
    r"in conclusion|in summary|it is evident|it is clear|notably|significantly|"
    r"comprehensive|robust|optimal|leverage|utilize|facilitate|implement|"
    r"multifaceted|herein|aforementioned)\b",
    re.IGNORECASE,
)

PASSIVE_INDICATORS = re.compile(
    r"\b(is|are|was|were|be|been|being)\s+\w+ed\b", re.IGNORECASE
)


def extract_stylometric(text: str) -> np.ndarray:
    """Extract 10 stylometric features from a text."""
    if not text or not text.strip():
        return np.zeros(10)

    words = text.split()
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    n_words = max(len(words), 1)
    n_sentences = max(len(sentences), 1)

    # 1. Avg word length
    avg_word_len = np.mean([len(w.strip(string.punctuation)) for w in words]) if words else 0

    # 2. Avg sentence length (words)
    avg_sent_len = n_words / n_sentences

    # 3. Sentence length variance
    sent_lens = [len(s.split()) for s in sentences]
    sent_len_var = np.var(sent_lens) if len(sent_lens) > 1 else 0

    # 4. Punctuation density
    punct_count = sum(1 for c in text if c in string.punctuation)
    punct_density = punct_count / max(len(text), 1)

    # 5. Type-token ratio (lexical diversity)
    unique_words = set(w.lower().strip(string.punctuation) for w in words)
    ttr = len(unique_words) / n_words

    # 6. Contraction density
    contractions = len(CONTRACTIONS.findall(text))
    contraction_density = contractions / n_words

    # 7. First-person pronoun density
    first_person = len(FIRST_PERSON.findall(text))
    first_person_density = first_person / n_words

    # 8. Hedge word density
    hedge = len(HEDGE_WORDS.findall(text))
    hedge_density = hedge / n_words

    # 9. Passive voice density
    passive = len(PASSIVE_INDICATORS.findall(text))
    passive_density = passive / n_sentences

    # 10. Exclamation + question mark ratio
    excl_q = text.count("!") + text.count("?")
    excl_q_ratio = excl_q / max(len(text), 1)

    return np.array([
        avg_word_len, avg_sent_len, sent_len_var, punct_density,
        ttr, contraction_density, first_person_density,
        hedge_density, passive_density, excl_q_ratio,
    ])


class StylometricTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for stylometric features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.vstack([extract_stylometric(text) for text in X])


# --- Combined feature pipeline ---

def build_feature_pipeline() -> FeatureUnion:
    """
    Build a FeatureUnion combining:
    - Word-level TF-IDF (unigrams + bigrams)
    - Char-level TF-IDF (for subtle AI writing patterns)
    - Stylometric features
    """
    tfidf_word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=20000,
        sublinear_tf=True,
        min_df=2,
    )

    tfidf_char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=10000,
        sublinear_tf=True,
        min_df=2,
    )

    feature_union = FeatureUnion([
        ("tfidf_word", tfidf_word),
        ("tfidf_char", tfidf_char),
        ("stylometric", StylometricTransformer()),
    ])

    return feature_union


def fit_transform_features(X_train: List[str], X_test: List[str]):
    """
    Fit on training data, transform both train and test.

    Returns:
        X_train_feat, X_test_feat, pipeline
    """
    pipeline = build_feature_pipeline()
    X_train_feat = pipeline.fit_transform(X_train)
    X_test_feat = pipeline.transform(X_test)

    print(f"[features] Feature matrix shape: train={X_train_feat.shape}, test={X_test_feat.shape}")
    return X_train_feat, X_test_feat, pipeline