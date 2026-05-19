"""
model.py - Classical ML models for AI Text Detection

Trains and evaluates multiple models:
1. Logistic Regression (fast, strong baseline)
2. Random Forest
3. Gradient Boosting (XGBoost-style via sklearn)
4. Linear SVM

Returns the best model based on F1 score.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    roc_auc_score, confusion_matrix
)
from typing import List, Dict, Any


MODELS = {
    "Logistic Regression": LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    ),
    "Linear SVM": CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=2000, random_state=42, class_weight="balanced"),
        cv=3,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        random_state=42,
    ),
}


def train_and_evaluate(
    X_train, X_test, y_train, y_test, verbose: bool = True
) -> Dict[str, Any]:
    """
    Train all models and evaluate them. Returns results dict + best model.
    """
    results = {}
    best_model = None
    best_f1 = -1
    best_name = ""

    for name, clf in MODELS.items():
        if verbose:
            print(f"\n[model] Training {name}...")

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        try:
            y_proba = clf.predict_proba(X_test)[:, 1]
            auc = round(roc_auc_score(y_test, y_proba), 4)
        except Exception:
            auc = None

        acc = round(accuracy_score(y_test, y_pred), 4)
        f1 = round(f1_score(y_test, y_pred, average="weighted"), 4)
        cm = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(y_test, y_pred, target_names=["Human", "AI-Generated"])

        results[name] = {
            "model": clf,
            "accuracy": acc,
            "f1_score": f1,
            "roc_auc": auc,
            "confusion_matrix": cm,
            "report": report,
        }

        if verbose:
            print(f"  Accuracy : {acc:.4f}")
            print(f"  F1 Score : {f1:.4f}")
            if auc:
                print(f"  ROC AUC  : {auc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = clf
            best_name = name

    if verbose:
        print(f"\n[model] Best model: {best_name} (F1={best_f1:.4f})")
        print("\n" + results[best_name]["report"])

    return {
        "results": results,
        "best_model": best_model,
        "best_name": best_name,
        "best_f1": best_f1,
    }


def predict_single(text_features, model) -> Dict[str, Any]:
    """
    Predict a single (already-transformed) sample.

    Returns:
        dict with label, confidence, and human/AI probabilities
    """
    pred = model.predict(text_features)[0]
    try:
        proba = model.predict_proba(text_features)[0]
        human_prob = round(float(proba[0]) * 100, 1)
        ai_prob = round(float(proba[1]) * 100, 1)
        confidence = round(max(float(proba)) * 100, 1)
    except Exception:
        human_prob = 100 if pred == 0 else 0
        ai_prob = 0 if pred == 0 else 100
        confidence = 100.0

    return {
        "label": int(pred),
        "label_name": "Human" if pred == 0 else "AI-Generated",
        "confidence": confidence,
        "human_probability": human_prob,
        "ai_probability": ai_prob,
    }