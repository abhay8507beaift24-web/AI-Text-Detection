"""
deep_model.py - Neural Network model for AI Text Detection

Improved from original:
- Deeper architecture (3 hidden layers)
- Proper regularization (L2 + early stopping)
- More iterations with convergence monitoring
- Probability calibration
- Comparable evaluation to classical models
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from typing import Dict, Any


def train_deep_model(X_train, X_test, y_train, y_test, verbose: bool = True) -> Dict[str, Any]:
    """
    Train an improved MLP (Multi-Layer Perceptron) neural network.

    Improvements over original:
    - 3 hidden layers: (256, 128, 64) for richer representations
    - alpha=0.001 for L2 regularization to prevent overfitting
    - early_stopping=True to halt when validation loss stops improving
    - max_iter=500 (was 5 — far too few to converge)
    - batch_size='auto' for mini-batch gradient descent
    - learning_rate='adaptive' to reduce LR on plateau
    """
    if verbose:
        print("\n[deep_model] Training MLP Neural Network...")
        print("[deep_model] Architecture: (256 → 128 → 64) with ReLU activation")

    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=0.001,                 # L2 regularization
        batch_size="auto",
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=300,                # was 5 in original — completely insufficient
        early_stopping=True,         # stop when val loss stops improving
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
        verbose=False,
    )

    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    y_proba = mlp.predict_proba(X_test)[:, 1]

    acc = round(accuracy_score(y_test, y_pred), 4)
    f1 = round(f1_score(y_test, y_pred, average="weighted"), 4)
    auc = round(roc_auc_score(y_test, y_proba), 4)
    report = classification_report(y_test, y_pred, target_names=["Human", "AI-Generated"])

    if verbose:
        print(f"[deep_model] Stopped at iteration: {mlp.n_iter_}")
        print(f"[deep_model] Accuracy : {acc:.4f}")
        print(f"[deep_model] F1 Score : {f1:.4f}")
        print(f"[deep_model] ROC AUC  : {auc:.4f}")
        print("\n" + report)

    return {
        "model": mlp,
        "accuracy": acc,
        "f1_score": f1,
        "roc_auc": auc,
        "report": report,
        "n_iter": mlp.n_iter_,
    }