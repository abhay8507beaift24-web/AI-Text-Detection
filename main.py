"""
main.py - Main training pipeline for AI Text Detection

Runs the full pipeline:
1. Load data
2. Preprocess
3. EDA
4. Feature engineering
5. Train classical models
6. Train deep model
7. Save the best model + feature pipeline
8. Print final summary
"""

import pickle
import warnings
import sys
from pathlib import Path

warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from load_data import load_data
from preprocess import preprocess_dataframe, split_data
from eda import run_eda
from features import fit_transform_features
from model import train_and_evaluate
from deep_model import train_deep_model


MODEL_DIR = Path("saved_models")


def save_artifacts(feature_pipeline, best_model, model_name: str):
    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / "feature_pipeline.pkl", "wb") as f:
        pickle.dump(feature_pipeline, f)
    with open(MODEL_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(MODEL_DIR / "model_name.txt", "w") as f:
        f.write(model_name)
    print(f"\n[main] Models saved to '{MODEL_DIR}/'")


def main():
    print("=" * 60)
    print("   AI TEXT DETECTION — Training Pipeline")
    print("=" * 60)

    # 1. Load data
    df = load_data(source="synthetic", n_samples=2000)

    # 2. Preprocess
    df = preprocess_dataframe(df)

    # 3. EDA
    print("\n[main] Running EDA...")
    run_eda(df, save_plots=True)

    # 4. Split
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)

    # 5. Feature engineering
    X_train_feat, X_test_feat, feature_pipeline = fit_transform_features(X_train, X_test)

    # 6. Train classical models
    print("\n[main] Training classical ML models...")
    results = train_and_evaluate(X_train_feat, X_test_feat, y_train, y_test)

    # 7. Train deep model
    deep_result = train_deep_model(X_train_feat, X_test_feat, y_train, y_test)

    # 8. Compare and pick overall best
    all_models = {name: info for name, info in results["results"].items()}
    all_models["MLP Neural Network"] = deep_result

    best_name = max(all_models, key=lambda k: all_models[k]["f1_score"])
    best_model = all_models[best_name]["model"]

    # 9. Save artifacts
    save_artifacts(feature_pipeline, best_model, best_name)

    # 10. Final summary
    print("\n" + "=" * 60)
    print("   FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<25} {'Accuracy':>10} {'F1 Score':>10} {'ROC AUC':>10}")
    print("-" * 60)
    for name, info in sorted(all_models.items(), key=lambda x: -x[1]["f1_score"]):
        auc = f"{info['roc_auc']:.4f}" if info.get("roc_auc") else "  N/A  "
        marker = " ◀ BEST" if name == best_name else ""
        print(f"{name:<25} {info['accuracy']:>10.4f} {info['f1_score']:>10.4f} {auc:>10}{marker}")
    print("=" * 60)
    print(f"\n✓ Training complete. Best model: {best_name}")
    print("✓ Run 'python app.py' to start the web server.")


if __name__ == "__main__":
    main()