"""
app.py - Flask backend for AI Text Detection web app

Endpoints:
  GET  /           → serves the frontend
  POST /predict    → classify a text snippet
  GET  /stats      → return model training stats
  GET  /health     → health check
"""

import pickle
import warnings
import os
import sys
import json
import time
from pathlib import Path
from functools import lru_cache

warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Base paths
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

# Ensure local modules are importable
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(BASE_DIR))

from preprocess import clean_text
from model import predict_single
from features import extract_stylometric

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR = ROOT_DIR / "saved_models"
FRONTEND_DIR = BASE_DIR
APP_VERSION = "1.0.0"

app = Flask(__name__, static_folder=str(FRONTEND_DIR))
CORS(app)

# ── Load artifacts ─────────────────────────────────────────────────────────────
feature_pipeline = None
best_model = None
model_name = "Not loaded"


def load_artifacts():
    global feature_pipeline, best_model, model_name
    try:
        with open(MODEL_DIR / "feature_pipeline.pkl", "rb") as f:
            feature_pipeline = pickle.load(f)
        with open(MODEL_DIR / "best_model.pkl", "rb") as f:
            best_model = pickle.load(f)
        with open(MODEL_DIR / "model_name.txt") as f:
            model_name = f.read().strip()
        print(f"[app] Loaded model: {model_name}")
        return True
    except FileNotFoundError:
        print("[app] WARNING: No saved model found. Run 'python main.py' first.")
        return False


def ensure_model():
    """Auto-train if no saved model exists."""
    if not (MODEL_DIR / "best_model.pkl").exists():
        print("[app] No model found — training now...")
        from main import main as train
        train()
    return load_artifacts()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/health")
def health():
    model_loaded = best_model is not None
    return jsonify({
        "status": "ok",
        "model_loaded": model_loaded,
        "model_name": model_name,
        "version": APP_VERSION,
    })


@app.route("/predict", methods=["POST"])
def predict():
    if best_model is None:
        return jsonify({"error": "Model not loaded. Run 'python main.py' first."}), 503

    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Request must include a 'text' field."}), 400

    text = str(data["text"]).strip()
    if len(text) < 20:
        return jsonify({"error": "Text is too short. Please enter at least 20 characters."}), 400
    if len(text) > 10000:
        return jsonify({"error": "Text is too long. Maximum 10,000 characters."}), 400

    start_time = time.time()

    # Preprocess + extract features
    cleaned = clean_text(text)
    features = feature_pipeline.transform([cleaned])

    # Predict
    result = predict_single(features, best_model)

    # Stylometric breakdown for explanation
    stylo = extract_stylometric(cleaned)
    stylo_labels = [
        "avg_word_length", "avg_sentence_length", "sentence_length_variance",
        "punctuation_density", "type_token_ratio", "contraction_density",
        "first_person_density", "hedge_word_density", "passive_voice_density",
        "exclamation_question_ratio",
    ]
    stylometric_features = {k: round(float(v), 4) for k, v in zip(stylo_labels, stylo)}

    elapsed_ms = round((time.time() - start_time) * 1000, 1)

    # Build explanation
    indicators = []
    if stylo[5] > 0.02:  # contractions
        indicators.append("Contains contractions (more human-like)")
    if stylo[7] > 0.01:  # hedge words
        indicators.append("Contains formal AI-style hedging phrases")
    if stylo[6] > 0.03:  # first person
        indicators.append("Uses first-person pronouns (more human-like)")
    if stylo[2] > 10:  # sentence length variance
        indicators.append("High sentence length variation (more human-like)")
    if stylo[8] > 0.5:  # passive voice
        indicators.append("Heavy use of passive voice (more AI-like)")
    if stylo[4] > 0.7:  # high TTR
        indicators.append("High lexical diversity")

    return jsonify({
        "text_length": len(text),
        "word_count": len(text.split()),
        "label": result["label"],
        "label_name": result["label_name"],
        "confidence": result["confidence"],
        "human_probability": result["human_probability"],
        "ai_probability": result["ai_probability"],
        "model_used": model_name,
        "stylometric_features": stylometric_features,
        "indicators": indicators,
        "processing_time_ms": elapsed_ms,
    })


@app.route("/stats")
def stats():
    """Return model metadata for the frontend dashboard."""
    stats_path = MODEL_DIR / "training_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            return jsonify(json.load(f))
    return jsonify({
        "model_name": model_name,
        "status": "Model loaded and ready",
        "note": "Run main.py to generate detailed training stats."
    })


@app.route("/examples")
def examples():
    """Return example texts for demo purposes."""
    examples = [
        {
            "label": "human",
            "title": "Casual Blog Post",
            "text": "Honestly I wasn't planning to write about this but here we are. I've been thinking about how weird it is that I can eat the same meal every day for a week and not get bored, but if someone *tells* me I have to eat it, I'm immediately done. Like, my brain just rebels against obligation. Anyway. Made pasta again tonight. No regrets."
        },
        {
            "label": "ai",
            "title": "Formal Report",
            "text": "In conclusion, it is evident that the implementation of a comprehensive strategy is paramount to achieving optimal outcomes. Furthermore, the data presented herein demonstrates a statistically significant correlation between the adoption of advanced methodologies and improved performance metrics. It is important to note that these findings have significant implications for future research and development endeavors."
        },
        {
            "label": "human",
            "title": "Reddit Comment",
            "text": "lol no way this works. I tried the same thing three months ago and gave up after like two hours. maybe I was doing it wrong but the docs are genuinely horrible. props to you for figuring it out though, gonna try again this weekend"
        },
        {
            "label": "ai",
            "title": "Product Description",
            "text": "This innovative solution leverages state-of-the-art artificial intelligence to deliver a seamless and intuitive user experience. By harnessing the power of machine learning algorithms, it provides robust, scalable, and enterprise-grade capabilities that facilitate streamlined workflows and enhanced productivity across multiple dimensions."
        },
    ]
    return jsonify(examples)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  AI Text Detector — Web Server")
    print("=" * 50)
    ensure_model()
    port = int(os.environ.get("PORT", 5000))
    print(f"\n[app] Running on http://localhost:{port}")
    app.run(debug=False, host="0.0.0.0", port=port)