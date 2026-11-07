import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

app = Flask(__name__)
MODEL_DIR = Path("models")

def load_assets():
    config_path = MODEL_DIR / "config.json"
    tokenizer_path = MODEL_DIR / "tokenizer.json"
    if not config_path.exists() or not tokenizer_path.exists():
        return None, None, None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tokenizer = tokenizer_from_json(tokenizer_path.read_text(encoding="utf-8"))
    models = {}
    for name in ("RNN", "LSTM", "GRU"):
        path = MODEL_DIR / f"{name.lower()}.keras"
        if path.exists():
            models[name] = tf.keras.models.load_model(path)
    return config, tokenizer, models

CONFIG, TOKENIZER, MODELS = load_assets()

@app.route("/")
def index():
    trained = bool(MODELS)
    return render_template(
        "index.html",
        trained=trained,
        results=(CONFIG or {}).get("results", {}),
    )

@app.route("/api/predict", methods=["POST"])
def predict():
    global CONFIG, TOKENIZER, MODELS
    if not MODELS:
        return jsonify({"error": "Models are not trained yet. Run: python train.py"}), 400
    body = request.get_json(silent=True) or {}
    text = str(body.get("text", "")).strip().lower()
    model_name = str(body.get("model", "LSTM")).upper()
    top_k = int(body.get("top_k", 5))
    top_k = max(1, min(top_k, 10))

    if not text:
        return jsonify({"error": "Please enter a sentence or phrase."}), 400
    if model_name not in MODELS:
        return jsonify({"error": "Select RNN, LSTM, or GRU."}), 400

    ids = TOKENIZER.texts_to_sequences([text])[0]
    if not ids:
        return jsonify({"error": "No known words found. Try a phrase from the training text."}), 400
    max_len = CONFIG["max_len"]
    padded = pad_sequences([ids[-max_len:]], maxlen=max_len, padding="pre")
    probabilities = MODELS[model_name].predict(padded, verbose=0)[0]
    probabilities[0] = 0
    indices = np.argsort(probabilities)[-top_k:][::-1]
    word_index = TOKENIZER.word_index
    reverse_index = {v: k for k, v in word_index.items()}
    predictions = [
        {"word": reverse_index.get(int(i), "<unk>"), "probability": round(float(probabilities[i]) * 100, 2)}
        for i in indices if probabilities[i] > 0
    ]
    return jsonify({"model": model_name, "predictions": predictions})

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
