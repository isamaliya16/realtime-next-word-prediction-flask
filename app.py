"""Flask web application: real-time next-word prediction + evaluation dashboard.

    python app.py                      # http://127.0.0.1:5000
    PORT=8000 python app.py            # another port

Models and tokenizer are loaded ONCE when the process starts.  Nothing is trained or downloaded
at request time.  Environment overrides (mainly for tests): NW_MODEL_DIR, NW_OUTPUT_DIR, NW_DATA_DIR.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

from predict import MAX_CHARS, Predictor, PredictorError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

BASE = Path(__file__).resolve().parent
ALLOWED_IMAGES = {"loss_curves.png", "accuracy_curves.png", "model_comparison.png"}


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def create_app(model_dir: str | None = None, output_dir: str | None = None, data_dir: str | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024            # a prediction request is a few hundred bytes
    model_dir = Path(model_dir or os.getenv("NW_MODEL_DIR", BASE / "models"))
    output_dir = Path(output_dir or os.getenv("NW_OUTPUT_DIR", BASE / "outputs"))
    data_dir = Path(data_dir or os.getenv("NW_DATA_DIR", BASE / "data"))

    predictor = Predictor(model_dir)                         # loaded once per process
    app.extensions["predictor"] = predictor

    def error(message: str, status: int):
        return jsonify({"status": "error", "error": message}), status

    @app.get("/")
    def index():
        metrics = _read_json(output_dir / "metrics.json")
        stats = _read_json(data_dir / "processed" / "stats.json")
        graphs = [f for f in ("loss_curves.png", "accuracy_curves.png", "model_comparison.png")
                  if (output_dir / f).exists()]
        return render_template("index.html", ready=predictor.ready, models=predictor.available_models,
                               metrics=metrics, stats=stats, graphs=graphs, max_chars=MAX_CHARS)

    @app.get("/api/models")
    def models():
        return jsonify({"ready": predictor.ready, "models": predictor.available_models})

    @app.get("/api/metrics")
    def metrics():
        data = _read_json(output_dir / "metrics.json")
        return (jsonify(data), 200) if data else error("Evaluation results are not available yet.", 404)

    @app.get("/outputs/<path:name>")
    def output_file(name):
        if name not in ALLOWED_IMAGES:
            return error("Not found.", 404)
        return send_from_directory(output_dir, name, max_age=60)

    @app.post("/api/predict")
    def api_predict():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return error("Send a JSON body with 'text' and 'model'.", 400)
        text, model_name = body.get("text", ""), body.get("model", "LSTM")
        if not isinstance(text, str) or not isinstance(model_name, str):
            return error("'text' and 'model' must be strings.", 400)
        try:
            result = predictor.predict(text, model_name, body.get("top_k", 5))
            return jsonify(result)
        except PredictorError as exc:
            return error(str(exc), 400 if predictor.ready else 503)
        except Exception:                                    # never leak internals to the browser
            log.exception("prediction failed")
            return error("The prediction service had a problem. Please try again.", 500)

    @app.errorhandler(404)
    def not_found(_):
        return error("Not found.", 404)

    @app.errorhandler(413)
    def too_large(_):
        return error("Request too large.", 413)

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    log.info("Open http://127.0.0.1:%d", port)
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=port, debug=False, threaded=True, use_reloader=False)
