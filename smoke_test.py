"""Small end-to-end smoke test: every trained model + the Flask prediction API.

    python smoke_test.py

Needs trained models (python train.py). Exits with status 1 if any check fails.
Environment overrides: NW_MODEL_DIR, NW_OUTPUT_DIR, NW_DATA_DIR.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np

from app import create_app
from preprocess import PUNCT, SPECIALS

failures = []


def check(cond: bool, label: str, detail: str = ""):
    print(("  PASS  " if cond else "  FAIL  ") + label + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append(label)


def main() -> int:
    app = create_app()
    predictor = app.extensions["predictor"]
    if not predictor.ready:
        print("No trained models found. Run: python train.py")
        return 1
    client = app.test_client()
    print(f"Models loaded: {predictor.available_models}\n")

    dists = {}
    for name in predictor.available_models:
        print(f"[{name}]")
        r = client.post("/api/predict", json={"text": "A neural network learns", "model": name, "top_k": 5})
        body = r.get_json()
        check(r.status_code == 200 and body["status"] == "ok", "API returns status ok", str(body))
        preds = body.get("predictions", [])
        words = [p["word"] for p in preds]
        probs = [p["probability"] for p in preds]
        check(body.get("model") == name, "response names the model that was requested")
        check(len(preds) == 5, "returns 5 suggestions", str(words))
        check(all(0 < p <= 1 for p in probs) and sum(probs) <= 1.0001, "probabilities are valid", str(probs))
        check(probs == sorted(probs, reverse=True), "sorted by probability")
        check(not any(w in SPECIALS or w in PUNCT for w in words), "no <pad>/<unk>/punctuation suggested", str(words))
        check(len(set(words)) == len(words), "no duplicate suggestions")
        check("duration_ms" in body, "reports prediction duration")
        shown = ", ".join("%s (%.3f)" % (p["word"], p["probability"]) for p in preds)
        print("        suggestions: " + shown)
        full = predictor.predict("the training loss of the", name, 10)
        dists[name] = [p["probability"] for p in full["predictions"]]

    print("\n[model selection]")
    vals = list(dists.values())
    check(len(vals) < 2 or any(vals[0] != v for v in vals[1:]),
          "different models give different outputs (selected model is really used)")

    print("\n[edge cases]")
    m = predictor.available_models[0]
    r = client.post("/api/predict", json={"text": "", "model": m}).get_json()
    check(r["status"] == "empty" and r["predictions"] == [], "empty text -> no predictions, no error")
    r = client.post("/api/predict", json={"text": "qzxv wjkq plmn ", "model": m}).get_json()
    check(r["status"] == "unknown_words" and r["predictions"] == [], "only unknown words -> handled")
    r = client.post("/api/predict", json={"text": "the gradient desc", "model": m}).get_json()
    check(r["status"] == "ok" and r["mode"] == "complete_word" and all(p["word"].startswith("desc") for p in r["predictions"]),
          "unfinished word is completed with matching vocabulary words", json.dumps(r)[:200])
    r = client.post("/api/predict", json={"text": "Hello. A model", "model": m, "top_k": 3}).get_json()
    check(r["status"] == "ok" and len(r["predictions"]) == 3, "top_k respected")
    r = client.post("/api/predict", json={"text": "x " * 5000, "model": m})
    check(r.status_code in (200, 413), "very long text does not crash", str(r.status_code))

    print("\n[error handling]")
    r = client.post("/api/predict", json={"text": "hello", "model": "NOPE"})
    check(r.status_code == 400 and "error" in r.get_json(), "unknown model -> 400 with message")
    r = client.post("/api/predict", data="not json", content_type="text/plain")
    check(r.status_code == 400, "non-JSON body -> 400")
    r = client.post("/api/predict", json={"text": 123, "model": m})
    check(r.status_code == 400, "wrong types -> 400")
    r = client.post("/api/predict", json={"text": "hello", "model": m, "top_k": "abc"})
    check(r.status_code == 400, "bad top_k -> 400")
    check("Traceback" not in r.get_data(as_text=True), "errors do not leak internals")

    print("\n[web page]")
    page = client.get("/")
    html = page.get_data(as_text=True)
    check(page.status_code == 200, "GET / works")
    check('id="predict"' not in html and "Predict next word" not in html and "<button" in html and "Predict</button>" not in html,
          "there is no Predict button")
    check('id="editor"' in html and "app.js" in html, "editor and autocomplete script are present")
    check(client.get("/api/models").get_json()["ready"] is True, "/api/models reports ready")

    print(f"\n{'ALL CHECKS PASSED' if not failures else str(len(failures)) + ' CHECK(S) FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
