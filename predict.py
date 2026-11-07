"""Shared next-word inference used by the Flask API and the smoke test.

``Predictor`` loads the tokenizer and every trained model ONCE, then answers requests with real
forward passes of the selected network.  Special tokens (<pad>, <unk>, <num>, <math>, <code>)
and punctuation are masked out, so they can never be suggested.

Two modes are returned to the front end:

* ``next_word``      - the text ends in a complete word (or a space / punctuation): suggest the next word.
* ``complete_word``  - the text ends in a fragment that is not a known word (e.g. "neura"): the model's
                       next-word distribution after the *preceding* words is restricted to vocabulary
                       words starting with that fragment ("neural", "neurons"...).
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from preprocess import Vocabulary, tokenize

log = logging.getLogger("predict")
MODEL_NAMES = ("RNN", "LSTM", "GRU")
MAX_CHARS = 4000       # only the tail of very long texts matters (context is a few words)
MAX_TOP_K = 10
_TAIL_FRAGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*$")


class PredictorError(ValueError):
    """Raised for invalid requests; the message is safe to show to users."""


class Predictor:
    def __init__(self, model_dir: str | Path = "models"):
        self.model_dir = Path(model_dir)
        self.models: dict = {}
        self.config: dict = {}
        self.vocab: Vocabulary | None = None
        self.seq_len = 0
        self._lock = threading.Lock()
        self._load()

    # ------------------------------------------------------------------ loading
    def _load(self) -> None:
        cfg_path, tok_path = self.model_dir / "config.json", self.model_dir / "tokenizer.json"
        if not cfg_path.exists() or not tok_path.exists():
            log.warning("No trained assets in %s - run: python train.py", self.model_dir)
            return
        import tensorflow as tf  # imported lazily so `import predict` stays cheap for tools

        self.config = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.vocab = Vocabulary.load(tok_path)
        self.seq_len = int(self.config["data"]["seq_len"])
        for name in MODEL_NAMES:
            path = self.model_dir / f"{name.lower()}.keras"
            if not path.exists():
                log.warning("Model file missing: %s", path)
                continue
            try:
                self.models[name] = tf.keras.models.load_model(path, compile=False)
            except Exception:  # corrupt file / incompatible Keras version
                log.exception("Could not load %s - retrain or check the TensorFlow version", path)
        for name, model in self.models.items():  # warm-up: build graphs so the first user request is fast
            model(np.zeros((1, self.seq_len), dtype="int32"), training=False)
        log.info("Loaded models %s (vocab %d, context %d tokens)", list(self.models), len(self.vocab), self.seq_len)

    @property
    def ready(self) -> bool:
        return bool(self.models) and self.vocab is not None

    @property
    def available_models(self) -> list[str]:
        return [m for m in MODEL_NAMES if m in self.models]

    # ------------------------------------------------------------------ inference
    def _split_partial(self, text: str):
        """Return (context_tokens, fragment). fragment != '' means the last word is unfinished."""
        tokens = tokenize(text)
        if not tokens or not text or text[-1].isspace():
            return tokens, ""
        m = _TAIL_FRAGMENT.search(text)
        if not m or tokens[-1] != m.group(0).lower():
            return tokens, ""                       # ends in punctuation etc.
        frag = tokens[-1]
        if self.vocab.token_to_id.get(frag, 0) >= self.vocab.first_word_id or frag.isdigit():
            return tokens, ""                       # a complete, known word
        return tokens[:-1], frag

    def predict(self, text: str, model_name: str, top_k: int = 5) -> dict:
        if not self.ready:
            raise PredictorError("Models are not available yet. Train them first with: python train.py")
        name = str(model_name).upper()
        if name not in self.models:
            raise PredictorError(f"Unknown model. Choose one of: {', '.join(self.available_models)}.")
        try:
            top_k = max(1, min(int(top_k), MAX_TOP_K))
        except (TypeError, ValueError):
            raise PredictorError("top_k must be a number.")
        text = (text or "")[-MAX_CHARS:]

        t0 = time.perf_counter()
        base = {"model": name, "predictions": [], "mode": "next_word", "prefix": "", "unknown_in_context": 0}
        if not text.strip():
            return {**base, "status": "empty", "message": "Start typing to get suggestions.", "duration_ms": 0.0}

        context, frag = self._split_partial(text)
        ids = self.vocab.encode_tokens(context)[-self.seq_len:]
        known = sum(1 for i in ids if i >= self.vocab.first_word_id)
        unknown = sum(1 for i in ids if i == self.vocab.unk_id)
        if known == 0:
            return {**base, "status": "unknown_words", "unknown_in_context": unknown, "mode": "complete_word" if frag else "next_word",
                    "prefix": frag, "message": "These words are outside the model's vocabulary.",
                    "duration_ms": round((time.perf_counter() - t0) * 1000, 1)}

        x = np.zeros((1, self.seq_len), dtype="int32")      # left-padded, most recent word last
        x[0, self.seq_len - len(ids):] = ids
        with self._lock:                                     # one forward pass at a time
            probs = np.array(self.models[name](x, training=False))[0]
        probs[: self.vocab.first_word_id] = 0.0             # never <pad>/<unk>/<num>/<math>/<code>/punctuation

        if frag:
            allowed = np.array([i for i, t in enumerate(self.vocab.id_to_token)
                                if i >= self.vocab.first_word_id and t.startswith(frag) and t != frag], dtype=np.int64)
            if allowed.size == 0:
                return {**base, "status": "no_completion", "mode": "complete_word", "prefix": frag,
                        "unknown_in_context": unknown, "message": "No vocabulary word starts with that.",
                        "duration_ms": round((time.perf_counter() - t0) * 1000, 1)}
            candidate = allowed[np.argsort(-probs[allowed])[:top_k]]
        else:
            part = np.argpartition(-probs, top_k)[:top_k]
            candidate = part[np.argsort(-probs[part])]

        preds = [{"word": self.vocab.id_to_token[int(i)], "probability": round(float(probs[i]), 5)} for i in candidate]
        return {"model": name, "status": "ok", "mode": "complete_word" if frag else "next_word", "prefix": frag,
                "predictions": preds, "unknown_in_context": unknown,
                "duration_ms": round((time.perf_counter() - t0) * 1000, 1)}
