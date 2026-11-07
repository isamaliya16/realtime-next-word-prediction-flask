"""Train SimpleRNN, LSTM and GRU next-word models on the processed Deep Learning corpus.

    python train.py                                   # all three models, default settings
    python train.py --epochs 12 --units 192           # change hyper-parameters
    python train.py --models lstm gru                 # subset
    python train.py --seq-len 16 --vocab-size 8000    # data settings (dataset is rebuilt if they changed)
    python train.py --max-minutes 20                  # wall-clock budget per model (checked per epoch)

All three models share the same data, vocabulary, split, optimiser and training schedule; only the
recurrent cell differs.  The best epoch (lowest validation loss) of each model is checkpointed to
``models/<name>.keras``; histories and timings go to ``outputs/history_<name>.json``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

import preprocess

log = logging.getLogger("train")
MODEL_DIR, OUT_DIR = Path("models"), Path("outputs")
CELLS = {"RNN": tf.keras.layers.SimpleRNN, "LSTM": tf.keras.layers.LSTM, "GRU": tf.keras.layers.GRU}


def build_model(kind: str, vocab_size: int, seq_len: int, embedding_dim: int, units: int, dropout: float,
                lr: float) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_len,), dtype="int32"),
        tf.keras.layers.Embedding(vocab_size, embedding_dim, name="embedding"),
        tf.keras.layers.Dropout(dropout / 2, name="embedding_dropout"),
        CELLS[kind](units, name=kind.lower()),
        tf.keras.layers.Dropout(dropout, name="dropout"),
        tf.keras.layers.Dense(vocab_size, activation="softmax", name="next_word"),
    ], name=f"{kind.lower()}_next_word")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr, clipnorm=1.0),  # clipping keeps SimpleRNN stable
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                 tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy")])
    return model


class TimeBudget(tf.keras.callbacks.Callback):
    """Stop after an epoch if another epoch would exceed the wall-clock budget."""

    def __init__(self, seconds: float):
        super().__init__()
        self.seconds, self.t0, self.last = seconds, 0.0, 0.0
        self.hit = False

    def on_train_begin(self, logs=None):
        self.t0 = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.last = time.time()

    def on_epoch_end(self, epoch, logs=None):
        now = time.time()
        if self.seconds and (now - self.t0) + (now - self.last) > self.seconds:
            log.info("time budget reached after epoch %d - stopping", epoch + 1)
            self.hit = True
            self.model.stop_training = True


class EpochTimer(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self._t = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(round(time.time() - self._t, 1))


def make_dataset(X, y, batch_size, shuffle, seed):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(min(len(X), 200_000), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train_one(kind, args, data, vocab_size):
    X_train, y_train, X_val, y_val = data
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed)
    model = build_model(kind, vocab_size, args.seq_len, args.embedding_dim, args.units, args.dropout, args.lr)
    model.summary(print_fn=log.info)
    path = MODEL_DIR / f"{kind.lower()}.keras"
    timer, budget = EpochTimer(), TimeBudget(args.max_minutes * 60)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(path), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5),
        timer, budget,
    ]
    start = time.time()
    hist = model.fit(make_dataset(X_train, y_train, args.batch_size, True, args.seed),
                     validation_data=make_dataset(X_val, y_val, 1024, False, args.seed),
                     epochs=args.epochs, callbacks=callbacks, verbose=2)
    seconds = round(time.time() - start, 1)
    h = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    best = int(np.argmin(h["val_loss"]))
    record = {"model": kind, "train_seconds": seconds, "epoch_seconds": timer.times, "epochs_run": len(h["loss"]),
              "best_epoch": best + 1, "stopped_by_time_budget": budget.hit, "parameters": int(model.count_params()),
              "history": h}
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / f"history_{kind.lower()}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    log.info("%s finished in %.1f min: best epoch %d  val_loss %.4f  val_acc %.4f", kind, seconds / 60, best + 1,
             h["val_loss"][best], h["val_accuracy"][best])
    return record


def main() -> int:
    d = preprocess.DEFAULTS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["rnn", "lstm", "gru"], choices=["rnn", "lstm", "gru"])
    ap.add_argument("--seq-len", type=int, default=d["seq_len"], help="context length in tokens")
    ap.add_argument("--vocab-size", type=int, default=d["vocab_size"], help="vocabulary cap (incl. reserved ids)")
    ap.add_argument("--min-count", type=int, default=d["min_count"])
    ap.add_argument("--embedding-dim", type=int, default=96)
    ap.add_argument("--units", type=int, default=128, help="recurrent units")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", "10")))
    ap.add_argument("--patience", type=int, default=2, help="early-stopping patience (epochs)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=d["seed"])
    ap.add_argument("--val-fraction", type=float, default=d["val_fraction"])
    ap.add_argument("--max-minutes", type=float, default=0, help="per-model wall-clock budget; 0 = unlimited")
    ap.add_argument("--max-train-samples", type=int, default=0,
                    help="train on a random subset (smoke tests only; metrics are then NOT comparable)")
    ap.add_argument("--skip-eval", action="store_true", help="do not run evaluate.py afterwards")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    log.info("TensorFlow %s - %s", tf.__version__, f"{len(gpus)} GPU(s)" if gpus else "CPU only")

    stats, vocab, X_train, y_train, X_val, y_val = preprocess.load_processed(
        seq_len=args.seq_len, vocab_size=args.vocab_size, min_count=args.min_count,
        val_fraction=args.val_fraction, seed=args.seed)
    preprocess.print_report(stats)
    MODEL_DIR.mkdir(exist_ok=True)

    config_path = MODEL_DIR / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    config.update({
        "data": {"seq_len": args.seq_len, "vocab_cap": args.vocab_size, "min_count": args.min_count,
                 "val_fraction": args.val_fraction, "seed": args.seed},
        "vocab_size": len(vocab), "first_word_id": vocab.first_word_id,
        "hyperparameters": {"embedding_dim": args.embedding_dim, "units": args.units, "dropout": args.dropout,
                            "batch_size": args.batch_size, "epochs_max": args.epochs, "patience": args.patience,
                            "learning_rate": args.lr, "optimizer": "Adam (clipnorm=1.0) + ReduceLROnPlateau",
                            "loss": "sparse_categorical_crossentropy"},
        "trained_models": sorted(set(config.get("trained_models", [])) | {m.upper() for m in args.models}),
        "tensorflow": tf.__version__, "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    if args.max_train_samples and args.max_train_samples < len(X_train):
        idx = np.random.RandomState(args.seed).choice(len(X_train), args.max_train_samples, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
        log.warning("SMOKE TEST: training on a random subset of %s sequences", f"{len(X_train):,}")
        config["smoke_test_subset"] = int(len(X_train))
    else:
        config.pop("smoke_test_subset", None)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    data = (X_train, y_train, X_val, y_val)
    for name in args.models:
        log.info("=== training %s ===", name.upper())
        train_one(name.upper(), args, data, len(vocab))

    if not args.skip_eval:
        import evaluate
        evaluate.main()
    print("\nDone. Start the website with:  python app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
