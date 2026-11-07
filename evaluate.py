"""Evaluate the saved models on the held-out validation documents and draw comparison graphs.

    python evaluate.py

Every model is scored on the *same* validation sequences (documents never seen in training) with
the same code.  Results are written to ``outputs/metrics.json`` (read by the web dashboard),
``outputs/evaluation_report.md`` and the PNG graphs in ``outputs/``.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import preprocess

log = logging.getLogger("evaluate")
MODEL_DIR, OUT_DIR = Path("models"), Path("outputs")
NAMES = ["RNN", "LSTM", "GRU"]
COLORS = {"RNN": "#7a8ca3", "LSTM": "#2563eb", "GRU": "#0f9d8a"}


def score(model, X, y, first_word_id, batch=1024):
    """Cross-entropy, top-1/top-5 accuracy; plus top-5 using only suggestable words (what the app shows)."""
    n, nll, top1, top5, top5_masked = len(X), 0.0, 0, 0, 0
    for i in range(0, n, batch):
        p = np.array(model(X[i:i + batch], training=False))  # writable copy
        t = y[i:i + batch]
        nll += float(-np.log(np.clip(p[np.arange(len(t)), t], 1e-12, 1.0)).sum())
        order = np.argsort(-p, axis=1)[:, :5]
        top1 += int((order[:, 0] == t).sum())
        top5 += int((order == t[:, None]).any(axis=1).sum())
        p[:, :first_word_id] = 0.0
        om = np.argsort(-p, axis=1)[:, :5]
        top5_masked += int((om == t[:, None]).any(axis=1).sum())
    loss = nll / n
    return {"val_loss": loss, "val_accuracy": top1 / n, "val_top5_accuracy": top5 / n,
            "val_top5_accuracy_suggestable": top5_masked / n, "perplexity": float(np.exp(loss))}


def unigram_baseline(y_train, X_val, y_val, vocab_size, first_word_id):
    """What you get without any context: always guess the most frequent training words."""
    counts = np.bincount(y_train, minlength=vocab_size).astype(np.float64)
    probs = (counts + 1) / (counts + 1).sum()
    top5 = np.argsort(-counts)[:5]
    loss = float(-np.log(probs[y_val]).mean())
    return {"val_loss": loss, "val_accuracy": float((y_val == top5[0]).mean()),
            "val_top5_accuracy": float(np.isin(y_val, top5).mean()), "perplexity": float(np.exp(loss))}


def latency_ms(model, X, runs=40):
    x = X[:1]
    model(x, training=False)
    t = time.perf_counter()
    for i in range(runs):
        model(X[i:i + 1], training=False)
    return (time.perf_counter() - t) / runs * 1000


def plot_curves(histories, key, title, ylabel, fname):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, prefix, sub in zip(axes, ("", "val_"), ("Training", "Validation")):
        for name, rec in histories.items():
            vals = rec["history"].get(prefix + key)
            if vals:
                ax.plot(range(1, len(vals) + 1), vals, marker="o", ms=4, lw=1.8, color=COLORS[name], label=name)
        ax.set_title(f"{sub} {title}")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=.25)
    axes[0].set_ylabel(ylabel)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=130)
    plt.close(fig)


def plot_bars(results, fname):
    names = list(results)
    metrics = [("val_accuracy", "Top-1 accuracy (%)", 100), ("val_top5_accuracy", "Top-5 accuracy (%)", 100),
               ("val_loss", "Validation loss", 1), ("train_seconds", "Training time (min)", 1 / 60)]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, (key, label, mult) in zip(axes, metrics):
        vals = [results[n][key] * mult for n in names]
        bars = ax.bar(names, vals, color=[COLORS[n] for n in names], width=.6)
        ax.set_title(label, fontsize=10)
        ax.bar_label(bars, fmt="%.2f", fontsize=9)
        ax.grid(axis="y", alpha=.25)
        ax.set_ylim(0, max(vals) * 1.18)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=130)
    plt.close(fig)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    d = config["data"]
    stats, vocab, X_train, y_train, X_val, y_val = preprocess.load_processed(
        seq_len=d["seq_len"], vocab_size=d["vocab_cap"], min_count=d["min_count"],
        val_fraction=d["val_fraction"], seed=d["seed"])
    OUT_DIR.mkdir(exist_ok=True)

    results, histories = {}, {}
    for name in NAMES:
        path, hist_path = MODEL_DIR / f"{name.lower()}.keras", OUT_DIR / f"history_{name.lower()}.json"
        if not path.exists() or not hist_path.exists():
            log.warning("skipping %s (model or history missing)", name)
            continue
        model = tf.keras.models.load_model(path)  # the best checkpoint, exactly what the website serves
        rec = json.loads(hist_path.read_text(encoding="utf-8"))
        histories[name] = rec
        b, h = rec["best_epoch"] - 1, rec["history"]
        res = score(model, X_val, y_val, vocab.first_word_id)
        res.update({
            "train_loss": h["loss"][b], "train_accuracy": h["accuracy"][b],
            "keras_val_loss_at_best_epoch": h["val_loss"][b],
            "epochs_run": rec["epochs_run"], "best_epoch": rec["best_epoch"],
            "train_seconds": rec["train_seconds"], "parameters": rec["parameters"],
            "stopped_by_time_budget": rec.get("stopped_by_time_budget", False),
            "inference_ms_per_prediction": round(latency_ms(model, X_val), 2),
        })
        results[name] = res
        log.info("%s: val_loss %.4f  acc %.2f%%  top5 %.2f%%  ppl %.1f", name, res["val_loss"],
                 res["val_accuracy"] * 100, res["val_top5_accuracy"] * 100, res["perplexity"])

    if not results:
        log.error("No trained models found - run python train.py first")
        return 1
    baseline = unigram_baseline(y_train, X_val, y_val, len(vocab), vocab.first_word_id)
    metrics = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": ("Validation accuracy measures exact next-word agreement with held-out documents. It is a "
                 "classification metric, not a guarantee of fluent or correct language."),
        "smoke_test": bool(config.get("smoke_test_subset")),
        "dataset": {"documents": stats["clean"]["documents"], "words": stats["clean"]["words"],
                    "vocabulary_size": stats["vocabulary"]["size"],
                    "train_sequences": stats["sequences"]["train"], "validation_sequences": stats["sequences"]["validation"],
                    "sequence_length": stats["sequences"]["sequence_length"],
                    "train_documents": stats["split"]["train_documents"], "val_documents": stats["split"]["val_documents"]},
        "hyperparameters": config["hyperparameters"],
        "models": results, "baseline_unigram": baseline,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plot_curves(histories, "loss", "loss", "Cross-entropy loss", "loss_curves.png")
    plot_curves(histories, "accuracy", "accuracy", "Top-1 accuracy", "accuracy_curves.png")
    plot_bars(results, "model_comparison.png")

    lines = ["# Evaluation report", "", f"Generated {metrics['generated_at']}.", "",
             "All models were evaluated on the same held-out validation documents.", "",
             "| Model | Val loss | Val acc | Val top-5 | Perplexity | Train loss | Train acc | Epochs (best) | Time (min) | Params |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in results.items():
        lines.append(f"| {n} | {r['val_loss']:.4f} | {r['val_accuracy']:.2%} | {r['val_top5_accuracy']:.2%} | "
                     f"{r['perplexity']:.1f} | {r['train_loss']:.4f} | {r['train_accuracy']:.2%} | "
                     f"{r['epochs_run']} ({r['best_epoch']}) | {r['train_seconds'] / 60:.1f} | {r['parameters']:,} |")
    lines += ["", f"Unigram baseline (always guess most frequent words): top-1 {baseline['val_accuracy']:.2%}, "
              f"top-5 {baseline['val_top5_accuracy']:.2%}, perplexity {baseline['perplexity']:.1f}.", "",
              metrics["note"], "",
              "Training accuracy is measured with dropout active, so it is not directly comparable to validation accuracy."]
    (OUT_DIR / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
