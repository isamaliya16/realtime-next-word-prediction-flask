"""Clean the raw corpus, split it, build the vocabulary and generate next-word training sequences.

    python preprocess.py                              # defaults: seq-len 12, vocab 12000
    python preprocess.py --seq-len 16 --vocab-size 8000 --min-count 3

Pipeline (all steps are deterministic for a given seed):
  1. read ``data/raw/**/*.txt`` (written by download_dataset.py)
  2. normalise text, drop non-English / mostly-math / boilerplate paragraphs
  3. drop duplicate paragraphs and duplicate documents (global, before splitting)
  4. split *documents* into train / validation per source  -> no sliding-window leakage
  5. build the vocabulary from the TRAINING split only
  6. create sliding-window (context -> next word) sequences separately for each split
  7. save ``data/processed/corpus.txt``, ``docs.jsonl``, ``sequences.npz``, ``stats.json``
     and ``models/tokenizer.json``

This module has no TensorFlow dependency, so the Flask app can import ``Vocabulary`` cheaply.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import random
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

log = logging.getLogger("preprocess")

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
MODEL_DIR = Path("models")

PAD, UNK, NUM, MATH, CODE = "<pad>", "<unk>", "<num>", "<math>", "<code>"
SPECIALS = [PAD, UNK, NUM, MATH, CODE]
PUNCT = [".", ",", ";", ":", "?", "!"]
FIRST_WORD_ID = len(SPECIALS) + len(PUNCT)  # ids below this are never suggested to the user

DEFAULTS = dict(seq_len=12, vocab_size=12000, min_count=3, val_fraction=0.10, seed=42,
                min_paragraph_words=4, min_doc_words=100)

TOKEN_RE = re.compile(
    r"<(?:math|code|num)>"                                   # placeholders written by text_extract
    r"|\d+(?:[.,]\d+)*[a-z][a-z0-9]*(?:[-'][a-z0-9]+)*"       # 3d, 1x1, 2d-convolution
    r"|\d+(?:[.,]\d+)*%?"                                     # numbers
    r"|[a-z][a-z0-9]*(?:[-'][a-z0-9]+)*"                      # words incl. gpt-3, don't, word2vec
    r"|[.,;:!?]")                                             # the only punctuation we keep

_ABBREV = [(r"\be\.g\.", "for example"), (r"\bi\.e\.", "that is"), (r"\bet al\.", "et al"),
           (r"\betc\.", "etc"), (r"\bvs\.", "vs"), (r"\bfig\.", "fig"), (r"\beq\.", "eq"),
           (r"\bcf\.", "cf"), (r"\bapprox\.", "approx")]
_URL = re.compile(r"https?://\S+|www\.\S+")
_BOILER = re.compile(
    r"licensed under the apache license|you may not use this file|without warranties or conditions|"
    r"spdx-license|all rights reserved|google colab|view on tensorflow\.org|download notebook|"
    r"open in colab|run in google|view source on github|see the license for the specific|"
    r"pull request|click here to|\bpip install\b|join our discord|subscribe to", re.I)


# ----------------------------------------------------------------------------- tokenisation
def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = (text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
            .replace("\u2013", " - ").replace("\u2014", " - ").replace("\u2026", "."))
    text = _URL.sub(" ", text).lower()
    for pat, rep in _ABBREV:
        text = re.sub(pat, rep, text)
    return text


def tokenize(text: str) -> list[str]:
    toks = TOKEN_RE.findall(normalize(text))
    return [NUM if (t[0].isdigit() and not re.search(r"[a-z]", t)) else t for t in toks]


class Vocabulary:
    """Word-level vocabulary. ids 0..10 are reserved: <pad>, <unk>, <num>, <math>, <code> and punctuation."""

    def __init__(self, id_to_token: list[str]):
        self.id_to_token = id_to_token
        self.token_to_id = {t: i for i, t in enumerate(id_to_token)}
        self.pad_id, self.unk_id = 0, 1
        self.first_word_id = FIRST_WORD_ID

    def __len__(self):
        return len(self.id_to_token)

    def encode_tokens(self, tokens: list[str]) -> list[int]:
        get, unk = self.token_to_id.get, self.unk_id
        return [get(t, unk) for t in tokens]

    def encode(self, text: str) -> list[int]:
        return self.encode_tokens(tokenize(text))

    def is_word_id(self, i: int) -> bool:
        return i >= self.first_word_id

    @classmethod
    def build(cls, token_lists, vocab_size: int, min_count: int) -> "Vocabulary":
        counts = collections.Counter(t for toks in token_lists for t in toks)
        words = [(w, c) for w, c in counts.items() if w not in SPECIALS and w not in PUNCT and c >= min_count]
        words.sort(key=lambda wc: (-wc[1], wc[0]))
        words = words[: max(0, vocab_size - FIRST_WORD_ID)]
        return cls(SPECIALS + PUNCT + [w for w, _ in words])

    def save(self, path: Path, extra: dict | None = None):
        payload = {"version": 1, "first_word_id": self.first_word_id, "id_to_token": self.id_to_token}
        payload.update(extra or {})
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["id_to_token"])


# ----------------------------------------------------------------------------- cleaning
def load_raw_documents() -> list[dict]:
    manifest_path = RAW_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("data/raw/manifest.json not found - run: python download_dataset.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    docs = []
    for d in sorted(manifest["documents"], key=lambda d: (d["source"], d["doc"])):
        path = RAW_DIR / d["source"] / d["doc"]
        if path.exists():
            docs.append({**d, "text": path.read_text(encoding="utf-8", errors="ignore")})
    return docs


def paragraph_ok(raw: str, toks: list[str], min_words: int) -> str | None:
    """Return None when the paragraph is acceptable, else the reason it is rejected."""
    words = [t for t in toks if t not in PUNCT and t not in (NUM, MATH, CODE)]
    if len(words) < min_words:
        return "too_short"
    if len(words) < 0.5 * len([t for t in toks if t not in PUNCT]):
        return "mostly_math_or_code"
    letters = [c for c in raw if c.isalpha()]
    if letters and sum(1 for c in letters if ord(c) > 0x24F) / len(letters) > 0.1:
        return "non_english"
    if _BOILER.search(raw):
        return "boilerplate"
    if len(set(words)) < 0.3 * len(words) and len(words) > 8:
        return "repetitive"
    return None


def clean_documents(raw_docs: list[dict], min_paragraph_words: int, min_doc_words: int):
    seen_paragraphs, seen_docs = set(), set()
    removed = collections.Counter()
    docs = []
    for d in raw_docs:
        kept = []
        for para in re.split(r"\n\s*\n", d["text"]):
            toks = tokenize(para)
            if not toks:
                continue
            reason = paragraph_ok(para, toks, min_paragraph_words)
            if reason:
                removed[reason] += 1
                continue
            key = hashlib.md5(" ".join(toks).encode()).hexdigest()
            if key in seen_paragraphs:
                removed["duplicate_paragraph"] += 1
                continue
            seen_paragraphs.add(key)
            if toks[-1] not in PUNCT:
                toks.append(".")
            kept.append(toks)
        flat = [t for toks in kept for t in toks]
        n_words = sum(1 for t in flat if t not in PUNCT)
        if n_words < min_doc_words:
            removed["short_document"] += 1
            continue
        dkey = hashlib.md5(" ".join(flat).encode()).hexdigest()
        if dkey in seen_docs:
            removed["duplicate_document"] += 1
            continue
        seen_docs.add(dkey)
        docs.append({"source": d["source"], "doc": d["doc"], "origin": d.get("origin", ""),
                     "tokens": flat, "words": n_words})
    return docs, dict(removed)


def split_documents(docs: list[dict], val_fraction: float, seed: int) -> None:
    """Assign each *document* to train or validation, per source, so every source is represented."""
    rng = random.Random(seed)
    by_source = collections.defaultdict(list)
    for d in docs:
        by_source[d["source"]].append(d)
    for src, items in sorted(by_source.items()):
        items.sort(key=lambda d: d["doc"])
        rng.shuffle(items)
        total, val_words = sum(d["words"] for d in items), 0
        for d in items:
            d["split"] = "train"
        if len(items) < 5:  # too small to hold anything out
            continue
        for d in items:
            if val_words >= val_fraction * total:
                break
            d["split"] = "val"
            val_words += d["words"]


# ----------------------------------------------------------------------------- sequences
def make_windows(id_lists: list[np.ndarray], seq_len: int):
    """Sliding windows (previous <= seq_len tokens -> next token), never crossing document borders.

    Windows are left-padded with <pad>=0.  Targets must be regular words: <unk>, <num>, <math>,
    <code> and punctuation are valid *context* but are never prediction targets.
    """
    xs, ys = [], []
    for ids in id_lists:
        n = len(ids)
        if n < 2:
            continue
        padded = np.concatenate([np.zeros(seq_len, dtype=np.int32), ids.astype(np.int32)])
        windows = np.lib.stride_tricks.sliding_window_view(padded, seq_len)[:n]  # row i = context of ids[i]
        keep = np.arange(n) >= 1
        keep &= ids >= FIRST_WORD_ID
        xs.append(windows[keep])
        ys.append(ids[keep])
    if not xs:
        return np.zeros((0, seq_len), np.int32), np.zeros((0,), np.int32)
    return np.concatenate(xs).astype(np.int32), np.concatenate(ys).astype(np.int32)


def params_match(stats: dict, p: dict) -> bool:
    cfg = stats.get("config", {})
    return all(cfg.get(k) == p[k] for k in ("seq_len", "vocab_size", "min_count", "val_fraction", "seed"))


def build(seq_len=DEFAULTS["seq_len"], vocab_size=DEFAULTS["vocab_size"], min_count=DEFAULTS["min_count"],
          val_fraction=DEFAULTS["val_fraction"], seed=DEFAULTS["seed"],
          min_paragraph_words=DEFAULTS["min_paragraph_words"], min_doc_words=DEFAULTS["min_doc_words"]) -> dict:
    cfg = dict(seq_len=seq_len, vocab_size=vocab_size, min_count=min_count, val_fraction=val_fraction, seed=seed,
               min_paragraph_words=min_paragraph_words, min_doc_words=min_doc_words)
    raw_docs = load_raw_documents()
    raw_chars = sum(len(d["text"]) for d in raw_docs)
    raw_words = sum(len(d["text"].split()) for d in raw_docs)
    log.info("raw: %d documents, %s words", len(raw_docs), f"{raw_words:,}")

    docs, removed = clean_documents(raw_docs, min_paragraph_words, min_doc_words)
    if not docs:
        raise RuntimeError("No documents survived cleaning - check data/raw.")
    split_documents(docs, val_fraction, seed)
    train_docs = [d for d in docs if d["split"] == "train"]
    val_docs = [d for d in docs if d["split"] == "val"]

    vocab = Vocabulary.build((d["tokens"] for d in train_docs), vocab_size, min_count)
    enc = lambda ds: [np.asarray(vocab.encode_tokens(d["tokens"]), dtype=np.int32) for d in ds]
    train_ids, val_ids = enc(train_docs), enc(val_docs)
    X_train, y_train = make_windows(train_ids, seq_len)
    X_val, y_val = make_windows(val_ids, seq_len)

    all_tokens = collections.Counter(t for d in docs for t in d["tokens"])
    train_flat = np.concatenate(train_ids)
    train_word_tokens = int((train_flat >= FIRST_WORD_ID).sum())
    val_flat = np.concatenate(val_ids) if val_ids else np.zeros(0, np.int32)
    n_words_total = sum(d["words"] for d in docs)
    stats = {
        "config": cfg,
        "raw": {"documents": len(raw_docs), "words": raw_words, "characters": raw_chars},
        "removed_paragraphs": removed,
        "clean": {"documents": len(docs), "words": n_words_total,
                  "tokens_incl_punctuation": sum(len(d["tokens"]) for d in docs),
                  "characters": sum(len(" ".join(d["tokens"])) for d in docs),
                  "unique_tokens": len(all_tokens), "unique_words": sum(1 for t in all_tokens if t not in SPECIALS and t not in PUNCT)},
        "vocabulary": {"size": len(vocab), "reserved_ids": FIRST_WORD_ID, "min_count": min_count,
                       "train_word_coverage": round(float((train_flat >= FIRST_WORD_ID).sum()) / max(1, int(((train_flat >= FIRST_WORD_ID) | (train_flat == vocab.unk_id)).sum())), 4),
                       "val_word_coverage": round(float((val_flat >= FIRST_WORD_ID).sum()) / max(1, int(((val_flat >= FIRST_WORD_ID) | (val_flat == vocab.unk_id)).sum())), 4)},
        "split": {"method": "document-level, per source, before windowing; vocabulary built from train only",
                  "train_documents": len(train_docs), "val_documents": len(val_docs),
                  "train_words": sum(d["words"] for d in train_docs), "val_words": sum(d["words"] for d in val_docs)},
        "sequences": {"sequence_length": seq_len, "train": int(len(X_train)), "validation": int(len(X_val)),
                      "train_word_tokens": train_word_tokens},
        "by_source": {},
    }
    for src in sorted({d["source"] for d in docs}):
        sd = [d for d in docs if d["source"] == src]
        stats["by_source"][src] = {"documents": len(sd), "words": sum(d["words"] for d in sd),
                                   "val_documents": sum(1 for d in sd if d["split"] == "val")}

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROC_DIR / "corpus.txt", "w", encoding="utf-8") as f:   # one cleaned document per line
        for d in docs:
            f.write(" ".join(d["tokens"]) + "\n")
    with open(PROC_DIR / "docs.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({k: d[k] for k in ("source", "doc", "origin", "words", "split")}) + "\n")
    np.savez_compressed(PROC_DIR / "sequences.npz", X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    vocab.save(MODEL_DIR / "tokenizer.json", {"seq_len": seq_len, "specials": SPECIALS, "punctuation": PUNCT})
    (PROC_DIR / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def load_processed(**params):
    """Load cached sequences, rebuilding them when the requested parameters differ from the cache."""
    p = {**DEFAULTS, **params}
    stats_path = PROC_DIR / "stats.json"
    if stats_path.exists() and (PROC_DIR / "sequences.npz").exists() and (MODEL_DIR / "tokenizer.json").exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if params_match(stats, p):
            z = np.load(PROC_DIR / "sequences.npz")
            return stats, Vocabulary.load(MODEL_DIR / "tokenizer.json"), z["X_train"], z["y_train"], z["X_val"], z["y_val"]
        log.info("cached dataset was built with different settings - rebuilding")
    stats = build(**p)
    z = np.load(PROC_DIR / "sequences.npz")
    return stats, Vocabulary.load(MODEL_DIR / "tokenizer.json"), z["X_train"], z["y_train"], z["X_val"], z["y_val"]


def print_report(stats: dict) -> None:
    c, r, v, s, sp = stats["clean"], stats["raw"], stats["vocabulary"], stats["sequences"], stats["split"]
    print("\n================ DATASET REPORT ================")
    print(f"Raw documents            : {r['documents']:,}   words: {r['words']:,}   characters: {r['characters']:,}")
    print(f"Removed paragraphs       : {stats['removed_paragraphs']}")
    print(f"Clean documents          : {c['documents']:,}")
    print(f"Clean words              : {c['words']:,}  (tokens incl. punctuation: {c['tokens_incl_punctuation']:,})")
    print(f"Clean characters         : {c['characters']:,}")
    print(f"Unique words in corpus   : {c['unique_words']:,}")
    print(f"Model vocabulary size    : {v['size']:,}  (min_count={v['min_count']}; train word coverage {v['train_word_coverage']:.1%},"
          f" validation coverage {v['val_word_coverage']:.1%})")
    print(f"Split (document-level)   : train {sp['train_documents']} docs / {sp['train_words']:,} words;"
          f" validation {sp['val_documents']} docs / {sp['val_words']:,} words")
    print(f"Sequences (len {s['sequence_length']})      : train {s['train']:,}   validation {s['validation']:,}")
    for src, i in stats["by_source"].items():
        print(f"   {src:18s} {i['documents']:4d} docs {i['words']:>9,} words ({i['val_documents']} held out)")
    print("================================================\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq-len", type=int, default=DEFAULTS["seq_len"])
    ap.add_argument("--vocab-size", type=int, default=DEFAULTS["vocab_size"])
    ap.add_argument("--min-count", type=int, default=DEFAULTS["min_count"])
    ap.add_argument("--val-fraction", type=float, default=DEFAULTS["val_fraction"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print_report(build(a.seq_len, a.vocab_size, a.min_count, a.val_fraction, a.seed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
