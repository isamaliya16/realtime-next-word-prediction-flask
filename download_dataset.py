"""Collect a Deep Learning text corpus.

    python download_dataset.py                 # download every enabled source (cached afterwards)
    python download_dataset.py --force         # re-download everything
    python download_dataset.py --only d2l hf   # only some sources
    python download_dataset.py --list          # show configured sources

Sources are described in ``data/sources.json`` (created on first run from the defaults below).
Edit that file to add GitHub documentation repositories, Wikipedia titles or to disable sources.
Extra plain-text files dropped into ``data/extra/*.txt`` are picked up automatically.

Each source is converted to plain prose (code, tables and URLs removed) and stored as
``data/raw/<source>/<document>.txt``.  ``data/raw/manifest.json`` records, for each document,
its origin URL, licence and word count.  Nothing is downloaded again once it is in ``data/raw``.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import re
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

from text_extract import extract, markdown_to_text

log = logging.getLogger("download")
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
SOURCES_FILE = DATA_DIR / "sources.json"
MANIFEST = RAW_DIR / "manifest.json"
USER_AGENT = "DL-NextWord-Project/1.0 (educational; https://example.org)"

PYTORCH_INC = [f"{d}/{p}.{e}" for d in ("beginner_source", "intermediate_source", "recipes_source", "advanced_source")
               for p in ("*", "*/*") for e in ("py", "rst")]

DEFAULT_SOURCES = [
    {"name": "d2l", "type": "github", "repo": "d2l-ai/d2l-en", "ref": "master",
     "title": "Dive into Deep Learning",
     "include": ["chapter_*/*.md", "index.md"],
     "exclude": ["chapter_installation/*", "chapter_references/*", "chapter_notation/*",
                 "chapter_gaussian-processes/*", "chapter_reinforcement-learning/*",
                 "chapter_appendix-tools-for-deep-learning/*"],
     "license": "CC BY-SA 4.0 (text); sample code under a modified MIT licence",
     "license_url": "https://github.com/d2l-ai/d2l-en/blob/master/LICENSE", "enabled": True},
    {"name": "hf_course", "type": "github", "repo": "huggingface/course", "ref": "main",
     "title": "Hugging Face LLM / NLP Course (English)",
     "include": ["chapters/en/*/*.mdx"], "exclude": ["chapters/en/events/*", "chapters/en/chapter0/*"],
     "license": "Apache-2.0", "license_url": "https://github.com/huggingface/course/blob/main/LICENSE",
     "enabled": True},
    {"name": "keras_io", "type": "github", "repo": "keras-team/keras-io", "ref": "master",
     "title": "Keras developer guides and code examples",
     "include": ["guides/md/*.md", "examples/*/md/*.md"], "exclude": [],
     "license": "Apache-2.0", "license_url": "https://github.com/keras-team/keras-io/blob/master/LICENSE",
     "note": "The repository archive is ~0.5 GB; set enabled=false to skip it.", "enabled": True},
    {"name": "tensorflow_docs", "type": "github", "repo": "tensorflow/docs", "ref": "master",
     "title": "TensorFlow tutorials and guides",
     "include": ["site/en/tutorials/*/*.ipynb", "site/en/tutorials/*/*.md",
                 "site/en/guide/*.ipynb", "site/en/guide/*.md"], "exclude": [],
     "license": "Apache-2.0", "license_url": "https://github.com/tensorflow/docs/blob/master/LICENSE",
     "enabled": True},
    {"name": "pytorch_tutorials", "type": "github", "repo": "pytorch/tutorials", "ref": "main",
     "title": "PyTorch tutorials",
     "include": PYTORCH_INC, "exclude": ["*index.rst", "*README*"],
     "license": "BSD-3-Clause", "license_url": "https://github.com/pytorch/tutorials/blob/main/LICENSE",
     "enabled": True},
    {"name": "llm_course", "type": "github", "repo": "mlabonne/llm-course", "ref": "main",
     "title": "LLM Course (Maxime Labonne)", "include": ["README.md"], "exclude": [],
     "license": "Apache-2.0", "license_url": "https://github.com/mlabonne/llm-course/blob/main/LICENSE",
     "enabled": True},
    {"name": "wikipedia", "type": "wikipedia", "api": "https://en.wikipedia.org/w/api.php",
     "title": "Wikipedia articles on Deep Learning topics",
     "titles": [
         "Artificial neural network", "Deep learning", "Machine learning", "Perceptron", "Multilayer perceptron",
         "Recurrent neural network", "Long short-term memory", "Gated recurrent unit", "Bidirectional recurrent neural networks",
         "Echo state network", "Convolutional neural network", "Transformer (deep learning)",
         "Attention (machine learning)", "Seq2seq", "Natural language processing", "Computer vision",
         "Backpropagation", "Gradient descent", "Stochastic gradient descent", "Adam (optimization algorithm)",
         "Vanishing gradient problem", "Exploding gradient problem", "Activation function", "Rectifier (neural networks)",
         "Softmax function", "Sigmoid function", "Loss function", "Cross-entropy", "Word embedding", "Word2vec",
         "Language model", "Large language model", "Generative artificial intelligence", "Generative pre-trained transformer",
         "BERT (language model)", "Autoencoder", "Variational autoencoder", "Generative adversarial network",
         "Diffusion model", "Overfitting", "Regularization (mathematics)", "Dropout (neural networks)",
         "Batch normalization", "Layer normalization", "Hyperparameter optimization", "Early stopping",
         "Transfer learning", "Fine-tuning (deep learning)", "Tokenization (lexical analysis)", "Byte-pair encoding",
         "Image segmentation", "Object detection", "Image classification", "Reinforcement learning",
         "Supervised learning", "Unsupervised learning", "Self-supervised learning", "Feature learning",
         "Residual neural network", "Graph neural network", "Neural machine translation", "Speech recognition",
         "Knowledge distillation", "Quantization (machine learning)", "TensorFlow", "Keras", "PyTorch",
         "Cross-validation (statistics)", "Confusion matrix", "Perplexity", "Prompt engineering",
         "Retrieval-augmented generation", "Reinforcement learning from human feedback", "Mixture of experts",
         "Neural scaling law", "Vision transformer", "Embedding (machine learning)", "Mamba (deep learning architecture)",
         "Backpropagation through time", "Neural network (machine learning)", "Learning rate", "Mini-batch gradient descent"],
     "license": "CC BY-SA 4.0 (Wikipedia contributors)",
     "license_url": "https://en.wikipedia.org/wiki/Wikipedia:Text_of_the_Creative_Commons_Attribution-ShareAlike_4.0_International_License",
     "enabled": True},
]


def load_sources() -> list[dict]:
    DATA_DIR.mkdir(exist_ok=True)
    if not SOURCES_FILE.exists():
        SOURCES_FILE.write_text(json.dumps(DEFAULT_SOURCES, indent=2), encoding="utf-8")
        log.info("Wrote default source list to %s (edit it to customise the corpus)", SOURCES_FILE)
    return json.loads(SOURCES_FILE.read_text(encoding="utf-8"))


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")[:150]


def matches(rel: str, include: list[str], exclude: list[str]) -> bool:
    return any(fnmatch.fnmatch(rel, p) for p in include) and not any(fnmatch.fnmatch(rel, p) for p in exclude)


def fetch_github(src: dict) -> list[dict]:
    """Stream the repository tarball and keep only the matching documentation files."""
    url = f"https://codeload.github.com/{src['repo']}/tar.gz/refs/heads/{src['ref']}"
    log.info("[%s] streaming %s", src["name"], url)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    docs, out_dir = [], RAW_DIR / src["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(req, timeout=120) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            rel = member.name.split("/", 1)[1] if "/" in member.name else member.name
            if not matches(rel, src["include"], src.get("exclude", [])):
                continue
            raw = tar.extractfile(member).read().decode("utf-8", errors="ignore")
            text = extract(rel, raw)
            words = len(text.split())
            if words < 50:
                continue
            name = slug(rel)
            (out_dir / f"{name}.txt").write_text(text, encoding="utf-8")
            docs.append({"source": src["name"], "doc": f"{name}.txt", "words": words,
                         "origin": f"https://github.com/{src['repo']}/blob/{src['ref']}/{rel}",
                         "license": src["license"]})
    return docs


def fetch_wikipedia(src: dict) -> list[dict]:
    out_dir = RAW_DIR / src["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    docs, titles = [], src["titles"]
    for i in range(0, len(titles), 20):  # the API accepts up to 20 extracts per request
        batch = titles[i:i + 20]
        params = {"action": "query", "format": "json", "prop": "extracts|info", "inprop": "url",
                  "explaintext": 1, "exlimit": "max", "redirects": 1, "titles": "|".join(batch)}
        url = f"{src['api']}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.load(resp)
        for page in data.get("query", {}).get("pages", {}).values():
            if "missing" in page or not page.get("extract"):
                log.warning("[wikipedia] no text for %r", page.get("title"))
                continue
            text = markdown_to_text(page["extract"], wiki=True)
            text = re.split(r"\n\n(?:References|External links|See also|Further reading|Notes|Bibliography)\.?\n\n", text)[0]
            words = len(text.split())
            if words < 80:
                continue
            name = slug(page["title"])
            (out_dir / f"{name}.txt").write_text(text, encoding="utf-8")
            docs.append({"source": src["name"], "doc": f"{name}.txt", "words": words,
                         "origin": page.get("fullurl", f"https://en.wikipedia.org/wiki/{urllib.parse.quote(page['title'])}"),
                         "license": src["license"]})
        time.sleep(0.5)
    return docs


def collect_local_extras() -> list[dict]:
    """Plain-text files the user added by hand: data/extra/*.txt|*.md (licence is their responsibility)."""
    extra_dir, docs = DATA_DIR / "extra", []
    if not extra_dir.exists():
        return docs
    out_dir = RAW_DIR / "extra"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(list(extra_dir.glob("*.txt")) + list(extra_dir.glob("*.md"))):
        raw = f.read_text(encoding="utf-8", errors="ignore")
        text = markdown_to_text(raw) if f.suffix == ".md" else raw
        (out_dir / f"{slug(f.stem)}.txt").write_text(text, encoding="utf-8")
        docs.append({"source": "extra", "doc": f"{slug(f.stem)}.txt", "words": len(text.split()),
                     "origin": str(f), "license": "user-supplied (check before redistributing)"})
    return docs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="download again even if cached")
    ap.add_argument("--only", nargs="*", help="source names to process")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sources = load_sources()
    if args.list:
        for s in sources:
            print(f"{s['name']:18s} enabled={s.get('enabled', True)}  {s['title']}  [{s['license']}]")
        return 0

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {"sources": {}, "documents": []}
    failures = []
    for src in sources:
        name = src["name"]
        if not src.get("enabled", True) or (args.only and name not in args.only):
            continue
        cached = [d for d in manifest["documents"] if d["source"] == name]
        if cached and not args.force:
            log.info("[%s] cached: %d documents (use --force to refresh)", name, len(cached))
            continue
        try:
            fetch = {"github": fetch_github, "wikipedia": fetch_wikipedia}[src["type"]]
            docs = fetch(src)
        except Exception as exc:  # network blocked, rate limit, DNS ...
            log.error("[%s] FAILED: %s: %s", name, type(exc).__name__, exc)
            failures.append(name)
            continue
        manifest["documents"] = [d for d in manifest["documents"] if d["source"] != name] + docs
        manifest["sources"][name] = {"title": src["title"], "license": src["license"],
                                     "license_url": src["license_url"], "type": src["type"],
                                     "repo_or_api": src.get("repo", src.get("api")),
                                     "retrieved": time.strftime("%Y-%m-%d"), "documents": len(docs),
                                     "words": sum(d["words"] for d in docs)}
        log.info("[%s] %d documents, %s words", name, len(docs), f"{sum(d['words'] for d in docs):,}")
        MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    extras = collect_local_extras()
    if extras:
        manifest["documents"] = [d for d in manifest["documents"] if d["source"] != "extra"] + extras
        manifest["sources"]["extra"] = {"title": "User-supplied text in data/extra", "license": "user-supplied",
                                        "license_url": "", "documents": len(extras),
                                        "words": sum(d["words"] for d in extras)}
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_sources_md(manifest)

    total_docs, total_words = len(manifest["documents"]), sum(d["words"] for d in manifest["documents"])
    print(f"\nRaw corpus: {total_docs} documents, {total_words:,} words (before cleaning/deduplication)")
    for n, s in manifest["sources"].items():
        print(f"  {n:18s} {s['documents']:5d} docs {s['words']:>10,} words  {s['license']}")
    if failures:
        print(f"\nSources that failed: {', '.join(failures)}  (see troubleshooting in README.md)")
    if total_docs == 0:
        return 1
    print("Next: python preprocess.py")
    return 0


def write_sources_md(manifest: dict) -> None:
    lines = ["# Corpus sources and licences", "",
             "Generated by `download_dataset.py`. Texts are redistributed here only as processed prose;",
             "please keep the attribution below if you share the corpus. CC BY-SA content requires",
             "attribution and share-alike for derivatives of the text itself.", "",
             "| Source | Licence | Documents | Words | Retrieved | Licence text |", "|---|---|---|---|---|---|"]
    for n, s in manifest["sources"].items():
        lines.append(f"| {s['title']} (`{n}`) | {s['license']} | {s['documents']} | {s['words']:,} | "
                     f"{s.get('retrieved', '-')} | {s.get('license_url', '')} |")
    (DATA_DIR / "SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
