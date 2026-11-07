"""Turn markdown / MDX / Jupyter / reST / Sphinx-gallery / Wikipedia text into plain prose.

Code blocks, tables, images and URLs are dropped.  Maths and inline code are replaced by the
placeholder tokens ``<math>`` and ``<code>`` so sentence structure survives.  Every output
paragraph is separated by a blank line and ends in terminal punctuation.
"""
from __future__ import annotations

import html
import json
import re

MATH, CODE = "<math>", "<code>"

_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,}).*?^[ \t]*\1[ \t]*$", re.S | re.M)
_FRONT = re.compile(r"\A---\n.*?\n---\n", re.S)
_DISPLAY_MATH = re.compile(r"\$\$.*?\$\$|\\\[.*?\\\]", re.S)
_INLINE_MATH = re.compile(r"(?<![\\$\w])\$(?!\s)[^$\n]{1,300}?(?<!\s)\$(?!\w)|\\\(.*?\\\)")
_INLINE_CODE = re.compile(r"``[^`\n]+``|`[^`\n]+`")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)|!\[[^\]]*\]\[[^\]]*\]")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_REFLINK = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
_URL = re.compile(r"https?://\S+|www\.\S+")
_TAG = re.compile(r"</?(?!math>|code>|num>)[A-Za-z][^>]*>|<!--.*?-->", re.S)
_ROLE = re.compile(r":(numref|ref|eqref|cite|citet|citep|doc|class|func|meth|mod|attr|py:\w+):`[^`]*`")
_DIRECTIVE_LINE = re.compile(r"^\s*:[A-Za-z_]+:(`[^`]*`)?.*$")
_LIST_START = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_HEADING = re.compile(r"^\s*#{1,6}\s+")
_MDX_ANCHOR = re.compile(r"\[\[[\w\-]+\]\]")
_JINJA = re.compile(r"\{#?(?:if|else|/if|/#if)[^}]*\}|\{/?[#:/][^}]*\}")
_EMPH = re.compile(r"(\*{1,3}|_{1,3})(\S(?:.*?\S)?)\1")


_BOILERPLATE = re.compile(
    r"^(authors?|date created|last modified|description|view in colab|github source|copyright|"
    r"open in colab|run in google colab|download (notebook|python)|created by)\b[: ]", re.I)


def _find_closing_brace(s: str, i: int) -> int:
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return j
    return -1


def _strip_wiki_math(text: str) -> str:
    """Wikipedia plain-text extracts embed formulas as {\\displaystyle ...}."""
    out, i = [], 0
    while True:
        k = text.find("{\\displaystyle", i)
        if k < 0:
            out.append(text[i:])
            break
        out.append(text[i:k])
        j = _find_closing_brace(text, k)
        if j < 0:
            out.append(text[k + 14:])
            break
        out.append(f" {MATH} ")
        i = j + 1
    return "".join(out)


def _finish(paragraphs: list[str]) -> str:
    final = []
    for p in paragraphs:
        p = re.sub(r"\s+", " ", p).strip(" \t-*_#|").lstrip("> ")
        if not p or not re.search(r"[A-Za-z]{2}", p) or _BOILERPLATE.match(p):
            continue
        p = p.rstrip(",")
        if p[-1] not in ".?!:;":
            p += "."
        final.append(p)
    return "\n\n".join(final)


def markdown_to_text(text: str, wiki: bool = False) -> str:
    text = html.unescape(text.replace("\r\n", "\n").replace("\r", "\n"))
    text = _FRONT.sub("", text)
    text = _FENCE.sub("\n", text)
    if wiki:
        text = _strip_wiki_math(text)
        text = re.sub(r"^\s*=+\s*(.*?)\s*=+\s*$", r"\n\1\n", text, flags=re.M)
    text = re.sub(r":numref:`[^`]*`", "section <num>", text)
    text = re.sub(r":eqref:`[^`]*`", "<num>", text)
    text = _ROLE.sub("", text)
    text = _DISPLAY_MATH.sub(f" {MATH} ", text)
    text = _INLINE_MATH.sub(MATH, text)
    text = _INLINE_CODE.sub(CODE, text)
    text = _IMAGE.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = _REFLINK.sub(r"\1", text)
    text = re.sub(r"^\s*\[[^\]]+\]:\s+\S+.*$", "", text, flags=re.M)  # link definitions
    text = _MDX_ANCHOR.sub("", text)
    text = _JINJA.sub("", text)
    text = _TAG.sub(" ", text)
    text = _URL.sub("", text)
    text = re.sub(r"\[\^[^\]]+\]", "", text)
    text = re.sub(r"^\s*>\s?(\[![A-Z]+\])?", "", text, flags=re.M)

    paragraphs, current = [], []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("|") or _DIRECTIVE_LINE.match(line) \
                or re.fullmatch(r"[-=*_ ]{3,}", stripped):
            if current:
                paragraphs.append(" ".join(current)); current = []
            continue
        if _HEADING.match(line):
            if current:
                paragraphs.append(" ".join(current)); current = []
            paragraphs.append(_HEADING.sub("", line))  # headings are their own paragraph
            continue
        if _LIST_START.match(line):
            if current:
                paragraphs.append(" ".join(current))
            current = [_LIST_START.sub("", line).strip()]  # list item; wrapped lines are appended
            continue
        current.append(stripped)
    if current:
        paragraphs.append(" ".join(current))
    paragraphs = [_EMPH.sub(r"\2", p) for p in paragraphs]
    paragraphs = [p.replace("*", " ").replace("__", " ") for p in paragraphs]
    return _finish(paragraphs)


def ipynb_to_text(raw: str) -> str:
    try:
        nb = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    cells = nb.get("cells") or nb.get("worksheets", [{}])[0].get("cells", [])
    md = ["".join(c.get("source", [])) for c in cells if c.get("cell_type") == "markdown"]
    return markdown_to_text("\n\n".join(md))


def rst_to_text(raw: str) -> str:
    lines, out, skip_indent = raw.replace("\r\n", "\n").split("\n"), [], None
    for i, line in enumerate(lines):
        if skip_indent is not None:
            if not line.strip() or len(line) - len(line.lstrip()) > skip_indent:
                continue
            skip_indent = None
        if re.match(r"^\s*\.\.\s", line) or re.match(r"^\s*\.\.$", line):
            skip_indent = len(line) - len(line.lstrip())
            continue
        if re.fullmatch(r"[=\-~^\"'`#*+]{3,}\s*", line):
            continue
        if re.match(r"^\s*::\s*$", line) or line.rstrip().endswith("::"):
            line = line.rstrip()[:-2].rstrip() + ":"
            skip_indent = len(line) - len(line.lstrip())
        out.append(line)
    text = "\n".join(out)
    text = re.sub(r":\w+(?::\w+)?:`~?([^`<]*?)(?:\s*<[^>]*>)?`", r"\1", text)
    text = re.sub(r"``([^`]+)``", CODE, text)
    text = re.sub(r"`([^`<]+?)(?:\s*<[^>]*>)?`_{1,2}", r"\1", text)
    return markdown_to_text(text)


def sphinx_gallery_py_to_text(raw: str) -> str:
    """Prose from sphinx-gallery style .py tutorials (docstring + '# ' comment blocks)."""
    raw = raw.replace("\r\n", "\n")
    prose = []
    for quote in ('"""', "'''"):
        start = raw.find(quote)
        head = raw[:start].rstrip("r") if start >= 0 else ""
        # only treat it as a module docstring if nothing but comments/blank lines precede it
        if start >= 0 and all(not l.strip() or l.lstrip().startswith("#") for l in head.split("\n")):
            end = raw.find(quote, start + 3)
            if end > 0:
                prose.append(rst_to_text(raw[start + 3:end]))
                raw = raw[end + 3:]
            break
    block, in_block = [], False
    for line in raw.split("\n"):
        if re.match(r"^#{20,}\s*$", line) or re.match(r"^#\s*%%", line):
            if block:
                prose.append(rst_to_text("\n".join(block)))
            block, in_block = [], True
        elif in_block and line.startswith("#"):
            block.append(line[2:] if line.startswith("# ") else line[1:])
        elif in_block:
            if block:
                prose.append(rst_to_text("\n".join(block)))
            block, in_block = [], False
    if block:
        prose.append(rst_to_text("\n".join(block)))
    return "\n\n".join(p for p in prose if p)


def extract(path: str, raw: str) -> str:
    p = path.lower()
    if p.endswith(".ipynb"):
        return ipynb_to_text(raw)
    if p.endswith(".rst"):
        return rst_to_text(raw)
    if p.endswith(".py"):
        return sphinx_gallery_py_to_text(raw)
    return markdown_to_text(raw)  # .md / .mdx
