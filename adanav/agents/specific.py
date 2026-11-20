"""SpecificAgent (paper §3.4): resolve explicit page / figure / table references without a model call.

References are parsed from the question with regular expressions (f_rule) and looked up
in indexes built from the tree:
  (i)   page labels: the printed label of each page (``page_label``; two-page spreads such
        as "4-5" answer to both numbers). A page reference resolves to the pages printed
        with that number, else to the physical page with that number. Trees built before
        page labels existed use the page_number elements instead, with unnumbered pages
        indexed under their physical number.
  (ii)  element index: "figure N" / "table N" in image / table captions and text, and in
        text elements that begin like a caption ("Figure 5: ..."), which is how MinerU
        often emits a caption it did not attach to its figure
  (iii) entity index: the REFERENCES line of each page's entities
The union of all matches is the evidence set.

Besides "page N", the parser reads "slide N", page ranges ("pages 5-10", "from page 3
to page 10", "the first 10 pages") and Roman-numbered tables / figures ("Table II").
Answer-format examples such as ``['Page 2', 'Page 4']`` are not references.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

MAX_RANGE_PAGES = 40   # longer ranges ("pages 400-640") are left to the section router

_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10,
          "xi": 11, "xii": 12, "xiii": 13, "xiv": 14, "xv": 15, "xvi": 16, "xvii": 17, "xviii": 18,
          "xix": 19, "xx": 20}

# ['Page 2', 'Page 4'] / ["p1","p2"]: a quoted list in an answer-format instruction
_FORMAT_EXAMPLE = re.compile(r"\[[^\[\]]*[\"'][^\[\]]*\]")
_PAGE_RANGE = re.compile(
    r"\b(?:pages?|slides?|pp?\.)\s*(\d+)\s*(?:-|–|—|to)\s*(?:(?:pages?|slides?)\s*)?(\d+)\b", re.IGNORECASE)
_FIRST_N_PAGES = re.compile(r"\bfirst\s+(\d+)\s+(?:pages|slides)\b", re.IGNORECASE)
_PAGE_PATTERNS = [
    re.compile(r"\b(?:page|slide)\s+(\d+)\b", re.IGNORECASE),
    re.compile(r"\bp\.?\s*(\d+)\b", re.IGNORECASE),
]
# coordinated references: "page 21 and 62", "pages 3, 5 and 10", "slides 12 and 14"
_PAGE_COORD_PATTERN = re.compile(
    r"\b(?:pages?|slides?)\s+(\d+)(?:\s*(?:,|and|&)\s*(?:(?:pages?|slides?)\s+)?(\d+))+\b", re.IGNORECASE)
_NUMBER = r"(\d+|[ivxIVX]+\b)"
_FIGURE_PATTERNS = [
    re.compile(r"\bfigure\s+" + _NUMBER, re.IGNORECASE),
    re.compile(r"\bfig\.?\s*(\d+)\b", re.IGNORECASE),
]
_TABLE_PATTERNS = [
    re.compile(r"\btable\s+" + _NUMBER, re.IGNORECASE),
    re.compile(r"\btab\.?\s*(\d+)\b", re.IGNORECASE),
]
_ENTITY_REF_LINE = re.compile(r"REFERENCES:\s*(.+?)(?:\n|$)", re.IGNORECASE)
# in captions, Roman numerals must be upper case ("TABLE II"), so prose like "table i.e." is not one
_REF_ITEM = re.compile(r"((?i:figure|fig|table|tab))\.?\s*(\d+|[IVX]+\b)")
_CAPTION_START = re.compile(r"^\s*((?i:figure|fig\.?|table))\s*(\d+|[IVX]+)\s*[:.|]")


@dataclass
class SpecificResult:
    pages: List[int] = field(default_factory=list)          # 0-indexed, in reference order
    matched: List[str] = field(default_factory=list)        # e.g. ["figure 3", "page 12"]
    not_found: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)        # index that resolved each match


def _number(text: str) -> str:
    """'II' -> '2', '03' -> '3'."""
    low = text.strip().lower()
    if low in _ROMAN:
        return str(_ROMAN[low])
    return low.lstrip("0") or "0"


def parse_references(question: str) -> List[tuple]:
    """[(type, number)] with type in page / figure / table, in order of first appearance per type."""
    refs, seen = [], set()

    def add(kind, num):
        if (kind, num) not in seen:
            seen.add((kind, num))
            refs.append((kind, num))

    q = _FORMAT_EXAMPLE.sub(" ", question)
    spans = []
    for m in _PAGE_RANGE.finditer(q):
        first, last = int(m.group(1)), int(m.group(2))
        if first < last and last - first < MAX_RANGE_PAGES:
            for num in range(first, last + 1):
                add("page", str(num))
        spans.append(m.span())
    for m in _FIRST_N_PAGES.finditer(q):
        for num in range(1, min(int(m.group(1)), MAX_RANGE_PAGES) + 1):
            add("page", str(num))
        spans.append(m.span())
    for start, end in sorted(spans, reverse=True):   # a too-long range is not read as two single pages
        q = q[:start] + " " + q[end:]

    for pattern in _PAGE_PATTERNS:
        for m in pattern.finditer(q):
            add("page", m.group(1))
    for m in _PAGE_COORD_PATTERN.finditer(q):
        for num in re.findall(r"\d+", m.group(0)):
            add("page", num)
    for pattern in _FIGURE_PATTERNS:
        for m in pattern.finditer(q):
            add("figure", _number(m.group(1)))
    for pattern in _TABLE_PATTERNS:
        for m in pattern.finditer(q):
            add("table", _number(m.group(1)))
    return refs


def _normalize_page_number(text: str) -> str:
    """'-2-' -> '2', '0286' -> '286'."""
    return text.strip().strip("-").strip().lstrip("0") or "0"


def _is_page_number(text: str) -> bool:
    return bool(re.fullmatch(r"0*\d+", text.strip().strip("-").strip()))


def _label_keys(label: str) -> List[str]:
    """Lookup keys of a printed label: '0286' -> ['286'], '4-5' -> ['4', '5'], 'iv' -> ['iv']."""
    label = label.strip().lower()
    m = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", label)
    if m and 0 < int(m.group(2)) - int(m.group(1)) <= 3:     # a spread, not e.g. a year span
        return [str(n) for n in range(int(m.group(1)), int(m.group(2)) + 1)]
    return [label.lstrip("0") or "0"] if label.isdigit() else [label]


def _ref_key(kind_raw: str, num: str) -> str:
    return f"{'figure' if kind_raw.lower().startswith('fig') else 'table'} {_number(num)}"


def _pages(tree: dict):
    """Each page once (boundary pages appear in two sections)."""
    seen = set()
    for section in tree.get("sections", []):
        for page in section.get("pages", []):
            if page["page_idx"] not in seen:
                seen.add(page["page_idx"])
                yield page


def has_page_labels(tree: dict) -> bool:
    return any("page_label" in page for page in _pages(tree))


def build_indexes(tree: dict) -> Dict[str, Dict[str, List[int]]]:
    page_labels: Dict[str, List[int]] = {}
    page_numbers: Dict[str, List[int]] = {}
    elements: Dict[str, List[int]] = {}
    entities: Dict[str, List[int]] = {}

    def add(index, key, idx):
        index.setdefault(key, [])
        if idx not in index[key]:
            index[key].append(idx)

    labelled = has_page_labels(tree)
    numbered = set()
    for page in _pages(tree):
        idx = page["page_idx"]
        if page.get("page_label"):
            for key in _label_keys(page["page_label"]):
                add(page_labels, key, idx)
        for elem in page.get("elements", []):
            kind = elem.get("type", "")
            if kind == "page_number" and not labelled:
                raw = (elem.get("text") or "").strip()
                if raw and _is_page_number(raw):
                    numbered.add(idx)
                    add(page_numbers, _normalize_page_number(raw), idx)
            elif kind in ("image", "table"):
                caption = elem.get("caption") or ""
                if isinstance(caption, list):
                    caption = " ".join(caption)
                for m in _REF_ITEM.finditer(f"{caption} {elem.get('text') or ''}"):
                    add(elements, _ref_key(m.group(1), m.group(2)), idx)
            else:
                m = _CAPTION_START.match(elem.get("text") or "")
                if m:
                    add(elements, _ref_key(m.group(1), m.group(2)), idx)
        line = _ENTITY_REF_LINE.search(page.get("entities") or "")
        if line:
            for m in _REF_ITEM.finditer(line.group(1)):
                add(entities, _ref_key(m.group(1), m.group(2)), idx)
    if not labelled:
        for page in _pages(tree):
            if page["page_idx"] not in numbered:
                add(page_numbers, str(page["page_idx"] + 1), page["page_idx"])
    return {"page_label": page_labels, "page_number": page_numbers, "element": elements, "entity": entities}


class SpecificAgent:
    """Rule-based lookup; no model, no state."""

    def retrieve(self, question: str, tree: dict) -> SpecificResult:
        refs = parse_references(question)
        result = SpecificResult()
        if not refs:
            return result
        indexes = build_indexes(tree)
        labelled = has_page_labels(tree)
        num_pages = tree.get("num_pages") or (max((p["page_idx"] for p in _pages(tree)), default=-1) + 1)
        for kind, num in refs:
            label = f"{kind} {num}"
            if kind == "page" and labelled:
                key = _normalize_page_number(num)
                physical = int(num) - 1
                lookups = [("page_label", indexes["page_label"].get(key, [])),
                           ("physical", [physical] if 0 <= physical < num_pages else [])]
            elif kind == "page":
                lookups = [("page_number", indexes["page_number"].get(_normalize_page_number(num), []))]
            else:  # caption match first
                lookups = [("element", indexes["element"].get(label, [])), ("entity", indexes["entity"].get(label, []))]
            for source, pages in lookups:
                if pages:
                    result.matched.append(label)
                    result.sources.append(source)
                    result.pages.extend(p for p in pages if p not in result.pages)
                    break
            else:
                result.not_found.append(label)
        return result
