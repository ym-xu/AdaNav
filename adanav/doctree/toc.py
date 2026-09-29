"""Derive a TOC from MinerU headings when the document provides none (paper §3.2).

MMLongBench-Doc ships process/toc.json and process/logical_pages.json with its
MinerU output. PaperTab and FetaTab do not; for them we read the headings MinerU
tags with ``text_level`` and infer their levels:

* PaperTab (academic papers): from the numbering scheme — "1 Intro" / "1." / "I."
  are level 1, "1.1" level 2, "1.1.1" level 3, well-known titles such as
  "Abstract" or "References" level 1, anything else level 2.
* FetaTab (Wikipedia pages): from rendered heading height. Wikipedia h2 headings
  sit on a rule that makes their boxes a few pixels taller than h3, so an Otsu
  split of the heights separates level 1 from level 2.

Logical pages are the physical pages (1:1).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


def find_content_list(doc_dir: Path) -> Optional[Path]:
    """The MinerU content list in ``doc_dir``, preferring the UUID-prefixed original."""
    matches = [m for m in doc_dir.glob("*_content_list.json") if "v2" not in m.name]
    if not matches:
        return None
    uuid = re.compile(r"^[0-9a-f]{8}-")
    return next((m for m in matches if uuid.match(m.name)), matches[0])


def identity_logical_pages(num_pages: int) -> Dict[str, Any]:
    return {"pages": [{"logical_page": str(i + 1), "physical_page": i + 1} for i in range(num_pages)]}


def build_heading_tree(flat: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Nest a flat [{level, title, logical_page}] list by level."""
    root = {"level": 0, "children": []}
    stack = [root]
    for h in flat:
        node = {"level": h["level"], "title": h["title"], "logical_page": h["logical_page"], "children": []}
        while len(stack) > 1 and stack[-1]["level"] >= node["level"]:
            stack.pop()
        stack[-1]["children"].append(node)
        stack.append(node)
    return root["children"]


def _height(elem: Dict[str, Any]) -> float:
    bbox = elem.get("bbox", [0, 0, 0, 0])
    return (bbox[3] - bbox[1]) if len(bbox) >= 4 else 0


def _page_label(elem: Dict[str, Any]) -> str:
    return str(elem.get("page_idx", 0) + 1)


class FetaTabHeadings:
    """Wikipedia headings, levelled by box height."""

    INFOBOX_X_FRACTION = 0.55     # headings starting right of this are infobox rows
    MIN_SEPARATION = 0.10         # Otsu between-class / total variance needed to split levels
    WIKI_TEMPLATE_NOISE = {
        "alerting users", "editnotices", "editnotes", "edit notices",
        "talk page notices", "miscellaneous", "page notices",
        "protection templates", "cleanup templates", "general fixes",
        "wikipedia", "contentious topics", "arbitration enforcement",
        "top icon templates", "banner shell", "wikimedia commons",
        "categories", "hidden categories",
    }

    def extract(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        headings = [e for e in elements if e.get("type") == "text" and "text_level" in e]
        if not headings:
            return []

        page_width = max((e["bbox"][2] for e in elements if len(e.get("bbox", [])) >= 4), default=0) or 1000
        headings = [h for h in headings if h.get("bbox", [0])[0] < page_width * self.INFOBOX_X_FRACTION]
        headings = [h for h in headings if h.get("text", "").strip().lower() not in self.WIKI_TEMPLATE_NOISE]

        seen, unique = set(), []
        for h in headings:
            key = (h.get("text", "").strip(), h.get("page_idx", 0))
            if key not in seen:
                seen.add(key)
                unique.append(h)
        headings = self._skip_doc_title(unique)
        if not headings:
            return []

        heights = [_height(h) for h in headings]
        threshold = 0 if len(headings) <= 3 else self._otsu_threshold(heights)
        flat = [{"level": 1 if ht > threshold else 2, "title": h.get("text", "").strip(),
                 "logical_page": _page_label(h)} for h, ht in zip(headings, heights)]
        if all(f["level"] == 2 for f in flat):
            for f in flat:
                f["level"] = 1
        return flat

    @staticmethod
    def _skip_doc_title(headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drop the first heading on page 0 when it is taller than every other heading."""
        if len(headings) < 2 or headings[0].get("page_idx", -1) != 0:
            return headings
        first = _height(headings[0])
        if first > 0 and first > max(_height(h) for h in headings[1:]):
            return headings[1:]
        return headings

    def _otsu_threshold(self, heights: List[float]) -> float:
        """Height T maximising between-class variance; heights > T are level 1. 0 = no split."""
        unique = sorted(set(heights))
        if len(unique) <= 1:
            return 0
        n = len(heights)
        mean = sum(heights) / n
        total_var = sum((h - mean) ** 2 for h in heights) / n
        best_t, best_var = 0, 0.0
        for t in unique[:-1]:
            lo = [h for h in heights if h <= t]
            hi = [h for h in heights if h > t]
            var = (len(lo) / n) * (len(hi) / n) * (sum(lo) / len(lo) - sum(hi) / len(hi)) ** 2
            if var > best_var:
                best_t, best_var = t, var
        if total_var > 0 and best_var / total_var < self.MIN_SEPARATION:
            return 0
        return best_t


class PaperTabHeadings:
    """Academic-paper headings, levelled by their numbering."""

    SUBSUB = re.compile(r"^(\d+)\.(\d+)\.(\d+)")   # 1.1.1
    SUB = re.compile(r"^(\d+)\.(\d+)")             # 1.1
    ARABIC_DOT = re.compile(r"^(\d+)\.\s+")        # 1. Intro
    ARABIC_SPACE = re.compile(r"^(\d+)\s+[A-Z]")   # 1 Intro
    ROMAN = re.compile(r"^([IVX]+)\.\s+")          # I. INTRO
    ALPHA = re.compile(r"^([A-Z])\.\s+")           # A. (level 2 under roman numbering)
    APPENDIX = re.compile(r"^Appendix", re.IGNORECASE)

    STANDARD_L1 = {
        "abstract", "references", "acknowledgments", "acknowledgements",
        "acknowledgment", "acknowledgement", "conclusion", "conclusions",
        "introduction", "related work", "bibliography",
    }
    NOISE_WORDS = {"gold", "predicted", "input", "output", "source", "target"}

    def _numbered(self, text: str) -> bool:
        return bool(self.ARABIC_SPACE.match(text) or self.ARABIC_DOT.match(text) or self.ROMAN.match(text))

    def extract(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        headings = [e for e in elements if "text_level" in e]
        if not headings:
            return []
        roman = self._convention(headings) == "roman"

        kept = []
        for h in headings:
            text = h.get("text", "").strip()
            if not text or text.lower() in self.NOISE_WORDS or re.match(r"^\([A-Z]\)", text):
                continue
            # Short unnumbered lines on the first page are author names and affiliations.
            if (h.get("page_idx", 0) == 0 and not self._numbered(text)
                    and text.lower() not in self.STANDARD_L1 and len(text.split()) <= 3):
                continue
            kept.append(h)

        if kept and kept[0].get("page_idx", -1) == 0:
            text = kept[0].get("text", "").strip()
            if not (self._numbered(text) or self.APPENDIX.match(text)) and text.lower() not in self.STANDARD_L1:
                kept = kept[1:]  # the paper title

        return [{"level": self._level(h.get("text", "").strip(), roman), "title": h.get("text", "").strip(),
                 "logical_page": _page_label(h)} for h in kept]

    def _convention(self, headings: List[Dict[str, Any]]) -> str:
        counts = Counter()
        for h in headings:
            text = h.get("text", "").strip()
            if self.ARABIC_SPACE.match(text):
                counts["arabic_space"] += 1
            if self.ARABIC_DOT.match(text):
                counts["arabic_dot"] += 1
            if self.ROMAN.match(text):
                counts["roman"] += 1
        return counts.most_common(1)[0][0] if counts else "unnumbered"

    def _level(self, text: str, roman: bool) -> int:
        if text.lower() in self.STANDARD_L1 or self.APPENDIX.match(text):
            return 1
        if self.SUBSUB.match(text):
            return 3
        if self.SUB.match(text):
            return 2
        if self._numbered(text):
            return 1
        if roman and self.ALPHA.match(text):
            return 2
        return 2


EXTRACTORS = {"papertab": PaperTabHeadings, "fetatab": FetaTabHeadings}


def derive_toc(elements: List[Dict[str, Any]], kind: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """(toc, logical_pages) for a MinerU content list, using the ``kind`` heuristics."""
    flat = EXTRACTORS[kind]().extract(elements)
    if not flat:
        flat = [{"level": 1, "title": "Document Content", "logical_page": "1"}]
    num_pages = max((e.get("page_idx", 0) for e in elements), default=0) + 1
    return {"headings": build_heading_tree(flat)}, identity_logical_pages(num_pages)


def write_toc(doc_dir: Path, kind: str, force: bool = False) -> Optional[int]:
    """Write process/toc.json and process/logical_pages.json; returns the heading count.

    Returns None when the files already exist (and not ``force``) or no content list is found.
    """
    process = doc_dir / "process"
    toc_path, lp_path = process / "toc.json", process / "logical_pages.json"
    if not force and toc_path.exists() and lp_path.exists():
        return None
    content_list = find_content_list(doc_dir)
    if content_list is None:
        print(f"[WARN] {doc_dir.name}: no *_content_list.json")
        return None
    elements = json.loads(content_list.read_text(encoding="utf-8")) or []
    toc, logical_pages = derive_toc(elements, kind)
    process.mkdir(parents=True, exist_ok=True)
    toc_path.write_text(json.dumps(toc, ensure_ascii=False, indent=2), encoding="utf-8")
    lp_path.write_text(json.dumps(logical_pages, ensure_ascii=False, indent=2), encoding="utf-8")
    return sum(1 for _ in _walk(toc["headings"]))


def _walk(nodes):
    for n in nodes:
        yield n
        yield from _walk(n.get("children", []))
