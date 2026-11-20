"""Benchmark questions in one normalised form.

Each sample: qid, doc_id, question, answer, answer_format,
             evidence_pages (1-indexed, [] for unanswerable), evidence_sources (TXT/LAY/CHA/TAB/FIG).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import List

# MMLongBench-Doc evidence source labels -> paper column names
SOURCE_NAMES = {
    "Pure-text (Plain-text)": "TXT",
    "Generalized-text (Layout)": "LAY",
    "Chart": "CHA",
    "Table": "TAB",
    "Figure": "FIG",
}


def _as_list(raw) -> list:
    if isinstance(raw, list):
        return raw
    if not raw:
        return []
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        value = json.loads(raw)
    return value if isinstance(value, list) else [value]


def load_samples(path: str | Path) -> List[dict]:
    """MMLongBench-Doc samples.json; qid is the position in the file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = []
    for i, s in enumerate(raw):
        samples.append({
            "qid": f"{i:04d}",
            "doc_id": s["doc_id"][:-4] if s["doc_id"].endswith(".pdf") else s["doc_id"],
            "question": s["question"],
            "answer": str(s["answer"]),
            "answer_format": s.get("answer_format", ""),
            "evidence_pages": [int(p) for p in _as_list(s.get("evidence_pages"))],
            "evidence_sources": [SOURCE_NAMES.get(x, x) for x in _as_list(s.get("evidence_sources"))],
        })
    return samples
