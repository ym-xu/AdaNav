"""Retrieval and QA metrics as reported in the paper.

Retrieval (questions with annotated evidence pages only):
    Recall   mean fraction of gold pages retrieved
    All-Hit  fraction of questions whose gold pages are all retrieved
    F1       macro average of per-question page F1
QA: binary accuracy over all questions, and per evidence source (TXT/LAY/CHA/TAB/FIG),
page span (SIN/MUL/UNA) and route category.
A question that failed to run counts as retrieving nothing and answering wrong.
"""

from __future__ import annotations

from typing import Dict, List, Optional

SOURCES = ("TXT", "LAY", "CHA", "TAB", "FIG")
SPANS = ("SIN", "MUL", "UNA")


def page_scores(pred: List[int], gold: List[int]) -> Optional[Dict[str, float]]:
    """pred is 0-indexed, gold 1-indexed; None when there is no gold page."""
    gold_set, pred_set = set(gold), {p + 1 for p in pred}
    if not gold_set:
        return None
    tp = len(gold_set & pred_set)
    recall = tp / len(gold_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return {"recall": recall, "precision": precision, "f1": f1, "hit": float(gold_set <= pred_set)}


def span(sample: dict) -> str:
    n = len(sample["evidence_pages"])
    return "UNA" if n == 0 else "SIN" if n == 1 else "MUL"


def _mean(xs) -> Optional[float]:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def retrieval_metrics(samples: List[dict], preds: Dict[str, dict]) -> Dict[str, object]:
    rows = []
    for s in samples:
        pages = preds.get(s["qid"], {}).get("pages") or []
        scores = page_scores(pages, s["evidence_pages"])
        if scores is not None:
            rows.append({**scores, "pages": len(set(pages)), "span": span(s), "sources": s["evidence_sources"]})

    def agg(rs):
        return {"n": len(rs), **{k: _mean(r[k] for r in rs) for k in ("recall", "hit", "precision", "f1", "pages")}}

    return {
        "overall": agg(rows),
        "avg_pages_all_questions": _mean(len(set(preds.get(s["qid"], {}).get("pages") or [])) for s in samples),
        "by_span": {sp: agg([r for r in rows if r["span"] == sp]) for sp in ("SIN", "MUL")},
        "by_source": {src: agg([r for r in rows if src in r["sources"]]) for src in SOURCES},
    }


def qa_metrics(samples: List[dict], preds: Dict[str, dict], scores: Dict[str, int]) -> Dict[str, object]:
    def acc(ss):
        return {"n": len(ss), "acc": _mean(scores.get(s["qid"], 0) for s in ss)}

    categories = sorted({p.get("category") for p in preds.values() if p.get("category")})
    return {
        "overall": acc(samples),
        "by_source": {src: acc([s for s in samples if src in s["evidence_sources"]]) for src in SOURCES},
        "by_span": {sp: acc([s for s in samples if span(s) == sp]) for sp in SPANS},
        "by_category": {c: acc([s for s in samples if preds.get(s["qid"], {}).get("category") == c])
                        for c in categories},
    }


def routing(samples: List[dict], preds: Dict[str, dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for s in samples:
        c = preds.get(s["qid"], {}).get("category") or "failed"
        counts[c] = counts.get(c, 0) + 1
    return counts
