"""Page labels on built trees, and manual corrections to them.

Printed page labels come from MinerU's process/logical_pages.json (see
``MinerUDocument.page_labels``). Where that file is wrong, a correction goes in
``configs/doctree_fixes/<dataset>.json``, checked against the page images::

    {
      "<doc_id>": {
        "note": "what was wrong and how it was checked",
        "page_labels": {"1": null, "2-61": "1"}
      }
    }

Keys are 1-indexed physical pages (as a PDF viewer numbers them) or a range
"a-b". A single page takes the label as given; a range takes a numeric first
label and counts up; null removes the label.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from adanav.config import REPO_ROOT

FIXES_DIR = REPO_ROOT / "configs" / "doctree_fixes"


def fixes_path(dataset: str) -> Path:
    return FIXES_DIR / f"{dataset}.json"


def load_fixes(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def expand_page_labels(spec: Dict[str, Optional[str]]) -> Dict[int, Optional[str]]:
    """{"2-4": "1", "1": None} -> {1: "1", 2: "2", 3: "3", 0: None} (0-indexed pages)."""
    labels: Dict[int, Optional[str]] = {}
    for key, label in spec.items():
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", key.strip())
        if not m:
            labels[int(key) - 1] = label
            continue
        first, last = int(m.group(1)), int(m.group(2))
        if label is not None and not str(label).isdigit():
            raise ValueError(f"page range {key!r} needs a numeric first label, got {label!r}")
        for offset, page in enumerate(range(first, last + 1)):
            labels[page - 1] = None if label is None else str(int(label) + offset)
    return labels


def _pages(tree: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Every page dict, including both copies of a shared boundary page."""
    for section in tree.get("sections", []):
        yield from section.get("pages", [])


def set_page_labels(tree: Dict[str, Any], labels: Dict[int, Optional[str]], *, only_listed: bool = False) -> int:
    """Write ``page_label`` on every page; returns how many page dicts changed.

    With ``only_listed`` pages missing from ``labels`` keep their label; otherwise they get None.
    """
    changed = 0
    for page in _pages(tree):
        idx = page["page_idx"]
        if only_listed and idx not in labels:
            continue
        label = labels.get(idx)
        if "page_label" not in page or page["page_label"] != label:
            changed += 1
        # keep page_label next to page_idx so trees stay readable
        rest = {k: v for k, v in page.items() if k not in ("page_idx", "page_label")}
        page.clear()
        page.update({"page_idx": idx, "page_label": label, **rest})
    return changed


def apply_fixes(tree: Dict[str, Any], fixes: Dict[str, Any]) -> List[str]:
    """Apply the corrections listed for this tree's document; returns what was applied."""
    entry = fixes.get(tree.get("doc_id", ""))
    if not entry:
        return []
    applied = []
    if "page_labels" in entry:
        labels = expand_page_labels(entry["page_labels"])
        if set_page_labels(tree, labels, only_listed=True):
            applied.append(f"page_labels: {len(labels)} pages")
    return applied
