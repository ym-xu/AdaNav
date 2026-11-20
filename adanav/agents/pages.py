"""Figure descriptions shown alongside a page's summary (trees from scripts/add_visual_descriptions.py)."""

from __future__ import annotations

import threading

VISUAL_NOTE_CHARS = 1200
_lock = threading.Lock()


def visual_notes(page: dict, max_chars: int = VISUAL_NOTE_CHARS) -> str:
    """One line per described image or table on the page: "- <caption or type>: <description>"."""
    lines = []
    for elem in page.get("elements", []):
        desc = (elem.get("description") or "").strip()
        if not desc:
            continue
        caption = elem.get("caption") or ""
        if isinstance(caption, list):
            caption = " ".join(caption)
        label = caption.strip()[:150] or elem.get("type", "figure")
        lines.append(f"- {label}: {desc}")
    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[:max_chars] + "..."


def add_visual_notes(tree: dict) -> None:
    """Store visual_notes on every page of the tree, once."""
    if tree.get("_visual_notes"):
        return
    with _lock:
        if tree.get("_visual_notes"):
            return
        for section in tree.get("sections", []):
            for page in section.get("pages", []):
                notes = visual_notes(page)
                if notes:
                    page["visual_notes"] = notes
        tree["_visual_notes"] = True
