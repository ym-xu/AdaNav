"""Hierarchical summary generation (paper §3.2, *Hierarchical Summary Generation*).

1. Page summary p_j: a VLM reads the page image plus its OCR text and returns a
   <summary> and a categorised <entities> list (prompts/page_summary_prompt.txt).
2. Section index card s_i: an LLM aggregates the section's page summaries and
   entities (prompts/section_summary_prompt.txt). Figure / Table / Chart labels
   found in the page summaries are appended programmatically as a
   VISUAL_REFS_FROM_PAGES block, so the card lists them even if the LLM drops some.
3. Root summary: the section cards joined by "\\n\\n---\\n\\n".

Page summaries are cached per (doc_id, page_idx) in a JSON file, so a re-run only
calls the model for pages that failed or are new.
"""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..llm import EndpointPool, image_part
from ..prompts import load_prompt

VISUAL_REFS_MARKER = "\n\nVISUAL_REFS_FROM_PAGES:"
_VISUAL_REF = re.compile(r"\b(Figure|Table|Chart)\s+(\d+)\b", re.IGNORECASE)


class PageSummaryCache:
    """{"<doc_id>_p<page_idx>": {"summary": str, "entities": str}} persisted as JSON."""

    SAVE_EVERY = 50

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, str]] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[WARN] failed to load summary cache {path}: {e}")

    @staticmethod
    def key(doc_id: str, page_idx: int) -> str:
        return f"{doc_id}_p{page_idx}"

    def get(self, doc_id: str, page_idx: int) -> Optional[Dict[str, str]]:
        with self._lock:
            return self._data.get(self.key(doc_id, page_idx))

    def set(self, doc_id: str, page_idx: int, value: Dict[str, str]) -> None:
        with self._lock:
            self._data[self.key(doc_id, page_idx)] = value
            if len(self._data) % self.SAVE_EVERY == 0:
                self._write()

    def save(self) -> None:
        with self._lock:
            self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")

    def __len__(self) -> int:
        return len(self._data)


def _between(text: str, tag: str) -> Optional[str]:
    start, end = text.find(f"<{tag}>"), text.find(f"</{tag}>")
    if start == -1 or end == -1:
        return None
    return text[start + len(tag) + 2:end].strip()


def parse_page_response(response: str) -> Dict[str, str]:
    """<summary> and <entities> from a page response; untagged output becomes the summary."""
    if not response:
        return {"summary": "", "entities": ""}
    summary = _between(response, "summary") or ""
    entities = _between(response, "entities") or ""
    return {"summary": summary or response.strip(), "entities": entities}


def parse_section_response(response: str) -> str:
    """<section_card> (or legacy <summary>) from a section response; else the whole text."""
    if not response:
        return ""
    for tag in ("section_card", "summary"):
        found = _between(response, tag)
        if found is not None:
            return found
    return response.strip()


def extract_visual_refs(page_texts: List[str], start_page: int) -> List[Tuple[str, int]]:
    """First occurrence of each "Figure N" / "Table N" / "Chart N" with its 1-indexed page.

    Ordered figures, then tables, then charts, each by number.
    """
    seen, refs = set(), {"Figure": [], "Table": [], "Chart": []}
    for i, text in enumerate(page_texts):
        if not text:
            continue
        for m in _VISUAL_REF.finditer(text):
            kind = m.group(1).capitalize()
            label = f"{kind} {m.group(2)}"
            if label not in seen:
                seen.add(label)
                refs[kind].append((label, start_page + i + 1))
    ordered = []
    for kind in ("Figure", "Table", "Chart"):
        ordered.extend(sorted(refs[kind], key=lambda r: int(r[0].split()[-1])))
    return ordered


def add_visual_refs(card: str, refs: List[Tuple[str, int]]) -> str:
    if not refs:
        return card
    return card + VISUAL_REFS_MARKER + "\n" + "\n".join(f"- {label} (Page {page})" for label, page in refs)


def _section_page_texts(section: dict) -> List[str]:
    texts = []
    for p in section.get("pages", []):
        text = f"Page {p.get('page_idx', 0) + 1}:\n{p.get('summary', '')}"
        if p.get("entities"):
            text += f"\nEntities: {p['entities']}"
        texts.append(text)
    return texts


def root_summary(tree: dict) -> str:
    return "\n\n---\n\n".join(s["summary"] for s in tree.get("sections", []) if s.get("summary"))


class Summarizer:
    """Fills page, section and root summaries into a tree in place."""

    def __init__(self, model: str, pool: EndpointPool, cache: Optional[PageSummaryCache] = None,
                 page_workers: int = 4, max_tokens: int = 1024, temperature: float = 0.1):
        self.model = model
        self.pool = pool
        self.cache = cache
        self.page_workers = page_workers
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.page_prompt = load_prompt("page_summary_prompt")
        self.section_prompt = load_prompt("section_summary_prompt")

    def _chat(self, content) -> str:
        response = self.pool.next().chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    # ---- pages ------------------------------------------------------------

    def page_summary(self, doc_id: str, page: dict, image_root: Path) -> Dict[str, str]:
        page_idx = page.get("page_idx", 0)
        if self.cache is not None:
            cached = self.cache.get(doc_id, page_idx)
            if cached:
                return cached

        content = [{"type": "text",
                    "text": self.page_prompt.format(PAGE_TEXT=page.get("ocr_text") or "(No OCR text available)")}]
        image_path = page.get("image_path", "")
        image = image_part(image_root / image_path) if image_path else None
        if image:
            content.append(image)

        try:
            result = parse_page_response(self._chat(content))
        except Exception as e:
            print(f"[ERROR] {doc_id} page {page_idx}: {e}")
            return {"summary": "", "entities": ""}  # not cached, retried on the next run
        if self.cache is not None:
            self.cache.set(doc_id, page_idx, result)
        return result

    def summarize_pages(self, tree: dict, image_root: Path) -> None:
        pages = [p for s in tree["sections"] for p in s["pages"]]
        with ThreadPoolExecutor(max_workers=self.page_workers) as pool:
            futures = {pool.submit(self.page_summary, tree["doc_id"], p, image_root): p for p in pages}
            for future in as_completed(futures):
                page = futures[future]
                result = future.result()
                page["summary"] = result["summary"]
                page["entities"] = result["entities"]

    # ---- sections and root ------------------------------------------------

    def section_card(self, section: dict) -> str:
        page_texts = _section_page_texts(section)
        start, end = section.get("start_page", 0), section.get("end_page", 0)
        # Each entry already starts with "Page N:"; the evaluated cards were produced
        # with this second prefix, so it is kept.
        joined = "\n\n".join(f"Page {start + i + 1}: {t}" for i, t in enumerate(page_texts) if t)
        if not joined:
            return ""
        prompt = self.section_prompt.format(
            SECTION_TITLE=section.get("title", ""), START_PAGE=start + 1, END_PAGE=end + 1,
            PAGE_SUMMARIES=joined,
        )
        try:
            card = parse_section_response(self._chat(prompt))
        except Exception as e:
            print(f"[ERROR] section '{section.get('title', '')}': {e}")
            return ""
        return add_visual_refs(card, extract_visual_refs(page_texts, start))

    def summarize_sections(self, tree: dict) -> None:
        for section in tree["sections"]:
            section["summary"] = self.section_card(section)
        tree["root_summary"] = root_summary(tree)


def refresh_visual_refs(tree: dict) -> bool:
    """Recompute the VISUAL_REFS block of every section card from its page summaries (no model calls).

    Returns True if any card changed.
    """
    changed = False
    for section in tree.get("sections", []):
        card = section.get("summary", "")
        if not card:
            continue
        base = card.split(VISUAL_REFS_MARKER)[0]
        texts = [p.get("summary", "") + ("\n" + p["entities"] if p.get("entities") else "")
                 for p in section.get("pages", [])]
        new = add_visual_refs(base, extract_visual_refs(texts, section.get("start_page", 0)))
        if new != card:
            section["summary"] = new
            changed = True
    if changed:
        tree["root_summary"] = root_summary(tree)
    return changed
