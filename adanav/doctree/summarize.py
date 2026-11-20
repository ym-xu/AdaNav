"""Hierarchical summary generation (paper §3.2, *Hierarchical Summary Generation*).

1. Page summary p_j: a VLM reads the page image plus its OCR text and returns a
   <summary> and a categorised <entities> list (prompts/page_summary_prompt.txt).
2. Section index card s_i: an LLM aggregates the section's page summaries and
   entities (prompts/section_summary_prompt.txt). Figure / Table / Chart labels
   found in the page summaries are appended programmatically as a
   VISUAL_REFS_FROM_PAGES block, so the card lists them even if the LLM drops some.
3. Root summary: the section cards joined by "\\n\\n---\\n\\n".

Each stage has a config (model, prompt and its SHA-256, sampling parameters)
identified by a short fingerprint. Page summaries are cached per (doc_id,
page_idx) together with the fingerprint of the page config that produced them;
a lookup only returns entries made with the same config, so a re-run calls the
model only for pages that failed, are new, or were made with another config.
Every model call can also be appended, unparsed, to a JSONL log.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..llm import EndpointPool, image_part
from ..prompts import load_prompt
from ..runs import now_iso

VISUAL_REFS_MARKER = "\n\nVISUAL_REFS_FROM_PAGES:"
_VISUAL_REF = re.compile(r"\b(Figure|Table|Chart)\s+(\d+)\b", re.IGNORECASE)

PAGE_PROMPT = "page_summary_prompt"
SECTION_PROMPT = "section_summary_prompt"


SAMPLING_KEYS = ("max_tokens", "temperature", "top_p", "top_k", "presence_penalty", "repetition_penalty")


def stage_config(model: str, prompt_name: str, prompt_text: str, sampling: dict) -> dict:
    """Everything that determines what one summary stage produces, apart from sampling noise.

    A sampling parameter set to None is not sent, so the endpoint's default applies
    (vLLM takes it from the model's generation_config.json).
    """
    return {
        "model": model,
        "prompt": prompt_name,
        "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        **{k: sampling.get(k) for k in SAMPLING_KEYS},
    }


def config_fingerprint(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def config_conflict(recorded: Optional[dict], current: dict) -> Optional[str]:
    """A description of how ``current`` differs from ``recorded``, or None if it does not."""
    if not recorded:
        return None
    keys = sorted(set(recorded) | set(current))
    diffs = [f"{k}: {recorded.get(k)!r} -> {current.get(k)!r}" for k in keys if recorded.get(k) != current.get(k)]
    return ", ".join(diffs) or None


class PageSummaryCache:
    """Page results persisted as JSON:

        {"<doc_id>_p<page_idx>": {"summary": str, "entities": str,
                                  "config": <page config fingerprint>, "finish_reason": str}}

    ``get`` only returns an entry whose "config" matches the caller's, so entries
    made with another model or prompt (or by older versions, which stored none)
    are regenerated rather than reused.
    """

    SAVE_EVERY = 50

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, str]] = {}
        self._unsaved = 0
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[WARN] failed to load summary cache {path}: {e}")

    @staticmethod
    def key(doc_id: str, page_idx: int) -> str:
        return f"{doc_id}_p{page_idx}"

    def get(self, doc_id: str, page_idx: int, config: str) -> Optional[Dict[str, str]]:
        with self._lock:
            entry = self._data.get(self.key(doc_id, page_idx))
        return entry if entry and entry.get("config") == config else None

    def set(self, doc_id: str, page_idx: int, value: Dict[str, str]) -> None:
        with self._lock:
            self._data[self.key(doc_id, page_idx)] = value
            self._unsaved += 1
            if self._unsaved >= self.SAVE_EVERY:
                self._write()

    def count(self, config: str) -> Tuple[int, int]:
        """(entries made with ``config``, entries made with anything else)."""
        with self._lock:
            same = sum(1 for e in self._data.values() if e.get("config") == config)
            return same, len(self._data) - same

    def save(self) -> None:
        with self._lock:
            self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)
        self._unsaved = 0

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
    """Fills page, section and root summaries into a tree in place.

    ``stats`` counts, per run: model calls, cache hits, errors, and results cut
    off by ``max_tokens`` (finish_reason "length", from a new call or the cache).
    """

    def __init__(self, model: str, pool: EndpointPool, cache: Optional[PageSummaryCache] = None,
                 page_workers: int = 4, max_tokens: int = 1024, temperature: float = 0.1,
                 top_p: Optional[float] = None, top_k: Optional[int] = None,
                 presence_penalty: Optional[float] = None, repetition_penalty: Optional[float] = None,
                 raw_log: Optional[Path] = None):
        self.model = model
        self.pool = pool
        self.cache = cache
        self.page_workers = page_workers
        self.sampling = {"max_tokens": max_tokens, "temperature": temperature, "top_p": top_p, "top_k": top_k,
                         "presence_penalty": presence_penalty, "repetition_penalty": repetition_penalty}
        self.raw_log = raw_log
        self.page_prompt = load_prompt(PAGE_PROMPT)
        self.section_prompt = load_prompt(SECTION_PROMPT)
        self.page_config = stage_config(model, PAGE_PROMPT, self.page_prompt, self.sampling)
        self.section_config = stage_config(model, SECTION_PROMPT, self.section_prompt, self.sampling)
        self.page_fp = config_fingerprint(self.page_config)
        self.section_fp = config_fingerprint(self.section_config)
        self.stats: Counter = Counter()
        self._stats_lock = threading.Lock()
        self._log_lock = threading.Lock()

    def _chat(self, content) -> Tuple[str, Optional[str], Dict[str, int]]:
        """(response text, finish_reason, token usage)."""
        # OpenAI parameters go as keywords; top_k and repetition_penalty are vLLM extensions.
        standard = {k: v for k, v in self.sampling.items()
                    if v is not None and k not in ("top_k", "repetition_penalty")}
        extra = {k: self.sampling[k] for k in ("top_k", "repetition_penalty") if self.sampling[k] is not None}
        response = self.pool.next().chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            **standard,
            **({"extra_body": extra} if extra else {}),
        )
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        tokens = {"prompt": usage.prompt_tokens, "completion": usage.completion_tokens} if usage else {}
        return choice.message.content or "", getattr(choice, "finish_reason", None), tokens

    def _log(self, record: dict) -> None:
        if self.raw_log is None:
            return
        line = json.dumps({"time": now_iso(), **record}, ensure_ascii=False)
        with self._log_lock:
            self.raw_log.parent.mkdir(parents=True, exist_ok=True)
            with self.raw_log.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _count(self, *keys: str) -> None:
        with self._stats_lock:
            self.stats.update(keys)

    # ---- pages ------------------------------------------------------------

    def page_summary(self, doc_id: str, page: dict, image_root: Path) -> Dict[str, str]:
        page_idx = page.get("page_idx", 0)
        if self.cache is not None:
            cached = self.cache.get(doc_id, page_idx, self.page_fp)
            if cached:
                self._count("page_cached", *(["page_truncated"] if cached.get("finish_reason") == "length" else []))
                return cached

        content = [{"type": "text",
                    "text": self.page_prompt.format(PAGE_TEXT=page.get("ocr_text") or "(No OCR text available)")}]
        image_path = page.get("image_path", "")
        image = image_part(image_root / image_path) if image_path else None
        if image:
            content.append(image)

        meta = {"kind": "page", "doc_id": doc_id, "page_idx": page_idx, "config": self.page_fp,
                "image": image is not None}
        failed = {"summary": "", "entities": ""}  # not cached, retried on the next run
        try:
            text, finish, tokens = self._chat(content)
        except Exception as e:
            print(f"[ERROR] {doc_id} page {page_idx}: {e}")
            self._log({**meta, "error": repr(e)})
            self._count("page_errors")
            return failed
        self._log({**meta, "finish_reason": finish, "tokens": tokens, "response": text})
        self._count("page_calls", *(["page_truncated"] if finish == "length" else []))

        result = {**parse_page_response(text), "config": self.page_fp, "finish_reason": finish}
        if not result["summary"]:
            print(f"[ERROR] {doc_id} page {page_idx}: empty response")
            self._count("page_errors")
            return failed
        if self.cache is not None:
            self.cache.set(doc_id, page_idx, result)
        return result

    def summarize_pages(self, tree: dict, image_root: Path) -> None:
        # A boundary page appears in every section that shares it; summarise it once.
        copies: Dict[int, List[dict]] = {}
        for s in tree["sections"]:
            for p in s["pages"]:
                copies.setdefault(p.get("page_idx", 0), []).append(p)
        with ThreadPoolExecutor(max_workers=self.page_workers) as pool:
            futures = {pool.submit(self.page_summary, tree["doc_id"], pages[0], image_root): pages
                       for pages in copies.values()}
            for future in as_completed(futures):
                result = future.result()
                for page in futures[future]:
                    page["summary"] = result["summary"]
                    page["entities"] = result["entities"]

    # ---- sections and root ------------------------------------------------

    def section_card(self, section: dict, doc_id: str = "") -> str:
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
        meta = {"kind": "section", "doc_id": doc_id, "section_id": section.get("id", ""),
                "config": self.section_fp}
        label = f"{doc_id} section '{section.get('title', '')}'"
        try:
            text, finish, tokens = self._chat(prompt)
        except Exception as e:
            print(f"[ERROR] {label}: {e}")
            self._log({**meta, "error": repr(e)})
            self._count("section_errors")
            return ""
        self._log({**meta, "finish_reason": finish, "tokens": tokens, "response": text})
        self._count("section_calls", *(["section_truncated"] if finish == "length" else []))

        card = parse_section_response(text)
        if not card:
            print(f"[ERROR] {label}: empty response")
            self._count("section_errors")
            return ""
        return add_visual_refs(card, extract_visual_refs(page_texts, start))

    def summarize_sections(self, tree: dict) -> None:
        sections = tree["sections"]
        with ThreadPoolExecutor(max_workers=self.page_workers) as pool:
            cards = list(pool.map(lambda s: self.section_card(s, tree.get("doc_id", "")), sections))
        for section, card in zip(sections, cards):
            section["summary"] = card
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
