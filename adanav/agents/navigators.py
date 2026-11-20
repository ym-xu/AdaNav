"""Page navigator pi_page and the two navigators built on it (paper §3.4).

pi_page(q, S_i, B) returns up to B pages of one section, most relevant first.
  PageNavigator        text only: page summaries, entities and OCR text
  VisionPageNavigator  page summaries and entities interleaved with the page images,
                       sampled several times and merged (the default)
  TreeNavAgent  (NORMAL):  pi_page over the router's K sections with budget B_nav
  GeneralAgent  (GENERAL): pi_page over every section with budget B_gen, then the
                           refilter (paper §4.4) prunes the union: up to `refilter_max`
                           pages for LOCATING questions, `refilter_max_counting` for COUNTING
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

from adanav.agents.chat import ChatModel, Trace, estimate_tokens
from adanav.llm import image_part
from adanav.prompts import load_prompt

OCR_CHARS = 1500  # OCR text shown per page to the navigator


def page_map(tree: dict) -> Dict[int, dict]:
    """page_idx -> page (a boundary page appears in two sections; the copies are identical)."""
    pages = {}
    for section in tree.get("sections", []):
        for page in section.get("pages", []):
            pages.setdefault(page["page_idx"], page)
    return pages


def _union(lists: List[List[int]]) -> List[int]:
    out = []
    for pages in lists:
        out.extend(p for p in pages if p not in out)
    return out


def parse_page_numbers(response: str, allowed: List[int]) -> tuple:
    """(0-indexed pages in response order, tag found?). Numbers are read as 1-indexed first
    (as shown to the model), then as 0-indexed; anything outside ``allowed`` is dropped."""
    match = re.search(r"<selected_pages>(.*?)</selected_pages>", response, re.DOTALL)
    numbers = [int(n) for n in re.findall(r"\d+", match.group(1) if match else response)]
    valid, out = set(allowed), []
    for num in numbers:
        idx = num - 1 if num - 1 in valid else num if num in valid else None
        if idx is not None and idx not in out:
            out.append(idx)
    return out, bool(match)


class PageNavigator:
    """pi_page: text-only page selection inside one section."""

    def __init__(self, chat: ChatModel, workers: int = 4):
        self.chat, self.workers = chat, workers
        self.template = load_prompt("page_navigator_prompt")

    @staticmethod
    def format_pages(pages: List[dict]) -> str:
        lines = []
        for page in pages:
            lines.append(f"Page {page['page_idx'] + 1}:")
            lines.append(f"  Summary: {page.get('summary') or '(No summary available)'}")
            if page.get("entities"):
                lines.append(f"  Key Entities: {page['entities']}")
            if page.get("visual_notes"):
                lines.append("  Figures:\n    " + page["visual_notes"].replace("\n", "\n    "))
            ocr = page.get("ocr_text") or ""
            if ocr:
                lines.append(f"  OCR Text: {ocr[:OCR_CHARS]}{'...' if len(ocr) > OCR_CHARS else ''}")
            else:
                lines.append("  OCR Text: (No OCR text available)")
            lines.append("")
        return "\n".join(lines)

    def select(self, question: str, section: dict, budget: int, trace: Trace,
               image_root: Optional[Path] = None) -> List[int]:
        pages = section.get("pages", [])
        if len(pages) <= 1:
            return [p["page_idx"] for p in pages]
        prompt = self.template.format(
            SECTION_TITLE=section.get("title", ""), START_PAGE=section.get("start_page", 0) + 1,
            END_PAGE=section.get("end_page", 0) + 1, PAGE_SUMMARIES=self.format_pages(pages), USER_QUERY=question)
        reply = self.chat(prompt, step="page_nav", max_tokens=512, trace=trace, section=section["id"])
        selected, tagged = parse_page_numbers(reply.text, [p["page_idx"] for p in pages])
        if not tagged:
            trace.fallback("page_nav", "no <selected_pages> tag; read numbers from the whole response",
                           section=section["id"])
        if not selected:
            trace.fallback("page_nav", "no valid page in response; section contributes no pages",
                           section=section["id"])
        return selected[:budget]

    def navigate(self, question: str, tree: dict, section_ids: List[str], budget: int, trace: Trace,
                 image_root: Optional[Path] = None) -> List[int]:
        """Union of pi_page over the given sections, in section order."""
        by_id = {s["id"]: s for s in tree.get("sections", [])}
        sections = [by_id[sid] for sid in section_ids if sid in by_id]
        with ThreadPoolExecutor(max_workers=max(1, min(self.workers, len(sections)))) as pool:
            results = list(pool.map(lambda s: self.select(question, s, budget, trace, image_root), sections))
        return _union(results)


class VisionPageNavigator(PageNavigator):
    """pi_page with page images: each call shows summary + entities + image for every page.

    Sections over `prefilter_pages` pages are first narrowed with a text-only call. Pages
    are chunked to fit `prompt_budget` (consecutive chunks overlap by `overlap` pages); each
    chunk is sampled `samples` times at `temperature` and the selections are merged (in
    sample order, then chunk order) before the budget is applied.
    """

    IMAGE_TOKENS = 2200  # a 1275x1650 page is about 2,050 Qwen3-VL tokens

    def __init__(self, chat: ChatModel, workers: int = 4, samples: int = 3, temperature: float = 0.3,
                 prompt_budget: int = 90000, overlap: int = 2, prefilter_pages: int = 30,
                 prefilter_ratio: float = 0.5):
        super().__init__(chat, workers)
        header, footer = load_prompt("vision_page_navigator_prompt").split("{PAGES}")
        self.header, self.footer = header, footer
        self.prefilter_template = load_prompt("vision_prefilter_prompt")
        self.samples, self.temperature, self.prompt_budget, self.overlap = samples, temperature, prompt_budget, overlap
        self.prefilter_pages, self.prefilter_ratio = prefilter_pages, prefilter_ratio

    @staticmethod
    def page_text(page: dict) -> str:
        text = f"\n--- Page {page['page_idx'] + 1} ---\nSummary: {page.get('summary') or '(No summary)'}\n"
        if page.get("entities"):
            text += f"Key Entities: {page['entities']}\n"
        if page.get("visual_notes"):
            text += f"Figures:\n{page['visual_notes']}\n"
        return text

    def _chunks(self, pages: List[dict]) -> List[List[dict]]:
        cost = lambda page: estimate_tokens(self.page_text(page)) + self.IMAGE_TOKENS
        chunks, current = [], []
        for page in pages:
            if current and sum(map(cost, current)) + cost(page) > self.prompt_budget:
                chunks.append(current)
                carry = current[-self.overlap:] if self.overlap else []
                current = carry if sum(map(cost, carry)) + cost(page) <= self.prompt_budget else []
            current.append(page)
        return chunks + [current] if current else chunks

    @staticmethod
    def prefilter_text(pages: List[dict], ocr_chars: int = 1000) -> str:
        lines = []
        for page in pages:
            lines.append(f"Page {page['page_idx'] + 1}:")
            lines.append(f"  Summary: {page.get('summary') or '(No summary)'}")
            if page.get("entities"):
                lines.append(f"  Entities: {page['entities']}")
            if page.get("visual_notes"):
                lines.append("  Figures:\n    " + page["visual_notes"].replace("\n", "\n    "))
            ocr = page.get("ocr_text") or ""
            if ocr:
                lines.append(f"  OCR: {ocr[:ocr_chars]}{'...' if len(ocr) > ocr_chars else ''}")
            lines.append("")
        return "\n".join(lines)

    def _prefilter(self, question, section, pages, budget, trace) -> List[dict]:
        keep = max(budget, int(len(pages) * self.prefilter_ratio))
        prompt = self.prefilter_template.format(
            SECTION_TITLE=section.get("title", ""), START_PAGE=section.get("start_page", 0) + 1,
            END_PAGE=section.get("end_page", 0) + 1, USER_QUERY=question, KEEP=keep,
            PAGE_SUMMARIES=self.prefilter_text(pages))
        reply = self.chat(prompt, step="vision_prefilter", max_tokens=256, trace=trace, section=section["id"])
        selected, _ = parse_page_numbers(reply.text, [p["page_idx"] for p in pages])
        if not selected:
            trace.fallback("vision_prefilter", "no valid page in response; keeping every page", section=section["id"])
            return pages
        by_idx = {p["page_idx"]: p for p in pages}
        return [by_idx[i] for i in selected]

    def _content(self, question, section, chunk, chunk_idx, num_chunks, trace, image_root) -> list:
        fields = {"SECTION_TITLE": section.get("title") or "Unknown", "NUM_PAGES": len(chunk), "USER_QUERY": question,
                  "CHUNK_INFO": f" (Chunk {chunk_idx}/{num_chunks})" if num_chunks > 1 else ""}
        content = [{"type": "text", "text": self.header.format(**fields)}]
        for page in chunk:
            content.append({"type": "text", "text": self.page_text(page)})
            image = image_part(Path(image_root) / page["image_path"], detail="high") \
                if image_root is not None and page.get("image_path") else None
            if image:
                content.append(image)
            else:
                trace.fallback("vision_nav", f"no image for page {page['page_idx'] + 1}", section=section["id"])
        content.append({"type": "text", "text": self.footer.format(**fields)})
        return content

    def select(self, question: str, section: dict, budget: int, trace: Trace,
               image_root: Optional[Path] = None) -> List[int]:
        pages = section.get("pages", [])
        if len(pages) <= 1:
            return [p["page_idx"] for p in pages]
        if len(pages) > self.prefilter_pages:
            pages = self._prefilter(question, section, pages, budget, trace)
        chunks = self._chunks(pages)
        contents = [self._content(question, section, chunk, c, len(chunks), trace, image_root)
                    for c, chunk in enumerate(chunks, 1)]
        temperature = self.temperature if self.samples > 1 else None  # one sample: the model's usual 0.1

        def call(job):
            c, sample = job
            reply = self.chat(contents[c - 1], step="vision_nav", max_tokens=512, trace=trace, temperature=temperature,
                              section=section["id"], chunk=c, sample=sample)
            return parse_page_numbers(reply.text, [p["page_idx"] for p in chunks[c - 1]])[0]

        jobs = [(c, k) for c in range(1, len(chunks) + 1) for k in range(1, self.samples + 1)]
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            results = list(pool.map(call, jobs))
        return _union(results)[:budget]


class TreeNavAgent:
    def __init__(self, navigator: PageNavigator, budget: int = 3):
        self.navigator, self.budget = navigator, budget

    def retrieve(self, question: str, tree: dict, section_ids: List[str], trace: Trace,
                 image_root: Optional[Path] = None) -> List[int]:
        return self.navigator.navigate(question, tree, section_ids, self.budget, trace, image_root)


class GeneralAgent:
    def __init__(self, navigator: PageNavigator, chat: ChatModel, budget: int = 10, refilter: bool = True,
                 refilter_max: int = 5, refilter_max_counting: int = 15, prompt_budget: int = 60000):
        self.navigator, self.chat, self.budget = navigator, chat, budget
        self.refilter, self.refilter_max, self.refilter_max_counting = refilter, refilter_max, refilter_max_counting
        self.prompt_budget = prompt_budget
        self.template = load_prompt("agentic_rag/general_refilter_prompt")

    def retrieve(self, question: str, tree: dict, trace: Trace, image_root: Optional[Path] = None) -> List[int]:
        all_ids = [s["id"] for s in tree.get("sections", [])]
        candidates = self.navigator.navigate(question, tree, all_ids, self.budget, trace, image_root)
        if not self.refilter or len(candidates) <= self.refilter_max:
            return candidates
        return self._refilter(question, candidates, page_map(tree), trace)

    @staticmethod
    def format_candidates(pages: List[int], pages_by_idx: Dict[int, dict]) -> str:
        lines = []
        for idx in pages:
            page = pages_by_idx[idx]
            lines.append(f"Page {idx + 1}:")
            lines.append(f"  Summary: {page.get('summary') or '(No summary)'}")
            if page.get("entities"):
                lines.append(f"  Key Entities: {page['entities']}")
            if page.get("visual_notes"):
                lines.append("  Figures:\n    " + page["visual_notes"].replace("\n", "\n    "))
            lines.append("")
        return "\n".join(lines)

    def _refilter(self, question, candidates, pages_by_idx, trace) -> List[int]:
        # Long documents yield more candidates than one prompt holds: prune them chunk by
        # chunk until the survivors fit, then make the final call.
        while estimate_tokens(self.format_candidates(candidates, pages_by_idx)) > self.prompt_budget:
            chunks, chunk, size = [], [], 0
            for idx in candidates:
                cost = estimate_tokens(self.format_candidates([idx], pages_by_idx))
                if chunk and size + cost > self.prompt_budget:
                    chunks.append(chunk)
                    chunk, size = [], 0
                chunk.append(idx)
                size += cost
            chunks.append(chunk)
            kept = _union([self._refilter_call(question, c, pages_by_idx, trace, step="refilter_chunk")
                           for c in chunks])
            if len(kept) >= len(candidates):  # no progress: every chunk kept all its pages
                trace.fallback("refilter", "chunked refilter kept every page; truncating to the prompt budget")
                while len(kept) > 1 and estimate_tokens(self.format_candidates(kept, pages_by_idx)) > self.prompt_budget:
                    kept = kept[:-1]
            candidates = kept
            if len(candidates) <= self.refilter_max:
                return candidates
        return self._refilter_call(question, candidates, pages_by_idx, trace, step="refilter")

    def _refilter_call(self, question, candidates, pages_by_idx, trace, step) -> List[int]:
        prompt = self.template.format(USER_QUERY=question,
                                      PAGE_SUMMARIES=self.format_candidates(candidates, pages_by_idx),
                                      MAX_PAGES=self.refilter_max, MAX_PAGES_COUNTING=self.refilter_max_counting)
        reply = self.chat(prompt, step=step, max_tokens=512, trace=trace)
        m = re.search(r"<query_type>\s*(.*?)\s*</query_type>", reply.text, re.DOTALL)
        counting = bool(m and "COUNTING" in m.group(1).upper())
        limit = self.refilter_max_counting if counting else self.refilter_max
        selected, _ = parse_page_numbers(reply.text, candidates)
        if not selected:
            trace.fallback(step, "no valid page in response; keeping the first candidates")
            selected = candidates
        return selected[:limit]
