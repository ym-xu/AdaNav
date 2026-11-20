"""Question router (paper §3.3): f_rule, then f_llm.

f_rule is the SpecificAgent's reference lookup: a resolved reference routes the question
to SPECIFIC with its pages and no model call. Otherwise f_llm reads the section titles and
index cards, selects the top-K sections S* and predicts the global flag g (the model writes
ALL); g = 1 routes to GENERAL, g = 0 to NORMAL.

Documents whose cards do not fit one prompt are filtered hierarchically: each chunk
recalls candidates (or says ALL), then one rerank call picks the top K among them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from adanav.agents.chat import ChatModel, Trace, estimate_tokens
from adanav.agents.specific import SpecificAgent, SpecificResult
from adanav.prompts import load_prompt

SPECIFIC, GENERAL, NORMAL = "specific", "general", "normal"


@dataclass
class SectionChoice:
    sections: List[str]       # S*, most relevant first
    global_flag: bool         # g


@dataclass
class Route:
    category: str
    sections: List[str] = field(default_factory=list)
    specific: SpecificResult = field(default_factory=SpecificResult)


def format_sections(sections: List[dict]) -> str:
    lines = []
    for sec in sections:
        lines.append(f"=== [{sec['id']}] {sec.get('title') or 'Untitled'} "
                     f"(Pages {sec.get('start_page', 0) + 1}-{sec.get('end_page', 0) + 1}) ===")
        lines.append(sec.get("summary") or "(No summary available)")
        lines.append("")
    return "\n".join(lines)


def parse_sections(response: str, sections: List[dict]) -> tuple:
    """(valid section ids in response order, ALL said?). Ids not in ``sections`` are dropped."""
    valid = {s["id"] for s in sections}
    match = re.search(r"<selected_sections>(.*?)</selected_sections>", response, re.DOTALL)
    content = match.group(1).strip() if match else response
    if match and (content.upper() == "ALL" or "ALL" in content.upper().split()):
        return [], True
    ids = []
    for sid in re.findall(r"sec_\w+", content):
        if sid in valid and sid not in ids:
            ids.append(sid)
    return ids, False


class SectionSelector:
    """f_llm: top-K sections and the global flag."""

    def __init__(self, chat: ChatModel, prompt: str = "agentic_rag/section_filter_prompt", top_k: int = 3,
                 prompt_budget: int = 60000, max_chunk_sections: int = 40, chunk_candidates: int = 3):
        self.chat, self.top_k = chat, top_k
        self.prompt_name = prompt
        self.template = load_prompt(prompt)
        self.chunk_template = load_prompt("section_filter_chunk_prompt")
        self.rerank_template = load_prompt("section_rerank_prompt")
        self.prompt_budget, self.max_chunk_sections, self.chunk_candidates = (
            prompt_budget, max_chunk_sections, chunk_candidates)

    def select(self, question: str, tree: dict, trace: Trace) -> SectionChoice:
        sections = tree.get("sections", [])
        if len(sections) <= 1:
            return SectionChoice([s["id"] for s in sections], False)
        chunks = self._chunks(sections)
        if len(chunks) == 1:
            return self._single(question, tree, sections, trace)
        return self._chunked(question, chunks, trace)

    def _chunks(self, sections: List[dict]) -> List[List[dict]]:
        """Consecutive sections, at most max_chunk_sections per chunk and within the prompt budget."""
        chunks, current, size = [], [], 0
        for sec in sections:
            cost = estimate_tokens(format_sections([sec]))
            if current and (len(current) >= self.max_chunk_sections or size + cost > self.prompt_budget):
                chunks.append(current)
                current, size = [], 0
            current.append(sec)
            size += cost
        return chunks + [current] if current else chunks

    def _doc_overview(self, tree: dict) -> str:
        # The paper's router reads section titles and cards only. root_summary is those same
        # cards concatenated, so it is not repeated here; this line just orients the model.
        sections = tree.get("sections", [])
        return (f"A {tree.get('num_pages') or len({p['page_idx'] for s in sections for p in s['pages']})}-page "
                f"document in {len(sections)} sections; the section summaries below cover all of it.")

    def _single(self, question, tree, sections, trace) -> SectionChoice:
        prompt = self.template.format(DOC_SUMMARY=self._doc_overview(tree),
                                      SECTION_SUMMARIES=format_sections(sections), USER_QUERY=question)
        reply = self.chat(prompt, step="section_filter", max_tokens=512, trace=trace)
        ids, is_all = parse_sections(reply.text, sections)
        if is_all:
            return SectionChoice([s["id"] for s in sections], True)
        if not ids:
            trace.fallback("section_filter", "no valid section id in response; using the first K sections")
            ids = [s["id"] for s in sections]
        return SectionChoice(ids[: self.top_k], False)

    def _chunked(self, question, chunks, trace) -> SectionChoice:
        candidates, any_all = [], False
        for i, chunk in enumerate(chunks, 1):
            prompt = self.chunk_template.format(
                CHUNK_INDEX=i, NUM_CHUNKS=len(chunks), SECTION_SUMMARIES=format_sections(chunk),
                USER_QUERY=question, MAX_CANDIDATES=self.chunk_candidates)
            reply = self.chat(prompt, step="section_filter_chunk", max_tokens=256, trace=trace, chunk=i)
            ids, is_all = parse_sections(reply.text, chunk)
            if is_all:
                any_all = True  # one chunk calling the question global is enough
                continue
            if not ids:
                trace.fallback("section_filter_chunk", "no valid section id in response; keeping the whole chunk",
                               chunk=i)
                ids = [s["id"] for s in chunk]
            by_id = {s["id"]: s for s in chunk}
            candidates.extend(by_id[sid] for sid in ids[: self.chunk_candidates])
        all_sections = [s for chunk in chunks for s in chunk]
        if any_all:
            return SectionChoice([s["id"] for s in all_sections], True)
        if len(candidates) <= self.top_k:
            return SectionChoice([s["id"] for s in candidates], False)
        return SectionChoice(self._rerank(question, candidates, trace), False)

    def _rerank(self, question, candidates, trace) -> List[str]:
        # At most chunk_candidates per chunk, so this normally fits one prompt; if not, drop
        # candidates from the end (later chunks) until it does, and say so in the trace.
        dropped = 0
        while len(candidates) > self.top_k and estimate_tokens(format_sections(candidates)) > self.prompt_budget:
            candidates, dropped = candidates[:-1], dropped + 1
        if dropped:
            trace.fallback("section_rerank", f"candidates over the prompt budget; dropped the last {dropped}")
        prompt = self.rerank_template.format(NUM_CANDIDATES=len(candidates), TOP_K=self.top_k,
                                             SECTION_SUMMARIES=format_sections(candidates), USER_QUERY=question)
        reply = self.chat(prompt, step="section_rerank", max_tokens=512, trace=trace)
        ids, _ = parse_sections(reply.text, candidates)
        if not ids:
            trace.fallback("section_rerank", "no valid section id in response; keeping recall order")
            ids = [s["id"] for s in candidates]
        return ids[: self.top_k]


class Router:
    """R(q) = (c, S*, P*)."""

    def __init__(self, selector: SectionSelector, use_specific: bool = True):
        self.selector, self.use_specific = selector, use_specific
        self.specific = SpecificAgent()

    def route(self, question: str, tree: dict, trace: Trace) -> Route:
        if self.use_specific:
            found = self.specific.retrieve(question, tree)
            if found.matched:
                return Route(SPECIFIC, specific=found)
        choice = self.selector.select(question, tree, trace)
        return Route(GENERAL if choice.global_flag else NORMAL, sections=choice.sections)
