"""Reasoner (paper §3.5): answer from the retrieved page images and their OCR text.

The prompt lists the question, the retrieved page numbers and each page's OCR text, then
attaches the page images, each labelled with its page number. The model writes a
<scratchpad>, then <answer> or <not_answerable> (reflection prompts may also write
<query_update> + <notes>, which asks for another retrieval round).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from adanav.agents.chat import ChatModel, Trace
from adanav.llm import image_part
from adanav.prompts import load_prompt

NOT_ANSWERABLE = "Not answerable"
DOC_SUMMARY_CHARS = 2000


@dataclass
class ReasonerOutput:
    answer: str              # "" when the model asked for another round (query_update)
    analysis: str
    query_update: str = ""
    notes: str = ""
    images: int = 0          # page images actually attached


class Reasoner:
    def __init__(self, chat: ChatModel, prompt: str = "global_reasoner_prompt", max_images: int = 12,
                 max_tokens: int = 4096):
        self.chat, self.prompt_name, self.max_images, self.max_tokens = chat, prompt, max_images, max_tokens
        self.template = load_prompt(prompt)

    def build_prompt(self, question: str, pages: List[dict], doc_summary: str = "", memory: str = "") -> str:
        fields = {
            "QUESTION": question,
            "RETRIEVED_PAGE_NUMBERS": ", ".join(str(p["page_idx"] + 1) for p in pages),
            "PAGE_TEXT": "\n\n".join(f"[Page {p['page_idx'] + 1}]\n{(p.get('ocr_text') or '').strip() or '(No text extracted)'}"
                                     for p in pages),
        }
        if "{DOCUMENT_SUMMARY}" in self.template:
            summary = doc_summary or "No document summary available."
            if len(summary) > DOC_SUMMARY_CHARS:
                summary = summary[:DOC_SUMMARY_CHARS] + "\n... (truncated)"
            fields["DOCUMENT_SUMMARY"] = summary
        if "{MEMORY_NOTES}" in self.template:
            fields["MEMORY_NOTES"] = memory or "No previous iterations."
        return self.template.format(**fields)

    def answer(self, question: str, pages: List[dict], image_root: Path, trace: Trace,
               doc_summary: str = "", memory: str = "", round_: int = 1) -> ReasonerOutput:
        content = [{"type": "text", "text": self.build_prompt(question, pages, doc_summary, memory)}]
        images = 0
        for page in pages:
            if images >= self.max_images:
                break
            part = image_part(image_root / page["image_path"], detail="auto") if page.get("image_path") else None
            if part:
                content += [{"type": "text", "text": f"[Page {page['page_idx'] + 1}]"}, part]
                images += 1
        if images < len(pages):
            trace.fallback("reasoner", f"{len(pages) - images} of {len(pages)} page images not attached "
                                       f"(limit {self.max_images} or missing file)", round=round_)
        reply = self.chat(content, step="reasoner", max_tokens=self.max_tokens, trace=trace, round=round_)
        out = parse_reasoner_response(reply.text)
        out.images = images
        return out


def parse_reasoner_response(response: str) -> ReasonerOutput:
    m = re.search(r"<scratchpad>(.*?)(?:</scratchpad>|$)", response, re.DOTALL)
    analysis = m.group(1).strip() if m else ""

    m = re.search(r"<query_update>(.*?)</query_update>", response, re.DOTALL)
    if m:
        notes = re.search(r"<notes>(.*?)</notes>", response, re.DOTALL)
        return ReasonerOutput("", analysis, m.group(1).strip(), notes.group(1).strip() if notes else "")

    answers = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)
    answer = answers[-1].strip() if answers else ""
    if not answer:  # unclosed <answer> (response cut off)
        m = re.search(r"<answer>\s*(.+)", response, re.DOTALL)
        if m:
            answer = re.sub(r"</?(?:scratchpad|not_answerable|query_update|notes)[^>]*>", "", m.group(1)).strip()
    if not answer and "<not_answerable>" in response:
        answer = NOT_ANSWERABLE
    if not answer:  # no protocol tags at all
        text = response.strip()
        answer = NOT_ANSWERABLE if "not answerable" in text.lower() else answer_from_free_text(text, analysis)
    return ReasonerOutput(answer, analysis)


def answer_from_free_text(text: str, analysis: str) -> str:
    """Best-effort answer from a response that ignored the tag protocol."""
    if len(text) <= 200:
        return text
    for pattern in (r"\*\*(?:Final\s+)?Answer\s*[:：]\s*\*\*\s*(.*?)(?:\n|$)", r"\\boxed\{(.*?)\}"):
        m = re.search(pattern, text, re.IGNORECASE)
        if m and m.group(1).strip():
            return m.group(1).strip()
    m = re.search(r"(?:^|\n)\s*(?:Final\s+)?Answer\s*[:：]\s*(.*?)(?:\n|$)", text, re.IGNORECASE)
    if m and m.group(1).strip() and len(m.group(1).strip()) <= 200:
        return m.group(1).strip()
    for block in reversed(re.findall(r"```(?:\w*\n)?(.*?)```", text, re.DOTALL)):
        lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
        lines = [l for l in lines if not l.startswith(("#", "import ", "def ", "print(", ">>>"))
                 and "=" not in l and len(l) < 200]
        if lines and lines[-1].strip("\"'` "):
            return lines[-1].strip("\"'` ")
    if analysis:
        tail = text.split(analysis)[-1] if analysis in text else text
        short = [l.strip() for l in tail.split("\n") if l.strip() and len(l.strip()) < 150 and not l.strip().startswith("#")]
        if short:
            return short[-1]
    return text[:200]


def default_prompt(variant: str, iterations: int) -> str:
    """Reasoner prompt per configuration (prompts/README.md)."""
    if iterations > 1:
        return "global_reasoner_reflect_prompt"
    return "global_reasoner_prompt_simpledoc" if variant == "base" else "global_reasoner_prompt"


