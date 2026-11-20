"""AdaNav online inference (paper Algorithm 1): route, navigate, reason.

    route    = Router(q, T)                      SPECIFIC / GENERAL / NORMAL
    pages    = SpecificAgent | GeneralAgent | TreeNavAgent
    answer   = Reasoner(q, pages, root summary)

Presets
    agentic  the full system (Router + three navigators, refilter on GENERAL): vision page
             navigator, B_nav = 4, B_gen = 10
    base     "AdaNav (Base)": no rule stage and no GeneralAgent, every question goes
             through section selection and TreeNav (a global question navigates all
             sections with B_nav); base section-filter prompt, SimpleDoc reasoner prompt
With iterations > 1 the reasoner may answer <query_update>; the updated query is routed
and navigated again, new pages are added, and the reasoner runs again (at most
`iterations` reasoner calls). The paper's Algorithm 1 is the single-round case.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

from adanav.agents.chat import ChatModel, Trace
from adanav.agents.navigators import GeneralAgent, PageNavigator, TreeNavAgent, VisionPageNavigator, page_map
from adanav.agents.reasoner import NOT_ANSWERABLE, Reasoner
from adanav.agents.router import GENERAL, NORMAL, SPECIFIC, Router, SectionSelector


@dataclass
class AdaNavConfig:
    use_specific: bool = True          # rule stage + SpecificAgent
    use_general: bool = True           # GeneralAgent for g = 1 (else TreeNav over all sections)
    top_k: int = 3                     # K sections selected by the router
    page_navigator: str = "vision"     # "vision" (page images + summaries) or "text" (summaries + OCR)
    vision_samples: int = 3            # vision navigator: samples per chunk, merged
    vision_temperature: float = 0.3
    b_nav: int = 4                     # pages per section, TreeNav
    b_gen: int = 10                    # pages per section, GeneralAgent (before the refilter)
    refilter: bool = True
    refilter_max: int = 5              # LOCATING questions
    refilter_max_counting: int = 15    # COUNTING questions
    filter_prompt: str = "agentic_rag/section_filter_prompt"
    reasoner_prompt: str = "global_reasoner_prompt"
    iterations: int = 1
    max_images: int = 12
    prompt_budget: int = 60000         # estimated tokens per navigation prompt before chunking
    temperature: float = 0.1

    PRESETS = {
        "agentic": {},
        "base": {"use_specific": False, "use_general": False,
                 "filter_prompt": "section_filter_prompt", "reasoner_prompt": "global_reasoner_prompt_simpledoc"},
    }

    @classmethod
    def preset(cls, name: str, **overrides: Any) -> "AdaNavConfig":
        """A preset, with explicit overrides; several iterations default to the reflection prompt."""
        values = {**cls.PRESETS[name], **{k: v for k, v in overrides.items() if v is not None}}
        if values.get("iterations", 1) > 1 and overrides.get("reasoner_prompt") is None:
            values["reasoner_prompt"] = "global_reasoner_reflect_prompt"
        return cls(**values)

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


class AdaNav:
    def __init__(self, model: str, pool, config: Optional[AdaNavConfig] = None, page_workers: int = 4):
        self.config = config = config or AdaNavConfig()
        chat = ChatModel(model, pool, temperature=config.temperature)
        selector = SectionSelector(chat, prompt=config.filter_prompt, top_k=config.top_k,
                                   prompt_budget=config.prompt_budget)
        if config.page_navigator == "vision":
            navigator = VisionPageNavigator(chat, workers=page_workers, samples=config.vision_samples,
                                            temperature=config.vision_temperature)
        else:
            navigator = PageNavigator(chat, workers=page_workers)
        self.router = Router(selector, use_specific=config.use_specific)
        self.treenav = TreeNavAgent(navigator, budget=config.b_nav)
        self.general = GeneralAgent(navigator, chat, budget=config.b_gen, refilter=config.refilter,
                                    refilter_max=config.refilter_max,
                                    refilter_max_counting=config.refilter_max_counting,
                                    prompt_budget=config.prompt_budget)
        self.reasoner = Reasoner(chat, prompt=config.reasoner_prompt, max_images=config.max_images)

    def retrieve(self, question: str, tree: dict, trace: Trace, image_root: Optional[Path] = None) -> Dict[str, Any]:
        route = self.router.route(question, tree, trace)
        if route.category == SPECIFIC:
            handler, pages = "specific", list(route.specific.pages)
        elif route.category == GENERAL and self.config.use_general:
            handler, pages = "general", self.general.retrieve(question, tree, trace, image_root)
        else:  # NORMAL, or GENERAL without the GeneralAgent
            handler, pages = "treenav", self.treenav.retrieve(question, tree, route.sections, trace, image_root)
        out = {"category": route.category, "handler": handler, "sections": route.sections, "pages": pages}
        if route.category == SPECIFIC:
            out["specific"] = asdict(route.specific)
        return out

    def run(self, question: str, tree: dict, image_root: Path, retrieval_only: bool = False) -> Dict[str, Any]:
        """One question end to end; the record includes the trace of every model call."""
        trace = Trace(key=tree.get("doc_id"))
        record = self.retrieve(question, tree, trace, Path(image_root))
        if not retrieval_only:
            record.update(self._reason(question, tree, Path(image_root), record["pages"], trace))
        record["usage"] = trace.summary()
        record["trace"] = trace.events
        return record

    def _reason(self, question, tree, image_root, pages, trace) -> Dict[str, Any]:
        by_idx = page_map(tree)
        pages, memory, rounds = list(pages), "", []
        if not pages:
            trace.fallback("reasoner", "retrieval returned no pages")
        for r in range(1, self.config.iterations + 1):
            out = self.reasoner.answer(question, [by_idx[p] for p in pages], image_root, trace,
                                       doc_summary=tree.get("root_summary", ""), memory=memory, round_=r)
            rounds.append({"round": r, "pages": list(pages), "images": out.images, "answer": out.answer,
                           "query_update": out.query_update, "notes": out.notes})
            if out.answer:
                return {"answer": out.answer, "rounds": rounds, "final_pages": pages}
            if r == self.config.iterations or not out.query_update:
                break
            if out.notes:
                memory += f"\n[Iteration {r}] {out.notes}"
            new = [p for p in self.retrieve(out.query_update, tree, trace, image_root)["pages"] if p not in pages]
            if not new:
                break  # the update found nothing new
            pages += new
        return {"answer": NOT_ANSWERABLE, "rounds": rounds, "final_pages": pages}


__all__ = ["AdaNav", "AdaNavConfig", "GENERAL", "NORMAL", "SPECIFIC"]
