"""Summary generation with a scripted fake model (no network).

Run: python -m unittest discover tests
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adanav.doctree.summarize import (
    PageSummaryCache, Summarizer, config_conflict, extract_visual_refs, parse_page_response,
    parse_section_response, refresh_visual_refs,
)


class FakePool:
    """Stands in for EndpointPool: records prompts, answers from a callback."""

    def __init__(self, answer, finish_reason="stop"):
        self.answer, self.calls, self.finish_reason = answer, [], finish_reason
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def next(self):
        return self

    def _create(self, model, messages, **sampling):
        content = messages[0]["content"]
        self.calls.append(content)
        self.sampling = sampling
        text = self.answer(content)
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20)
        return SimpleNamespace(usage=usage, choices=[SimpleNamespace(
            message=SimpleNamespace(content=text), finish_reason=self.finish_reason)])


def page_answer(content):
    if isinstance(content, list):  # page request: [text, image?]
        return "<summary>Shows Figure 2 and Table 1.</summary><entities>REFERENCES: Figure 2</entities>"
    return "<section_card>TOPICS: results</section_card>"


def make_tree():
    page = lambda i: {"page_idx": i, "image_path": "", "ocr_text": f"text {i}", "summary": "", "elements": []}
    return {"doc_id": "d", "root_summary": "", "sections": [
        {"id": "sec_00", "title": "A", "start_page": 0, "end_page": 1, "summary": "", "pages": [page(0), page(1)]},
        {"id": "sec_01", "title": "B", "start_page": 1, "end_page": 1, "summary": "", "pages": [page(1)]},
    ]}


class SummarizeTest(unittest.TestCase):
    def test_parsers(self):
        self.assertEqual(parse_page_response("<summary>s</summary><entities>e</entities>"),
                         {"summary": "s", "entities": "e"})
        self.assertEqual(parse_page_response("no tags"), {"summary": "no tags", "entities": ""})
        self.assertEqual(parse_section_response("x<section_card>c</section_card>"), "c")
        self.assertEqual(parse_section_response("<summary>old</summary>"), "old")

    def test_visual_refs_order_and_pages(self):
        refs = extract_visual_refs(["Table 10 and Figure 3", "", "table 2, Figure 3, Chart 1"], start_page=4)
        self.assertEqual(refs, [("Figure 3", 5), ("Table 2", 7), ("Table 10", 5), ("Chart 1", 7)])

    def test_full_tree(self):
        pool = FakePool(page_answer)
        with tempfile.TemporaryDirectory() as tmp:
            cache = PageSummaryCache(Path(tmp) / "c.json")
            s = Summarizer("m", pool, cache=cache, page_workers=2)
            tree = make_tree()
            s.summarize_pages(tree, Path(tmp))
            s.summarize_sections(tree)

        page0 = tree["sections"][0]["pages"][0]
        self.assertEqual(page0["summary"], "Shows Figure 2 and Table 1.")
        self.assertEqual(page0["entities"], "REFERENCES: Figure 2")
        card = tree["sections"][0]["summary"]
        self.assertEqual(card, "TOPICS: results\n\nVISUAL_REFS_FROM_PAGES:\n- Figure 2 (Page 1)\n- Table 1 (Page 1)")
        self.assertEqual(tree["root_summary"], card + "\n\n---\n\n" + tree["sections"][1]["summary"])

        # Section prompt keeps the evaluated double "Page N:" prefix.
        section_prompt = [c for c in pool.calls if isinstance(c, str)][0]
        self.assertIn("Page 1: Page 1:\nShows Figure 2", section_prompt)
        # Page 1 appears in both sections but is summarised once: 2 page + 2 section calls.
        self.assertEqual(len(pool.calls), 4)
        self.assertEqual(tree["sections"][1]["pages"][0]["summary"], "Shows Figure 2 and Table 1.")

    def test_failed_page_is_not_cached(self):
        def boom(content):
            raise RuntimeError("down")
        with tempfile.TemporaryDirectory() as tmp:
            cache = PageSummaryCache(Path(tmp) / "c.json")
            s = Summarizer("m", FakePool(boom), cache=cache)
            self.assertEqual(s.page_summary("d", make_tree()["sections"][0]["pages"][0], Path(tmp)),
                             {"summary": "", "entities": ""})
            self.assertEqual(len(cache), 0)

    def test_cache_only_serves_same_config(self):
        page = make_tree()["sections"][0]["pages"][0]
        with tempfile.TemporaryDirectory() as tmp:
            cache = PageSummaryCache(Path(tmp) / "c.json")
            a = Summarizer("model-a", FakePool(page_answer), cache=cache)
            a.page_summary("d", page, Path(tmp))
            cache.save()

            cache = PageSummaryCache(Path(tmp) / "c.json")
            self.assertEqual(cache.count(a.page_fp), (1, 0))
            a = Summarizer("model-a", FakePool(page_answer), cache=cache)
            a.page_summary("d", page, Path(tmp))
            self.assertEqual(a.stats["page_cached"], 1)

            b_pool = FakePool(page_answer)
            b = Summarizer("model-b", b_pool, cache=cache)
            self.assertNotEqual(a.page_fp, b.page_fp)
            self.assertEqual(cache.count(b.page_fp), (0, 1))
            b.page_summary("d", page, Path(tmp))
            self.assertEqual((b.stats["page_cached"], b.stats["page_calls"], len(b_pool.calls)), (0, 1, 1))

            cache.set("d", 9, {"summary": "legacy", "entities": ""})  # older entries have no config
            self.assertIsNone(cache.get("d", 9, b.page_fp))

    def test_truncation_and_raw_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "raw.jsonl"
            cache = PageSummaryCache(Path(tmp) / "c.json")
            s = Summarizer("m", FakePool(page_answer, finish_reason="length"), cache=cache, raw_log=log)
            tree = make_tree()
            s.summarize_pages(tree, Path(tmp))
            s.summarize_sections(tree)
            records = [json.loads(line) for line in log.read_text().splitlines()]

            again = Summarizer("m", FakePool(page_answer), cache=cache)
            again.summarize_pages(make_tree(), Path(tmp))

        self.assertEqual((s.stats["page_calls"], s.stats["page_truncated"]), (2, 2))
        self.assertEqual(s.stats["section_truncated"], 2)
        self.assertEqual((again.stats["page_cached"], again.stats["page_truncated"]), (2, 2))
        kinds = [r["kind"] for r in records]
        self.assertEqual(kinds.count("page"), 2)
        self.assertEqual(kinds.count("section"), 2)
        page = next(r for r in records if r["kind"] == "page")
        self.assertEqual(page["finish_reason"], "length")
        self.assertEqual(page["tokens"], {"prompt": 100, "completion": 20})
        self.assertEqual(page["config"], s.page_fp)
        self.assertFalse(page["image"])
        self.assertIn("<summary>", page["response"])
        section = next(r for r in records if r["kind"] == "section")
        self.assertEqual((section["doc_id"], section["config"]), ("d", s.section_fp))

    def test_empty_response_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = PageSummaryCache(Path(tmp) / "c.json")
            s = Summarizer("m", FakePool(lambda c: ""), cache=cache)
            tree = make_tree()
            s.summarize_pages(tree, Path(tmp))
            s.summarize_sections(tree)
        self.assertEqual(len(cache), 0)
        self.assertEqual(s.stats["page_errors"], 2)
        self.assertEqual(tree["sections"][0]["summary"], "")

    def test_sampling_parameters(self):
        pool = FakePool(page_answer)
        s = Summarizer("m", pool, max_tokens=4096, top_k=20, presence_penalty=1.5)
        s.section_card(make_tree()["sections"][0], "d")
        self.assertEqual(pool.sampling, {"max_tokens": 4096, "temperature": 0.1, "presence_penalty": 1.5,
                                         "extra_body": {"top_k": 20}})
        self.assertEqual((s.page_config["top_k"], s.page_config["top_p"]), (20, None))
        self.assertNotEqual(s.page_fp, Summarizer("m", pool, max_tokens=4096).page_fp)

    def test_config_conflict(self):
        a = {"model": "x", "prompt_sha256": "1", "max_tokens": 1024}
        self.assertIsNone(config_conflict(None, a))
        self.assertIsNone(config_conflict(dict(a), a))
        self.assertIsNone(config_conflict(a, {**a, "top_k": None}))  # unset == absent
        self.assertEqual(config_conflict(a, {**a, "model": "y"}), "model: 'x' -> 'y'")

    def test_refresh_visual_refs(self):
        tree = make_tree()
        tree["sections"][0]["summary"] = "CARD\n\nVISUAL_REFS_FROM_PAGES:\n- Figure 9 (Page 1)"
        tree["sections"][0]["pages"][1]["summary"] = "See Table 4"
        self.assertTrue(refresh_visual_refs(tree))
        self.assertEqual(tree["sections"][0]["summary"], "CARD\n\nVISUAL_REFS_FROM_PAGES:\n- Table 4 (Page 2)")
        self.assertEqual(tree["root_summary"], tree["sections"][0]["summary"])


if __name__ == "__main__":
    unittest.main()
