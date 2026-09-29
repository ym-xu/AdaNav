"""Summary generation with a scripted fake model (no network).

Run: python -m unittest discover tests
"""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adanav.doctree.summarize import (
    PageSummaryCache, Summarizer, extract_visual_refs, parse_page_response,
    parse_section_response, refresh_visual_refs,
)


class FakePool:
    """Stands in for EndpointPool: records prompts, answers from a callback."""

    def __init__(self, answer):
        self.answer, self.calls = answer, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def next(self):
        return self

    def _create(self, model, messages, max_tokens, temperature):
        content = messages[0]["content"]
        self.calls.append(content)
        text = self.answer(content)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


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
        # Page 1 appears in both sections but is summarised once (cache), page 0 once: 2 + 2 section calls.
        self.assertLessEqual(len(pool.calls), 5)

    def test_failed_page_is_not_cached(self):
        def boom(content):
            raise RuntimeError("down")
        with tempfile.TemporaryDirectory() as tmp:
            cache = PageSummaryCache(Path(tmp) / "c.json")
            s = Summarizer("m", FakePool(boom), cache=cache)
            self.assertEqual(s.page_summary("d", make_tree()["sections"][0]["pages"][0], Path(tmp)),
                             {"summary": "", "entities": ""})
            self.assertEqual(len(cache), 0)

    def test_refresh_visual_refs(self):
        tree = make_tree()
        tree["sections"][0]["summary"] = "CARD\n\nVISUAL_REFS_FROM_PAGES:\n- Figure 9 (Page 1)"
        tree["sections"][0]["pages"][1]["summary"] = "See Table 4"
        self.assertTrue(refresh_visual_refs(tree))
        self.assertEqual(tree["sections"][0]["summary"], "CARD\n\nVISUAL_REFS_FROM_PAGES:\n- Table 4 (Page 2)")
        self.assertEqual(tree["root_summary"], tree["sections"][0]["summary"])


if __name__ == "__main__":
    unittest.main()
