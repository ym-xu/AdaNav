"""Agents, pipeline and metrics with a scripted fake model (no network).

Run: python -m unittest discover tests
"""

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adanav.agents import AdaNav, AdaNavConfig, ChatModel, SectionSelector, SpecificAgent, Trace
from adanav.agents.navigators import GeneralAgent, PageNavigator, VisionPageNavigator, parse_page_numbers
from adanav.agents.reasoner import NOT_ANSWERABLE, parse_reasoner_response
from adanav.agents.router import GENERAL, NORMAL, SPECIFIC, Router
from adanav.agents.specific import parse_references
from adanav.eval.benchmark import load_samples
from adanav.eval.judge import normalize, parse_score
from adanav.eval.metrics import page_scores, qa_metrics, retrieval_metrics


def kind_of(content):
    if isinstance(content, list):
        return "vision_nav" if "page navigator with vision" in content[0]["text"] else "reasoner"
    for marker, kind in (("RECALL phase", "chunk"), ("FINAL SELECTION", "rerank"), ("Page Information:", "page_nav"),
                         ("pre-filtering pages", "prefilter"),
                         ("precision filter", "refilter"), ("Section Summaries:", "section_filter")):
        if marker in content:
            return kind
    raise AssertionError(f"unknown prompt: {content[:80]}")


class FakePool:
    """Stands in for EndpointPool; ``replies`` maps prompt kind -> reply text or callable(content)."""

    def __init__(self, replies):
        self.replies, self.calls, self.lock = replies, [], threading.Lock()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def next(self):
        return self

    def kinds(self):
        return [k for k, _ in self.calls]

    def _create(self, model, messages, **sampling):
        content = messages[0]["content"]
        kind = kind_of(content)
        with self.lock:
            self.calls.append((kind, content))
        reply = self.replies[kind]
        text = reply(content) if callable(reply) else reply
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
                               choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")])


def page(i, elements=(), entities=""):
    return {"page_idx": i, "image_path": f"images/page_{i + 1}.png", "ocr_text": f"text of page {i + 1}",
            "summary": f"summary {i + 1}", "entities": entities, "elements": list(elements)}


def make_tree():
    """Three sections over pages 0-6; page 2 is a boundary page shared by sec_00 and sec_01."""
    pn = lambda text: {"type": "page_number", "text": text}
    pages = [page(0), page(1, [pn("1")]), page(2, [pn("-2-")]),
             page(3, [pn("3"), {"type": "image", "caption": "['Figure 3: Revenue by year']", "text": ""}]),
             page(4, [pn("4")], entities="NUMBERS: 5\nREFERENCES: Table 2, Figure 3"),
             page(5, [pn("5")]), page(6)]
    return {"doc_id": "doc", "num_pages": 7, "root_summary": "cards", "sections": [
        {"id": "sec_00", "title": "Intro", "start_page": 0, "end_page": 2, "summary": "card A", "pages": pages[0:3]},
        {"id": "sec_01", "title": "Results", "start_page": 2, "end_page": 5, "summary": "card B", "pages": pages[2:6]},
        {"id": "sec_02", "title": "End", "start_page": 6, "end_page": 6, "summary": "card C", "pages": pages[6:7]},
    ]}


def chat(pool):
    return ChatModel("m", pool, retries=0)


class SpecificTest(unittest.TestCase):
    def test_parse_references(self):
        self.assertEqual(parse_references("What is on page 5 and in Figure 3?"), [("page", "5"), ("figure", "3")])
        self.assertEqual(parse_references("Compare pages 3, 5 and 10"),
                         [("page", "3"), ("page", "5"), ("page", "10")])
        self.assertEqual(parse_references("Table 2 vs tab. 4, fig 1"),
                         [("figure", "1"), ("table", "2"), ("table", "4")])
        self.assertEqual(parse_references("the top 5 companies"), [])

    def test_lookup(self):
        tree, agent = make_tree(), SpecificAgent()
        r = agent.retrieve("What does page 2 say?", tree)  # printed "-2-" is on page_idx 2
        self.assertEqual((r.pages, r.sources), ([2], ["page_number"]))
        # an unnumbered page is indexed under its physical number, which can collide with a printed one
        self.assertEqual(agent.retrieve("page 1", tree).pages, [1, 0])
        self.assertEqual(agent.retrieve("page 7", tree).pages, [6])  # no printed number: physical page
        r = agent.retrieve("In Figure 3, what grows?", tree)
        self.assertEqual((r.pages, r.sources), ([3], ["element"]))  # caption before entity index
        r = agent.retrieve("Table 2 and Table 9?", tree)
        self.assertEqual((r.pages, r.matched, r.not_found), ([4], ["table 2"], ["table 9"]))

    def test_parse_slides_ranges_roman_and_format_examples(self):
        self.assertEqual(parse_references("What is on slide 4?"), [("page", "4")])
        self.assertEqual(parse_references("Which stages are shown on slides 12 and 14?"),
                         [("page", "12"), ("page", "14")])
        self.assertEqual(parse_references("How many tables are in Pages 5-8?"),
                         [("page", "5"), ("page", "6"), ("page", "7"), ("page", "8")])
        self.assertEqual(parse_references("figures from Page 3 to Page 5"), [("page", "3"), ("page", "4"), ("page", "5")])
        self.assertEqual(parse_references("charts in the first 3 pages"), [("page", "1"), ("page", "2"), ("page", "3")])
        self.assertEqual(parse_references("figures in Pages 400-640"), [])   # too long: left to the router
        self.assertEqual(parse_references("According to table II and Figure iv"), [("figure", "4"), ("table", "2")])
        self.assertEqual(parse_references("Which pages show the logo? Format it like ['Page 2', 'Page 4']."), [])
        self.assertEqual(parse_references('List the pages, for example ["p1","p2"]'), [])

    def test_page_labels(self):
        tree, agent = make_tree(), SpecificAgent()
        labels = {0: "cover", 1: "i", 2: "1", 3: "2", 4: "3-4", 5: "0005"}   # page 6 unlabelled
        for section in tree["sections"]:
            for p in section["pages"]:
                p["page_label"] = labels.get(p["page_idx"])
        r = agent.retrieve("What does page 2 say?", tree)   # printed label wins over physical page 2
        self.assertEqual((r.pages, r.sources), ([3], ["page_label"]))
        self.assertEqual(agent.retrieve("page 4", tree).pages, [4])     # spread "3-4"
        self.assertEqual(agent.retrieve("page 005", tree).pages, [5])   # leading zeros
        r = agent.retrieve("page 7", tree)                              # no such label: physical page
        self.assertEqual((r.pages, r.sources), ([6], ["physical"]))
        self.assertEqual(agent.retrieve("page 30", tree).not_found, ["page 30"])

    def test_caption_in_text_element_and_roman_caption(self):
        tree, agent = make_tree(), SpecificAgent()
        tree["sections"][2]["pages"][0]["elements"] = [
            {"type": "text", "text": "Figure 7: An example of a generated program."},
            {"type": "text", "text": "As Figure 3 shows, revenue grows."},          # a mention, not a caption
            {"type": "table", "caption": ["TABLE IV. Results"], "text": ""}]
        r = agent.retrieve("In Figure 7, what is generated?", tree)
        self.assertEqual((r.pages, r.sources), ([6], ["element"]))
        self.assertEqual(agent.retrieve("What is in Figure 3?", tree).pages, [3])
        self.assertEqual(agent.retrieve("According to Table 4", tree).pages, [6])


class RouterTest(unittest.TestCase):
    def test_specific_needs_no_model(self):
        pool = FakePool({})
        route = Router(SectionSelector(chat(pool))).route("What is in Figure 3?", make_tree(), Trace())
        self.assertEqual((route.category, route.specific.pages, pool.calls), (SPECIFIC, [3], []))

    def test_unresolved_reference_goes_to_llm(self):
        pool = FakePool({"section_filter": "<selected_sections>sec_01</selected_sections>"})
        route = Router(SectionSelector(chat(pool))).route("What is in Figure 8?", make_tree(), Trace())
        self.assertEqual((route.category, route.sections), (NORMAL, ["sec_01"]))

    def test_global_and_normal(self):
        tree = make_tree()
        pool = FakePool({"section_filter": "<selected_sections>\nALL\n</selected_sections>"})
        route = Router(SectionSelector(chat(pool))).route("How many charts?", tree, Trace())
        self.assertEqual((route.category, route.sections), (GENERAL, ["sec_00", "sec_01", "sec_02"]))
        pool = FakePool({"section_filter": "<query_analysis>sec_02?</query_analysis>"
                                           "<selected_sections>sec_01, sec_09, sec_00, sec_02</selected_sections>"})
        route = Router(SectionSelector(chat(pool), top_k=2)).route("Revenue?", tree, Trace())
        self.assertEqual((route.category, route.sections), (NORMAL, ["sec_01", "sec_00"]))  # sec_09 unknown

    def test_unparseable_falls_back_and_says_so(self):
        pool = FakePool({"section_filter": "I cannot decide."})
        trace = Trace()
        choice = SectionSelector(chat(pool), top_k=2).select("q", make_tree(), trace)
        self.assertEqual(choice.sections, ["sec_00", "sec_01"])
        self.assertEqual(trace.summary()["section_filter_fallbacks"], 1)

    def test_chunked_global_from_any_chunk(self):
        # one section per chunk; the first chunk says ALL, later chunks name sections
        replies = iter(["<selected_sections>ALL</selected_sections>",
                        "<selected_sections>sec_01</selected_sections>",
                        "<selected_sections>sec_02</selected_sections>"])
        pool = FakePool({"chunk": lambda c: next(replies)})
        choice = SectionSelector(chat(pool), max_chunk_sections=1).select("How many tables?", make_tree(), Trace())
        self.assertTrue(choice.global_flag)
        self.assertEqual(pool.kinds(), ["chunk"] * 3)

    def test_chunked_rerank(self):
        pool = FakePool({"chunk": lambda c: "<selected_sections>" + c.split("=== [")[1][:6] + "</selected_sections>",
                         "rerank": "<selected_sections>sec_02, sec_00</selected_sections>"})
        choice = SectionSelector(chat(pool), top_k=2, max_chunk_sections=1).select("q", make_tree(), Trace())
        self.assertEqual((choice.sections, choice.global_flag), (["sec_02", "sec_00"], False))
        self.assertEqual(pool.kinds(), ["chunk", "chunk", "chunk", "rerank"])


class NavigatorTest(unittest.TestCase):
    def test_parse_page_numbers(self):
        self.assertEqual(parse_page_numbers("<selected_pages>4, 3, 4</selected_pages>", [2, 3, 4]), ([3, 2], True))
        self.assertEqual(parse_page_numbers("<selected_pages>2</selected_pages>", [2, 3]), ([2], True))  # 0-indexed
        self.assertEqual(parse_page_numbers("pages 3", [2]), ([2], False))

    def test_select_and_navigate(self):
        tree = make_tree()
        pool = FakePool({"page_nav": "<selected_pages>6, 3, 4, 5</selected_pages>"})
        trace = Trace()
        pages = PageNavigator(chat(pool)).navigate("q", tree, ["sec_01", "sec_02"], 3, trace)
        self.assertEqual(pages, [5, 2, 3, 6])  # sec_01 budget 3, then the single-page sec_02 (no call)
        self.assertEqual(pool.kinds(), ["page_nav"])
        self.assertIn("OCR Text: text of page 3", pool.calls[0][1])

    def test_general_refilter(self):
        tree = make_tree()
        nav = FakePool({"page_nav": lambda c: "<selected_pages>1, 2, 3, 4, 5, 6</selected_pages>"})
        for qtype, expected in (("LOCATING", [0, 1]), ("COUNTING", [0, 1, 2, 3, 4, 6])):
            pool = FakePool({**nav.replies, "refilter": f"<query_type>{qtype}</query_type>"
                                                        "<selected_pages>1, 2, 3, 4, 5, 6</selected_pages>"})
            agent = GeneralAgent(PageNavigator(chat(pool)), chat(pool), budget=3, refilter_max=2,
                                 refilter_max_counting=6)
            self.assertEqual(agent.retrieve("q", tree, Trace()), expected)
            self.assertEqual(pool.kinds().count("refilter"), 1)

    def test_general_refilter_chunks_when_over_budget(self):
        tree = make_tree()
        pool = FakePool({"page_nav": "<selected_pages>1, 2, 3, 4, 5, 6</selected_pages>",
                         "refilter": lambda c: "<selected_pages>" + c.split("Page ")[1].split(":")[0] + "</selected_pages>"})
        agent = GeneralAgent(PageNavigator(chat(pool)), chat(pool), budget=3, refilter_max=2, prompt_budget=30)
        trace = Trace()
        pages = agent.retrieve("q", tree, trace)
        self.assertLessEqual(len(pages), 2)
        self.assertIn("refilter_chunk", [e["step"] for e in trace.calls()])


class ReasonerTest(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_reasoner_response("<scratchpad>x</scratchpad><answer> 42 </answer>").answer, "42")
        self.assertEqual(parse_reasoner_response("<scratchpad>x</scratchpad><not_answerable>Not answerable"
                                                 "</not_answerable>").answer, NOT_ANSWERABLE)
        out = parse_reasoner_response("<query_update>find Table 4</query_update><notes>saw T3</notes>")
        self.assertEqual((out.answer, out.query_update, out.notes), ("", "find Table 4", "saw T3"))
        self.assertEqual(parse_reasoner_response("<scratchpad>long</scratchpad><answer>Paris").answer, "Paris")
        self.assertEqual(parse_reasoner_response("Paris").answer, "Paris")


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.image_root = Path(self.tmp.name)
        (self.image_root / "images").mkdir()
        for i in range(1, 8):
            (self.image_root / "images" / f"page_{i}.png").write_bytes(b"png")

    def tearDown(self):
        self.tmp.cleanup()

    def test_normal_question(self):
        pool = FakePool({"section_filter": "<selected_sections>sec_01</selected_sections>",
                         "page_nav": "<selected_pages>5, 4</selected_pages>",
                         "reasoner": "<scratchpad>s</scratchpad><answer>12%</answer>"})
        config = AdaNavConfig.preset("agentic", page_navigator="text")
        record = AdaNav("m", pool, config).run("Revenue growth?", make_tree(), self.image_root)
        self.assertEqual((record["category"], record["handler"], record["pages"], record["answer"]),
                         (NORMAL, "treenav", [4, 3], "12%"))
        self.assertEqual(pool.kinds(), ["section_filter", "page_nav", "reasoner"])
        reasoner_content = pool.calls[-1][1]
        self.assertIn("text of page 5", reasoner_content[0]["text"])
        # retrieval order is [4, 3]; the reasoner sees the pages in document order
        self.assertEqual([c["text"] for c in reasoner_content[1:] if c["type"] == "text"], ["[Page 4]", "[Page 5]"])
        self.assertEqual(record["usage"]["reasoner_calls"], 1)
        self.assertEqual([e["response"] for e in record["trace"]][-1], "<scratchpad>s</scratchpad><answer>12%</answer>")

    def test_specific_question_and_retrieval_only(self):
        pool = FakePool({})
        record = AdaNav("m", pool).run("What is in Figure 3?", make_tree(), self.image_root, retrieval_only=True)
        self.assertEqual((record["category"], record["pages"], pool.calls), (SPECIFIC, [3], []))
        self.assertNotIn("answer", record)

    def test_base_preset_has_no_rule_stage(self):
        pool = FakePool({"section_filter": "<selected_sections>sec_02</selected_sections>",
                         "reasoner": "<answer>x</answer>"})
        AdaNav("m", pool, AdaNavConfig.preset("base")).run("What is in Figure 3?", make_tree(), self.image_root)
        self.assertEqual(pool.kinds(), ["section_filter", "reasoner"])
        self.assertIn("Document level summary", pool.calls[-1][1][0]["text"])  # SimpleDoc reasoner prompt

    def test_reflection_rounds(self):
        replies = iter(["<query_update>Figure 3 revenue</query_update><notes>need fig</notes>",
                        "<answer>up</answer>"])
        pool = FakePool({"section_filter": "<selected_sections>sec_02</selected_sections>",
                         "reasoner": lambda c: next(replies)})
        config = AdaNavConfig.preset("agentic", iterations=3)
        self.assertEqual(config.reasoner_prompt, "global_reasoner_reflect_prompt")
        record = AdaNav("m", pool, config).run("How did revenue change?", make_tree(), self.image_root)
        self.assertEqual(record["answer"], "up")
        self.assertEqual([r["pages"] for r in record["rounds"]], [[6], [3, 6]])  # update routed to SPECIFIC
        self.assertIn("need fig", pool.calls[-1][1][0]["text"])

    def test_visual_notes_reach_navigator_and_reasoner(self):
        tree = make_tree()
        for section in tree["sections"]:
            for p in section["pages"]:
                if p["page_idx"] == 4:
                    p["elements"].append({"type": "image", "caption": ["Figure 9: Sales"], "text": "",
                                          "description": "A bar chart of sales by region."})
        pool = FakePool({"section_filter": "<selected_sections>sec_01</selected_sections>",
                         "page_nav": "<selected_pages>5</selected_pages>", "reasoner": "<answer>x</answer>"})
        config = AdaNavConfig.preset("agentic", page_navigator="text")
        AdaNav("m", pool, config).run("Which region sells most?", tree, self.image_root)
        nav_prompt = next(c for k, c in pool.calls if k == "page_nav")
        self.assertIn("Figures:\n    - Figure 9: Sales: A bar chart of sales by region.", nav_prompt)
        self.assertIn("[Figure descriptions]\n- Figure 9: Sales: A bar chart", pool.calls[-1][1][0]["text"])

    def test_retry_untagged_reasoner_response(self):
        replies = iter(["I think the answer is somewhere", "<answer>42</answer>"])
        pool = FakePool({"section_filter": "<selected_sections>sec_02</selected_sections>",
                         "reasoner": lambda c: next(replies)})
        record = AdaNav("m", pool, AdaNavConfig.preset("agentic")).run("q", make_tree(), self.image_root)
        self.assertEqual(record["answer"], "42")
        self.assertIn("reasoner_retry", [e["step"] for e in record["trace"]])

    def test_page_images_replace_reasoner_images(self):
        doc_dir = self.image_root / "hires" / "doc"
        doc_dir.mkdir(parents=True)
        (doc_dir / "page_7.jpg").write_bytes(b"jpg")
        pool = FakePool({"section_filter": "<selected_sections>sec_02</selected_sections>",
                         "reasoner": "<answer>x</answer>"})
        config = AdaNavConfig.preset("agentic", page_images=str(self.image_root / "hires"))
        AdaNav("m", pool, config).run("q", make_tree(), self.image_root)
        urls = [part["image_url"]["url"] for part in pool.calls[-1][1] if part["type"] == "image_url"]
        self.assertEqual(len(urls), 1)
        self.assertTrue(urls[0].startswith("data:image/jpeg;base64,"))

    def test_single_round_query_update_is_not_answerable(self):
        pool = FakePool({"section_filter": "<selected_sections>sec_02</selected_sections>",
                         "reasoner": "<query_update>more</query_update>"})
        record = AdaNav("m", pool).run("q", make_tree(), self.image_root)
        self.assertEqual(record["answer"], NOT_ANSWERABLE)


class EvalTest(unittest.TestCase):
    def test_page_scores(self):
        self.assertIsNone(page_scores([1], []))
        s = page_scores([0, 1, 5], [1, 2, 3])  # predicted 1-indexed {1, 2, 6}
        self.assertAlmostEqual(s["recall"], 2 / 3)
        self.assertAlmostEqual(s["precision"], 2 / 3)
        self.assertEqual(s["hit"], 0.0)
        self.assertEqual(page_scores([], [4])["f1"], 0.0)

    def test_metrics(self):
        samples = [{"qid": "0", "evidence_pages": [1], "evidence_sources": ["TAB"]},
                   {"qid": "1", "evidence_pages": [2, 3], "evidence_sources": ["TXT", "CHA"]},
                   {"qid": "2", "evidence_pages": [], "evidence_sources": []}]
        preds = {"0": {"pages": [0], "category": "specific"}, "2": {"pages": [5], "category": "normal"}}
        r = retrieval_metrics(samples, preds)
        self.assertEqual(r["overall"]["n"], 2)
        self.assertAlmostEqual(r["overall"]["recall"], 0.5)  # qid 1 failed: counts as nothing retrieved
        qa = qa_metrics(samples, preds, {"0": 1, "2": 1})
        self.assertAlmostEqual(qa["overall"]["acc"], 2 / 3)
        self.assertEqual(qa["by_span"]["UNA"]["acc"], 1.0)
        self.assertEqual(qa["by_category"]["specific"], {"n": 1, "acc": 1.0})

    def test_judge_helpers(self):
        self.assertEqual(normalize("41%"), normalize("41"))
        self.assertEqual(normalize("1,234."), "1234")
        self.assertEqual(parse_score('{"binary_correctness": 1}'), 1)
        self.assertEqual(parse_score('```json\n{"binary_correctness": 0}\n```'), 0)

    def test_load_samples(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump([{"doc_id": "a.pdf", "question": "q", "answer": 3, "evidence_pages": "[5, 6]",
                        "evidence_sources": "['Chart', 'Generalized-text (Layout)']", "answer_format": "Int"},
                       {"doc_id": "b.pdf", "question": "q2", "answer": "Not answerable", "evidence_pages": "[]",
                        "evidence_sources": "[]"}], f)
        s = load_samples(f.name)
        Path(f.name).unlink()
        self.assertEqual((s[0]["qid"], s[0]["doc_id"], s[0]["answer"]), ("0000", "a", "3"))
        self.assertEqual((s[0]["evidence_pages"], s[0]["evidence_sources"]), ([5, 6], ["CHA", "LAY"]))
        self.assertEqual(s[1]["evidence_pages"], [])


class VisionNavigatorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "images").mkdir()
        for i in range(1, 8):
            (self.root / "images" / f"page_{i}.png").write_bytes(b"png")

    def tearDown(self):
        self.tmp.cleanup()

    def test_interleaved_samples_merged_in_order(self):
        replies = iter(["<selected_pages>4</selected_pages>", "<selected_pages>3, 4</selected_pages>",
                        "<selected_pages>6</selected_pages>"])
        lock = threading.Lock()

        def reply(content):
            with lock:
                return next(replies)

        pool = FakePool({"vision_nav": reply})
        trace = Trace()
        nav = VisionPageNavigator(chat(pool), samples=3, workers=1)
        pages = nav.select("Revenue?", make_tree()["sections"][1], 2, trace, self.root)
        self.assertEqual(len(pool.calls), 3)
        content = pool.calls[0][1]
        kinds = [c["type"] for c in content]
        self.assertEqual(kinds, ["text"] + ["text", "image_url"] * 4 + ["text"])  # header, (info, image) x4, footer
        self.assertIn("--- Page 3 ---", content[1]["text"])
        self.assertEqual(len(pages), 2)
        self.assertTrue(set(pages) <= {2, 3, 5})
        self.assertEqual([e.get("sample") for e in trace.calls("vision_nav")].count(None), 0)

    def test_chunks_overlap_and_prefilter(self):
        nav = VisionPageNavigator(chat(FakePool({})), prompt_budget=3 * (VisionPageNavigator.IMAGE_TOKENS + 30),
                                  overlap=1)
        pages = make_tree()["sections"][0]["pages"] + make_tree()["sections"][1]["pages"][1:]  # pages 0-5
        chunks = [[p["page_idx"] for p in c] for c in nav._chunks(pages)]
        self.assertEqual(chunks, [[0, 1, 2], [2, 3, 4], [4, 5]])
        pool = FakePool({"prefilter": "<selected_pages>2, 4</selected_pages>",
                         "vision_nav": "<selected_pages>4</selected_pages>"})
        nav = VisionPageNavigator(chat(pool), samples=1, prefilter_pages=3)
        self.assertEqual(nav.select("q", make_tree()["sections"][1], 3, Trace(), self.root), [3])
        self.assertEqual(pool.kinds(), ["prefilter", "vision_nav"])
        shown = " ".join(c["text"] for c in pool.calls[1][1] if c["type"] == "text")
        self.assertIn("--- Page 3 ---", shown)  # prefilter kept page_idx 2 and 3 ("2" read 0-indexed)
        self.assertNotIn("--- Page 5 ---", shown)


class EndpointPoolTest(unittest.TestCase):
    def test_affinity_with_bounded_load(self):
        from contextlib import ExitStack
        from adanav.llm import EndpointPool
        pool = EndpointPool([f"http://127.0.0.1:{9000 + i}/v1" for i in range(4)], api_key="x")
        with pool.lease("doc-a") as first:
            pass
        with pool.lease("doc-a") as again:
            self.assertIs(again, first)  # idle pool: same endpoint for the same key
        with ExitStack() as stack:
            held = [stack.enter_context(pool.lease("doc-a")) for _ in range(6)]
            self.assertIs(held[0], first)
            self.assertGreater(len({id(c) for c in held}), 1)  # overflow went to other endpoints
        self.assertEqual(pool._inflight, [0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
