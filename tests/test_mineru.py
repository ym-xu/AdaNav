"""Tree construction from a synthetic MinerU document.

Run: python -m unittest discover tests
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adanav.doctree import build_tree, validate_tree
from adanav.doctree.mineru import MinerUDocument, SplitConfig, html_table_to_text


def make_doc(root: Path, num_pages: int, headings, logical_pages=None, items=(), doc_id="doc"):
    d = root / doc_id
    (d / "images").mkdir(parents=True)
    (d / "process").mkdir()
    for i in range(1, num_pages + 1):
        (d / "images" / f"page_{i}.png").write_bytes(b"")
    (d / "process" / "toc.json").write_text(json.dumps({"headings": headings}))
    if logical_pages is not None:
        (d / "process" / "logical_pages.json").write_text(json.dumps({"pages": logical_pages}))
    (d / f"{doc_id}_content_list.json").write_text(json.dumps(list(items)))
    return d


def h(title, page, children=()):
    node = {"title": title, "level": 1, "logical_page": page}
    if children:
        node["children"] = [{"title": t, "level": 2, "logical_page": p} for t, p in children]
    return node


class BuildTreeTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def ranges(self, tree):
        return [(s["title"], s["start_page"], s["end_page"]) for s in tree["sections"]]

    def test_sections_share_boundary_pages_and_get_a_preface(self):
        d = make_doc(self.root, 10, [h("A", "3"), h("B", "6")])
        tree = build_tree(d)
        self.assertEqual(self.ranges(tree), [("Preface", 0, 2), ("A", 2, 5), ("B", 5, 9)])
        self.assertEqual([s["id"] for s in tree["sections"]], ["sec_00", "sec_01", "sec_02"])
        self.assertEqual(validate_tree(tree), [])

    def test_logical_pages_offset_and_two_up_spreads(self):
        lp = [{"logical_page": "cover", "physical_page": 1},
              {"logical_page": "2-3", "physical_page": 2},
              {"logical_page": "4-5", "physical_page": 3}]
        d = make_doc(self.root, 3, [h("A", "cover"), h("B", "3"), h("C", "4")], logical_pages=lp)
        doc = MinerUDocument(d)
        self.assertEqual([doc.resolve_page(x) for x in ("cover", "2", "3", "2-3", "5", "9")],
                         [0, 1, 1, 1, 2, 8])  # unmapped "9" falls back to 9 - 1
        self.assertEqual(self.ranges(build_tree(d)), [("A", 0, 1), ("B", 1, 2), ("C", 2, 2)])

    def test_long_section_splits_on_subheadings(self):
        d = make_doc(self.root, 40, [h("A", "1", children=[("A.1", "5"), ("A.2", "20")])])
        tree = build_tree(d, SplitConfig(max_section_pages=30))
        self.assertEqual(self.ranges(tree), [
            ("A - Introduction", 0, 4), ("A - A.1", 4, 19), ("A - A.2", 19, 39)])

    def test_long_section_without_subheadings_splits_into_shared_chunks(self):
        d = make_doc(self.root, 60, [h("A", "1")])
        tree = build_tree(d, SplitConfig(max_section_pages=30, split_chunk_pages=25))
        # Fixed-size chunks are made disjoint, then share their boundary page.
        self.assertEqual(self.ranges(tree), [
            ("A (Part 1)", 0, 25), ("A (Part 2)", 25, 50), ("A (Part 3)", 50, 59)])
        self.assertEqual(validate_tree(tree), [])

    def test_elements_tables_and_ocr_text(self):
        items = [
            {"type": "text", "text": "Hello", "page_idx": 0, "bbox": [0, 0, 1, 1]},
            {"type": "table", "page_idx": 0, "img_path": "images/t.jpg", "bbox": [0, 0, 1, 1],
             "table_caption": ["Table 1: Results"], "table_footnote": [],
             "table_body": "<table><tr><td>a</td><td>1</td></tr></table>"},
            {"type": "image", "page_idx": 1, "img_path": "images/f.jpg", "bbox": [0, 0, 1, 1],
             "image_caption": [], "image_footnote": []},
        ]
        d = make_doc(self.root, 2, [h("A", "1")], items=items)
        tree = build_tree(d)
        p0, p1 = tree["sections"][0]["pages"]
        table = p0["elements"][1]
        self.assertEqual(table["text"], "Table: Table 1: Results\n| a | 1")
        self.assertEqual(table["caption"], ["Table 1: Results"])
        self.assertEqual(table["id"], "elem_0_1")
        self.assertEqual(p0["ocr_text"], "Hello\n\nTable: Table 1: Results\n| a | 1")
        self.assertEqual(p0["image_path"], "images/page_1.png")
        self.assertEqual(p1["elements"][0]["text"], "")  # captionless image: no text

    def test_missing_toc_gives_one_section(self):
        d = make_doc(self.root, 5, [])
        (d / "process" / "toc.json").unlink()
        self.assertEqual(self.ranges(build_tree(d)), [("Document Content", 0, 4)])

    def test_html_table_to_text(self):
        html = "<table><thead><tr><th>k</th><th>v</th></tr></thead><tr><td>x</td><td></td></tr></table>"
        self.assertEqual(html_table_to_text(html), "| k | v\n| x |")


if __name__ == "__main__":
    unittest.main()
