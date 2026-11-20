"""Page labels and manual tree corrections (adanav.doctree.fixes).

Run: python -m unittest discover tests
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adanav.doctree.fixes import apply_fixes, expand_page_labels, set_page_labels


def tree():
    pages = [{"page_idx": i, "image_path": f"images/page_{i + 1}.png", "elements": []} for i in range(4)]
    return {"doc_id": "doc", "num_pages": 4, "sections": [
        {"id": "sec_00", "pages": [dict(p) for p in pages[0:3]]},
        {"id": "sec_01", "pages": [dict(p) for p in pages[2:4]]},   # page 2 is shared
    ]}


def labels_of(t):
    return [(p["page_idx"], p["page_label"]) for s in t["sections"] for p in s["pages"]]


class FixesTest(unittest.TestCase):
    def test_expand(self):
        self.assertEqual(expand_page_labels({"1": "cover", "2-4": "1"}), {0: "cover", 1: "1", 2: "2", 3: "3"})
        self.assertEqual(expand_page_labels({"2-3": None}), {1: None, 2: None})
        with self.assertRaises(ValueError):
            expand_page_labels({"2-3": "iv"})

    def test_set_labels_reaches_both_copies_of_a_boundary_page(self):
        t = tree()
        self.assertEqual(set_page_labels(t, {0: "i", 2: "1"}), 5)
        self.assertEqual(labels_of(t), [(0, "i"), (1, None), (2, "1"), (2, "1"), (3, None)])
        self.assertEqual(list(t["sections"][0]["pages"][0])[:2], ["page_idx", "page_label"])
        self.assertEqual(set_page_labels(t, {0: "i", 2: "1"}), 0)   # idempotent

    def test_apply_fixes_only_touches_listed_pages(self):
        t = tree()
        set_page_labels(t, {0: "1", 1: "2", 2: "3", 3: "4"})
        applied = apply_fixes(t, {"doc": {"note": "x", "page_labels": {"1": "cover", "2-4": "1"}}, "other": {}})
        self.assertEqual(applied, ["page_labels: 4 pages"])
        self.assertEqual(labels_of(t), [(0, "cover"), (1, "1"), (2, "2"), (2, "2"), (3, "3")])
        self.assertEqual(apply_fixes(t, {"other": {"page_labels": {"1": "x"}}}), [])


if __name__ == "__main__":
    unittest.main()
