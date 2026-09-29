"""The tree_index.json schema: Document → Section → Page → Element.

Field names follow the evaluated trees exactly; see data/README.md for the
annotated layout and the mapping onto the paper's notation.
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict, Union


class Element(TypedDict):
    id: str                       # MinerU node_id, else "elem_<page>_<k>"
    type: str                     # MinerU layout category
    page_idx: int                 # 0-indexed physical page
    bbox: List[float]             # [x0, y0, x1, y1] on the source page
    text: str                     # OCR text; tables/images: linearised body + caption + footnote
    path: str                     # MinerU crop, relative to the document dir ("" if none)
    caption: Union[str, List[str]]  # as MinerU emits it: "" or a list of caption lines


class Page(TypedDict, total=False):
    page_idx: int
    image_path: str               # relative to the document dir, e.g. "images/page_1.png"
    ocr_text: str
    summary: str                  # page summary p_j (filled by summarize)
    entities: str                 # categorised entity list (filled by summarize)
    elements: List[Element]


class Section(TypedDict):
    id: str                       # "sec_00", "sec_01", ...
    title: str
    start_page: int               # 0-indexed, inclusive
    end_page: int                 # 0-indexed, inclusive
    summary: str                  # section index card s_i (filled by summarize)
    pages: List[Page]


class DocTree(TypedDict):
    doc_id: str
    source_dir: str               # "" in released trees
    num_pages: int
    root_summary: str             # concatenated section cards (filled by summarize)
    sections: List[Section]


def validate_tree(tree: Dict[str, Any], max_section_pages: int | None = None) -> List[str]:
    """Return a list of structural problems; an empty list means the tree is well formed.

    Checks: page ranges are in bounds and match each section's page list, every
    physical page is covered, sections are ordered, adjacent sections share their
    boundary page, and (optionally) no section exceeds ``max_section_pages``.
    """
    problems: List[str] = []
    n = tree.get("num_pages", 0)
    sections = tree.get("sections", [])
    if not sections:
        return ["no sections"]

    covered = set()
    for sec in sections:
        sid, start, end = sec["id"], sec["start_page"], sec["end_page"]
        if not (0 <= start <= end < n):
            problems.append(f"{sid}: range [{start}, {end}] outside [0, {n - 1}]")
        page_ids = [p["page_idx"] for p in sec["pages"]]
        if page_ids != list(range(start, end + 1)):
            problems.append(f"{sid}: pages {page_ids[:3]}... do not match range [{start}, {end}]")
        if max_section_pages is not None and len(page_ids) > max_section_pages:
            problems.append(f"{sid}: {len(page_ids)} pages > {max_section_pages}")
        for p in sec["pages"]:
            covered.add(p["page_idx"])
            if p.get("image_path", "").startswith("/"):
                problems.append(f"{sid}: absolute image_path on page {p['page_idx']}")

    missing = sorted(set(range(n)) - covered)
    if missing:
        problems.append(f"pages not covered by any section: {missing[:10]}")

    ids = [s["id"] for s in sections]
    if len(ids) != len(set(ids)):
        problems.append("duplicate section ids")

    for a, b in zip(sections, sections[1:]):
        if b["start_page"] < a["start_page"]:
            problems.append(f"{a['id']} -> {b['id']}: sections out of order")
        elif a["end_page"] < b["start_page"]:
            problems.append(f"{a['id']} -> {b['id']}: boundary page not shared")

    return problems
