"""Build the structural document tree from MinerU output (paper §3.2).

Input, one directory per document::

    <doc_dir>/
    ├── <doc_id>_content_list.json     MinerU elements (falls back to process/content_list.adapted.json)
    ├── images/page_<N>.png            rendered pages, N 1-indexed; element crops live alongside
    └── process/
        ├── toc.json                   heading tree: {"headings": [{title, level, logical_page, children}]}
        └── logical_pages.json         {"pages": [{logical_page, physical_page (1-indexed)}]}

Steps, in the order the evaluated trees were produced:

1. Sections from the level-1 TOC headings. Logical page labels are resolved to
   physical pages through logical_pages.json. Adjacent sections share their
   boundary page (a section ends on the page where the next one starts).
2. A "Preface" section covers any pages before the first heading.
3. Sections longer than ``max_section_pages`` (τ = 30) are split on their
   sub-headings, or into ``split_chunk_pages``-page chunks when they have none,
   recursing up to ``split_max_depth`` levels.
4. MinerU elements are attached to their pages. Table and image elements get
   their text from the linearised table body, captions and footnotes, which is
   also appended to the page's OCR text.
5. Boundary pages that step 3 left unshared (fixed-size chunks) are shared by
   copying the next section's first page onto the end of the previous section.
"""

from __future__ import annotations

import copy
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schema import DocTree


@dataclass(frozen=True)
class SplitConfig:
    max_section_pages: int = 30   # τ: sections longer than this are split
    split_chunk_pages: int = 25   # chunk size when a long section has no sub-headings
    split_max_depth: int = 3


class MinerUDocument:
    """Read access to one MinerU output directory."""

    def __init__(self, doc_dir: str | Path):
        self.doc_dir = Path(doc_dir)
        self.doc_id = self.doc_dir.name
        self.logical_to_physical = self._load_page_mapping()

    # ---- raw inputs -------------------------------------------------------

    def load_toc(self) -> Dict[str, Any]:
        path = self.doc_dir / "process" / "toc.json"
        if not path.exists():
            print(f"[WARN] {self.doc_id}: toc.json not found")
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def load_content_list(self) -> List[Dict[str, Any]]:
        path = self.doc_dir / f"{self.doc_id}_content_list.json"
        if not path.exists():
            path = self.doc_dir / "process" / "content_list.adapted.json"
        if not path.exists():
            # PaperTab / FetaTab name the file after a MinerU task UUID.
            from .toc import find_content_list
            path = find_content_list(self.doc_dir)
        if path is None:
            print(f"[WARN] {self.doc_id}: content_list.json not found")
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def page_images(self) -> Dict[int, Path]:
        """0-indexed page -> rendered page image."""
        images_dir = self.doc_dir / "images"
        if not images_dir.exists():
            return {}
        pages = {}
        for img in images_dir.glob("page_*.png"):
            m = re.search(r"page_(\d+)", img.stem)
            if m:
                pages[int(m.group(1)) - 1] = img
        if not pages:
            # No page_N.png naming: assume the sorted PNGs are the pages in order.
            pages = dict(enumerate(sorted(images_dir.glob("*.png"))))
        return pages

    # ---- logical -> physical page labels ----------------------------------

    def _load_page_mapping(self) -> Dict[str, int]:
        """logical page label -> 0-indexed physical page.

        Two-up spreads carry a range label such as "2-3"; both numbers and the
        range string itself map onto the one physical page.
        """
        path = self.doc_dir / "process" / "logical_pages.json"
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] {self.doc_id}: failed to load logical_pages.json: {e}")
            return {}

        mapping: Dict[str, int] = {}
        for info in data.get("pages", []):
            logical, physical = info.get("logical_page"), info.get("physical_page")
            if logical is None or physical is None:
                continue
            idx = physical - 1
            if isinstance(logical, str) and "-" in logical and logical not in ("cover", "封面"):
                parts = logical.split("-")
                if len(parts) == 2:
                    try:
                        for num in range(int(parts[0]), int(parts[1]) + 1):
                            mapping[str(num)] = idx
                    except ValueError:
                        pass
                mapping[logical] = idx
            else:
                mapping[str(logical)] = idx
        return mapping

    def resolve_page(self, label: Any) -> Optional[int]:
        """Resolve a TOC page label to a 0-indexed physical page, or None.

        Integers are taken as physical indices already. Unmapped numeric labels
        fall back to ``label - 1`` (no front-matter offset).
        """
        if label is None:
            return None
        if isinstance(label, int):
            return label
        if not isinstance(label, str):
            return None
        if label.lower() in ("cover", "封面"):
            return self.logical_to_physical.get("cover", 0)
        if label in self.logical_to_physical:
            return self.logical_to_physical[label]
        try:
            num = int(label)
        except ValueError:
            return None
        return self.logical_to_physical.get(str(num), num - 1)


# ---- sections -------------------------------------------------------------


def _level1_sections(doc: MinerUDocument, toc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One section per level-1 heading; end_page is the next level-1 heading's start page."""
    headings = toc.get("headings", [])
    sections = []
    for i, heading in enumerate(headings):
        if heading.get("level") != 1:
            continue
        start = doc.resolve_page(heading.get("logical_page"))
        if start is None and heading.get("children"):
            start = doc.resolve_page(heading["children"][0].get("logical_page"))
        end = None
        for nxt in headings[i + 1:]:
            if nxt.get("level") == 1:
                end = doc.resolve_page(nxt.get("logical_page"))
                break
        sections.append({
            "id": f"sec_{len(sections):02d}",
            "title": heading.get("title", "Untitled"),
            "start_page": start,
            "end_page": end,
            "children_headings": heading.get("children", []),
        })
    return sections


def _normalise_ranges(sections: List[Dict[str, Any]], num_pages: int) -> List[Dict[str, Any]]:
    """Fill missing bounds, clamp to the document, and prepend a Preface if needed."""
    if not sections:
        return [{
            "id": "sec_00", "title": "Document Content",
            "start_page": 0, "end_page": num_pages - 1, "children_headings": [],
        }]

    for i, sec in enumerate(sections):
        if sec["end_page"] is None:
            nxt = sections[i + 1]["start_page"] if i + 1 < len(sections) else None
            sec["end_page"] = nxt if nxt is not None else num_pages - 1
        if sec["start_page"] is None:
            sec["start_page"] = 0
        sec["start_page"] = max(0, min(sec["start_page"], num_pages - 1))
        sec["end_page"] = max(0, min(sec["end_page"], num_pages - 1))
        if sec["start_page"] > sec["end_page"]:
            sec["end_page"] = sec["start_page"]

    if sections[0]["start_page"] > 0:
        sections.insert(0, {
            "id": "sec_preface", "title": "Preface",
            "start_page": 0, "end_page": sections[0]["start_page"], "children_headings": [],
        })
    return sections


def _split_by_children(doc: MinerUDocument, parent: Dict[str, Any]) -> List[Dict[str, Any]]:
    start, end, title = parent["start_page"], parent["end_page"], parent["title"]
    children = []
    for child in parent.get("children_headings", []):
        page = doc.resolve_page(child.get("logical_page"))
        if page is not None:
            children.append({
                "title": child.get("title", "Untitled"),
                "start_page": page,
                "children_headings": child.get("children", []),
            })
    children.sort(key=lambda c: c["start_page"])
    children = [c for c in children if start <= c["start_page"] <= end]
    if not children:
        return [parent]

    parts = []
    if children[0]["start_page"] > start:
        parts.append({
            "id": "", "title": f"{title} - Introduction",
            "start_page": start, "end_page": children[0]["start_page"], "children_headings": [],
        })
    for i, child in enumerate(children):
        child_end = children[i + 1]["start_page"] if i + 1 < len(children) else end
        child_end = max(min(child_end, end), child["start_page"])
        parts.append({
            "id": "", "title": f"{title} - {child['title']}",
            "start_page": child["start_page"], "end_page": child_end,
            "children_headings": child["children_headings"],
        })
    return parts


def _split_fixed(sec: Dict[str, Any], chunk: int) -> List[Dict[str, Any]]:
    # Chunks do not share boundary pages; _share_boundary_pages() fixes that afterwards.
    parts, lo, k = [], sec["start_page"], 1
    while lo <= sec["end_page"]:
        hi = min(lo + chunk - 1, sec["end_page"])
        parts.append({
            "id": "", "title": f"{sec['title']} (Part {k})",
            "start_page": lo, "end_page": hi, "children_headings": [],
        })
        lo, k = hi + 1, k + 1
    return parts


def _split_large_sections(
    doc: MinerUDocument, sections: List[Dict[str, Any]], cfg: SplitConfig, depth: int
) -> List[Dict[str, Any]]:
    if depth <= 0:
        return sections
    out = []
    for sec in sections:
        if sec["end_page"] - sec["start_page"] + 1 <= cfg.max_section_pages:
            out.append(sec)
            continue
        if sec.get("children_headings"):
            parts = _split_by_children(doc, sec)
        else:
            parts = _split_fixed(sec, cfg.split_chunk_pages)
        out.extend(_split_large_sections(doc, parts, cfg, depth - 1))
    # Renumbering here also turns "sec_preface" into "sec_00", as in the evaluated trees.
    for i, sec in enumerate(out):
        sec["id"] = f"sec_{i:02d}"
    return out


# ---- elements -------------------------------------------------------------


def _as_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return value
    return [value] if isinstance(value, str) and value else []


def html_table_to_text(html: str) -> str:
    """Linearise an HTML table: one line per row, cells separated by " | "."""
    if not html:
        return ""
    text = re.sub(r"<tr[^>]*>", "\n", html)
    text = re.sub(r"<t[dh][^>]*>", " | ", text)
    text = re.sub(r"<[^>]+>", "", text)
    lines = [line.strip() for line in text.strip().split("\n")]
    return "\n".join(line for line in lines if line and line != "|")


def element_text(item: Dict[str, Any]) -> str:
    """Text for a table or image content_list item: caption, linearised body, footnote."""
    kind = item.get("type", "")
    parts = []
    if kind == "table":
        caption = _as_list(item.get("table_caption", []))
        if caption:
            parts.append(f"Table: {'; '.join(caption)}")
        body = html_table_to_text(item.get("table_body", ""))
        if body:
            parts.append(body)
        footnote = _as_list(item.get("table_footnote", []))
        if footnote:
            parts.append(f"Note: {'; '.join(footnote)}")
    elif kind == "image":
        caption = _as_list(item.get("image_caption", []))
        if caption:
            parts.append(f"Figure: {'; '.join(caption)}")
        footnote = _as_list(item.get("image_footnote", []))
        if footnote:
            parts.append(f"Note: {'; '.join(footnote)}")
    return "\n".join(parts)


def _build_page(page_idx: int, items: List[Dict[str, Any]], image: str,
                by_crop: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    elements, extra = [], []
    for item in items:
        elem = {
            "id": item.get("node_id", f"elem_{page_idx}_{len(elements)}"),
            "type": item.get("type", "text"),
            "page_idx": page_idx,
            "bbox": item.get("bbox", []),
            "text": item.get("text", ""),
            "path": item.get("img_path", ""),
            "caption": item.get("table_caption", "") or item.get("image_caption", ""),
        }
        # Tables and images carry no "text" of their own; derive it from the item
        # whose crop they point to.
        source = by_crop.get(elem["path"]) if elem["path"] else None
        if source is not None:
            text = element_text(source)
            if text:
                elem["text"] = text
                extra.append(text)
        elements.append(elem)

    ocr = " ".join(i.get("text", "") for i in items if i.get("type") == "text")
    if extra:
        ocr = ocr + ("\n\n" if ocr else "") + "\n\n".join(extra)
    return {"page_idx": page_idx, "image_path": image, "ocr_text": ocr, "summary": "", "elements": elements}


def _share_boundary_pages(sections: List[Dict[str, Any]]) -> None:
    for cur, nxt in zip(sections, sections[1:]):
        if not cur["pages"] or not nxt["pages"]:
            continue
        first_next = nxt["pages"][0]["page_idx"]
        if cur["pages"][-1]["page_idx"] < first_next:
            cur["pages"].append(copy.deepcopy(nxt["pages"][0]))
            cur["end_page"] = first_next


# ---- entry point ----------------------------------------------------------


def build_tree(doc_dir: str | Path, cfg: SplitConfig = SplitConfig()) -> DocTree:
    """Build the structural tree for one MinerU document (summaries left empty)."""
    doc = MinerUDocument(doc_dir)
    items = doc.load_content_list()

    by_page: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_page[item.get("page_idx", 0)].append(item)
    by_crop = {item["img_path"]: item for item in items if item.get("img_path")}

    images = doc.page_images()
    num_pages = max(images) + 1 if images else len(by_page)
    image_rel = {i: p.relative_to(doc.doc_dir).as_posix() for i, p in images.items()}

    sections = _normalise_ranges(_level1_sections(doc, doc.load_toc()), num_pages)
    sections = _split_large_sections(doc, sections, cfg, cfg.split_max_depth)

    tree_sections = []
    for sec in sections:
        start, end = sec["start_page"], sec["end_page"]
        tree_sections.append({
            "id": sec["id"],
            "title": sec["title"],
            "start_page": start,
            "end_page": end,
            "summary": "",
            "pages": [
                _build_page(i, by_page.get(i, []), image_rel.get(i, ""), by_crop)
                for i in range(start, end + 1)
            ],
        })
    _share_boundary_pages(tree_sections)

    return {
        "doc_id": doc.doc_id,
        "source_dir": "",
        "num_pages": num_pages,
        "root_summary": "",
        "sections": tree_sections,
    }
