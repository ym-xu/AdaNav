"""Render PDF pages to images/page_<N>.png (N 1-indexed) next to the MinerU output.

MMLongBench-Doc's MinerU output already contains the page images; PaperTab and
FetaTab need them rendered.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def find_pdf(doc_dir: Path) -> Optional[Path]:
    """The source PDF in a MinerU directory, preferring MinerU's *_origin.pdf copy."""
    for pattern in ("*_origin.pdf", "*.pdf"):
        found = sorted(doc_dir.glob(pattern))
        if found:
            return found[0]
    return None


def render_pages(doc_dir: Path, dpi: int = 150, force: bool = False) -> int:
    """Render every page of the document's PDF; returns the number of pages rendered.

    Documents that already have page images are left alone unless ``force``.
    """
    import fitz  # PyMuPDF

    images = doc_dir / "images"
    if not force and any(images.glob("page_*.png")):
        return 0
    pdf = find_pdf(doc_dir)
    if pdf is None:
        raise FileNotFoundError(f"no PDF in {doc_dir}")

    images.mkdir(exist_ok=True)
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    rendered = 0
    with fitz.open(str(pdf)) as doc:
        for i, page in enumerate(doc):
            out = images / f"page_{i + 1}.png"
            if out.exists() and not force:
                continue
            page.get_pixmap(matrix=matrix).save(str(out))
            rendered += 1
    return rendered
