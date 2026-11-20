#!/usr/bin/env python3
"""Render each document's pages from its source PDF.

    default   <out>/<doc_id>/page_<N>.jpg (N from 1), for run_adanav.py --page-images <out>
    --png     <out>/<doc_id>/images/page_<N>.png, the layout of MinerU's page images that the
              trees' image_path points to; at --dpi 150 these match MinerU's page images, so
              <out> can stand in for mineru_root when answering questions

Documents are those with a tree in the dataset's doctree_dir; the PDF is
<pdf-dir>/<doc_id>.pdf and must have as many pages as the tree. Existing images are kept.

Usage:
    python scripts/render_pages.py --dataset mmlongbench \
        --pdf-dir /path/to/MMLongBench-Doc/documents --out data/images/mmlongbench_200dpi
    python scripts/render_pages.py --dataset mmlongbench --dpi 150 --png \
        --pdf-dir /path/to/MMLongBench-Doc/documents --out data/images/mmlongbench
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adanav.config import dataset_paths, load_config


def render(doc_id: str, tree_root: Path, pdf_dir: Path, out: Path, dpi: int, png: bool = False) -> str:
    import fitz  # PyMuPDF

    pdf = pdf_dir / f"{doc_id}.pdf"
    if not pdf.exists():
        return f"{doc_id}: no {pdf.name}"
    num_pages = json.loads((tree_root / doc_id / "tree_index.json").read_text(encoding="utf-8"))["num_pages"]
    with fitz.open(str(pdf)) as doc:
        if len(doc) != num_pages:
            return f"{doc_id}: {len(doc)} pages in the PDF, {num_pages} in the tree"
        folder = out / doc_id / "images" if png else out / doc_id
        folder.mkdir(parents=True, exist_ok=True)
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        for i, page in enumerate(doc):
            path = folder / f"page_{i + 1}.{'png' if png else 'jpg'}"
            if path.exists():
                continue
            pixmap = page.get_pixmap(matrix=matrix)
            if png:
                pixmap.save(str(path))
            else:
                pixmap.save(str(path), jpg_quality=92)
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="dataset name in configs/datasets.yaml")
    ap.add_argument("--pdf-dir", type=Path, required=True, help="directory of <doc_id>.pdf")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--png", action="store_true", help="<out>/<doc_id>/images/page_<N>.png (see above)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--config", default=None, help="config file (default: configs/datasets.yaml)")
    args = ap.parse_args()

    ds = dataset_paths(args.dataset, load_config(args.config) if args.config else load_config(), require_mineru=False)
    tree_root = ds["doctree_dir"]
    docs = sorted(p.name for p in tree_root.iterdir() if (p / "tree_index.json").exists())
    job = partial(render, tree_root=tree_root, pdf_dir=args.pdf_dir, out=args.out, dpi=args.dpi, png=args.png)
    with ProcessPoolExecutor(args.workers) as ex:
        problems = [msg for msg in ex.map(job, docs) if msg]
    for msg in problems:
        print(f"[WARN] {msg}")
    print(f"rendered {len(docs) - len(problems)} of {len(docs)} documents to {args.out}")


if __name__ == "__main__":
    main()
