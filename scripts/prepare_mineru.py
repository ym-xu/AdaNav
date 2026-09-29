#!/usr/bin/env python3
"""Prepare MinerU output for tree building: render page images and derive a TOC.

Only needed for datasets whose MinerU output lacks them (PaperTab, FetaTab);
MMLongBench-Doc's output already has images/page_N.png and process/toc.json.
Files are written into the MinerU document directories.

Usage:
    python scripts/prepare_mineru.py --dataset papertab --render --toc
    python scripts/prepare_mineru.py --dataset fetatab --toc --doc "San Diego" --force
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adanav.config import dataset_paths, load_config
from adanav.doctree.render import render_pages
from adanav.doctree.toc import EXTRACTORS, write_toc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--render", action="store_true", help="render images/page_N.png from the PDF")
    ap.add_argument("--toc", action="store_true", help="derive process/toc.json + logical_pages.json")
    ap.add_argument("--doc", action="append", help="only these doc ids (repeatable)")
    ap.add_argument("--first", type=int)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="overwrite existing images / TOC files")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    if not (args.render or args.toc):
        ap.error("nothing to do: pass --render and/or --toc")

    config = load_config(args.config) if args.config else load_config()
    ds = dataset_paths(args.dataset, config)
    root = ds["mineru_root"]
    doc_dirs = ([root / d for d in args.doc] if args.doc
                else sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")))
    if args.first:
        doc_dirs = doc_dirs[: args.first]

    if args.render:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {d: pool.submit(render_pages, d, args.dpi, args.force) for d in doc_dirs}
        failed = 0
        for d, f in futures.items():
            try:
                f.result()
            except Exception as e:
                print(f"[FAIL] render {d.name}: {e}")
                failed += 1
        print(f"rendered {len(doc_dirs) - failed}/{len(doc_dirs)} documents")

    if args.toc:
        kind = ds.get("toc")
        if kind not in EXTRACTORS:
            ap.error(f"dataset '{args.dataset}' has toc: {kind!r}; derivation supports {sorted(EXTRACTORS)}")
        written = 0
        for d in doc_dirs:
            if write_toc(d, kind, force=args.force) is not None:
                written += 1
        print(f"wrote TOC for {written}/{len(doc_dirs)} documents (others already had one)")


if __name__ == "__main__":
    main()
