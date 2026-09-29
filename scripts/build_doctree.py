#!/usr/bin/env python3
"""Build structural document trees from MinerU output (summaries are added separately).

Usage:
    python scripts/build_doctree.py --dataset mmlongbench              # all documents
    python scripts/build_doctree.py --dataset mmlongbench --doc SnapNTell
    python scripts/build_doctree.py --dataset papertab --first 10
    python scripts/build_doctree.py --mineru-dir /path/<doc_id> --out /tmp/trees

Existing trees are not overwritten (they may already hold summaries) unless --force.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adanav.config import dataset_paths, load_config
from adanav.doctree import build_tree, validate_tree
from adanav.doctree.mineru import SplitConfig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", help="dataset name in configs/datasets.yaml")
    src.add_argument("--mineru-dir", type=Path, help="a single MinerU document directory")
    ap.add_argument("--doc", action="append", help="only these doc ids (repeatable)")
    ap.add_argument("--first", type=int, help="only the first N documents")
    ap.add_argument("--out", type=Path, help="output root (default: the dataset's doctree_dir)")
    ap.add_argument("--config", default=None, help="config file (default: configs/datasets.yaml)")
    ap.add_argument("--force", action="store_true", help="overwrite existing trees")
    ap.add_argument("--indent", type=int, default=None)
    args = ap.parse_args()

    config = load_config(args.config) if args.config else load_config()
    split = SplitConfig(**config.get("split", {}))

    if args.mineru_dir:
        doc_dirs = [args.mineru_dir]
        out_root = args.out or Path(".")
    else:
        ds = dataset_paths(args.dataset, config)
        root = ds["mineru_root"]
        doc_dirs = [root / d for d in args.doc] if args.doc else sorted(p for p in root.iterdir() if p.is_dir())
        out_root = args.out or ds["doctree_dir"]
    if args.first:
        doc_dirs = doc_dirs[: args.first]

    built = skipped = failed = 0
    for doc_dir in doc_dirs:
        out = out_root / doc_dir.name / "tree_index.json"
        if out.exists() and not args.force:
            skipped += 1
            continue
        try:
            tree = build_tree(doc_dir, split)
        except Exception as e:  # keep going; report at the end
            print(f"[FAIL] {doc_dir.name}: {e!r}")
            failed += 1
            continue
        if not tree["num_pages"]:
            print(f"[SKIP] {doc_dir.name}: no pages")
            skipped += 1
            continue
        for problem in validate_tree(tree):
            print(f"[WARN] {doc_dir.name}: {problem}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(tree, ensure_ascii=False, indent=args.indent), encoding="utf-8")
        built += 1

    print(f"built {built}, skipped {skipped}, failed {failed} -> {out_root}")


if __name__ == "__main__":
    main()
