#!/usr/bin/env python3
"""Export document trees from a TreeDoc-style working directory into the release layout.

The trees produced during development store `image_path` and `source_dir` as
absolute paths on the machine that ran MinerU. This script rewrites them to be
relative to each document directory so the released trees are portable:

    <src>/<doc_id>/tree_index.json   ->   <dst>/<doc_id>/tree_index.json
      image_path: /abs/.../<doc_id>/images/page_7.png  ->  images/page_7.png
      source_dir: /abs/.../<doc_id>                    ->  "" (unset)

Usage:
    python scripts/export_doctree.py \
        --src /path/to/TreeDoc/data/processed \
        --dst data/doctree/mmlongbench
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def relativize(tree: dict) -> dict:
    """Rewrite machine-specific absolute paths in a tree index in place."""
    tree["source_dir"] = ""
    for section in tree.get("sections", []):
        for page in section.get("pages", []):
            image_path = page.get("image_path", "")
            if image_path:
                # Keep only the trailing "images/<file>" segment.
                page["image_path"] = "/".join(Path(image_path).parts[-2:])
            for element in page.get("elements", []):
                path = element.get("path", "")
                if path and Path(path).is_absolute():
                    element["path"] = "/".join(Path(path).parts[-2:])
    return tree


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True,
                        help="Directory holding <doc_id>/tree_index.json")
    parser.add_argument("--dst", type=Path, required=True,
                        help="Output directory for the released trees")
    parser.add_argument("--indent", type=int, default=None,
                        help="JSON indent (default: compact, smallest files)")
    args = parser.parse_args()

    sources = sorted(args.src.glob("*/tree_index.json"))
    if not sources:
        raise SystemExit(f"no tree_index.json found under {args.src}")

    args.dst.mkdir(parents=True, exist_ok=True)
    n_pages = 0
    n_sections = 0
    skipped = []
    for src in sources:
        tree = relativize(json.loads(src.read_text(encoding="utf-8")))
        if not tree.get("num_pages"):
            # Stray MinerU scratch folders parse into empty trees; not documents.
            skipped.append(src.parent.name)
            continue
        n_sections += len(tree.get("sections", []))
        n_pages += sum(len(s.get("pages", [])) for s in tree.get("sections", []))

        out_dir = args.dst / src.parent.name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "tree_index.json").write_text(
            json.dumps(tree, ensure_ascii=False, indent=args.indent),
            encoding="utf-8",
        )

    print(f"exported {len(sources) - len(skipped)} documents "
          f"({n_sections} sections, {n_pages} pages) to {args.dst}")
    if skipped:
        print(f"skipped {len(skipped)} empty tree(s): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
