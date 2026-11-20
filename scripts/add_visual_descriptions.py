#!/usr/bin/env python3
"""Attach figure / table descriptions and recovered captions to the elements of existing trees.

Reads <mineru_root>/<doc_id>/process/media_llm_results.json (one record per image or
table crop, matched to tree elements by crop path) and sets, per element:
    description   the VLM description of the crop ("" when there is none)
    caption       the recovered caption, only where MinerU attached none
Trees are updated in place; summaries are kept. Running it again gives the same trees.

Usage:
    python scripts/add_visual_descriptions.py --dataset mmlongbench
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adanav.config import dataset_paths, load_config
from adanav.runs import git_state, now_iso, redacted_argv, write_json


def enrich(tree: dict, media: list, stats: Counter) -> None:
    by_path = {m["img_path"]: m for m in media if m.get("img_path")}
    for section in tree["sections"]:
        for page in section["pages"]:
            for elem in page.get("elements", []):
                if elem.get("type") not in ("image", "table") or not elem.get("path"):
                    continue
                m = by_path.get(elem["path"])
                if m is None:
                    continue
                desc = (m.get("description") or "").strip()
                elem["description"] = desc
                stats["described"] += bool(desc)
                caption = m.get("image_caption" if elem["type"] == "image" else "table_caption") or []
                if caption and not elem.get("caption"):
                    elem["caption"] = caption
                    stats["caption_added"] += 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="dataset name in configs/datasets.yaml")
    ap.add_argument("--doc", action="append", help="only these doc ids (repeatable)")
    ap.add_argument("--config", default=None, help="config file (default: configs/datasets.yaml)")
    args = ap.parse_args()

    ds = dataset_paths(args.dataset, load_config(args.config) if args.config else load_config())
    tree_root, mineru_root = ds["doctree_dir"], ds["mineru_root"]
    docs = args.doc or sorted(p.name for p in tree_root.iterdir() if (p / "tree_index.json").exists())
    stats, changed = Counter(), 0
    for doc_id in docs:
        path = tree_root / doc_id / "tree_index.json"
        media_path = mineru_root / doc_id / "process" / "media_llm_results.json"
        if not media_path.exists():
            stats["docs_without_media"] += 1
            continue
        tree = json.loads(path.read_text(encoding="utf-8"))
        before = json.dumps(tree, ensure_ascii=False)
        enrich(tree, json.loads(media_path.read_text(encoding="utf-8")), stats)
        after = json.dumps(tree, ensure_ascii=False)
        if after != before:
            path.write_text(after, encoding="utf-8")
            changed += 1
    print(f"changed {changed} of {len(docs)} trees; {dict(stats)}")

    if changed:
        info_path = tree_root / "run_info.json"
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {"dataset": args.dataset}
        info.setdefault("patches", []).append({
            "time": now_iso(), "argv": redacted_argv(), "git": git_state(),
            "what": "element descriptions and captions from process/media_llm_results.json",
            "changed_docs": changed, "stats": dict(stats),
        })
        write_json(info_path, info)


if __name__ == "__main__":
    main()
