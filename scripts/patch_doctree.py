#!/usr/bin/env python3
"""Refresh page labels on existing trees without rebuilding them (summaries are kept).

For each tree: ``page_label`` is set on every page from MinerU's
process/logical_pages.json, then the manual corrections in
configs/doctree_fixes/<dataset>.json are applied. Running it again gives the
same trees. build_doctree.py does the same for newly built trees.

Usage:
    python scripts/patch_doctree.py --dataset mmlongbench
    python scripts/patch_doctree.py --dataset mmlongbench --doc PI_2017.10.04_Automation_FINAL --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adanav.config import dataset_paths, load_config
from adanav.doctree import MinerUDocument
from adanav.doctree.fixes import apply_fixes, fixes_path, load_fixes, set_page_labels
from adanav.runs import git_state, now_iso, redacted_argv, write_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="dataset name in configs/datasets.yaml")
    ap.add_argument("--doc", action="append", help="only these doc ids (repeatable)")
    ap.add_argument("--fixes", type=Path, help="corrections file (default: configs/doctree_fixes/<dataset>.json)")
    ap.add_argument("--config", default=None, help="config file (default: configs/datasets.yaml)")
    ap.add_argument("--dry-run", action="store_true", help="report changes without writing")
    args = ap.parse_args()

    config = load_config(args.config) if args.config else load_config()
    ds = dataset_paths(args.dataset, config)
    tree_root, mineru_root = ds["doctree_dir"], ds["mineru_root"]
    fixes_file = args.fixes or fixes_path(args.dataset)
    fixes = load_fixes(fixes_file)
    unknown = sorted(set(fixes) - {p.name for p in tree_root.iterdir() if p.is_dir()})
    if unknown:
        sys.exit(f"{fixes_file}: no tree for {unknown}")

    docs = args.doc or sorted(p.name for p in tree_root.iterdir() if (p / "tree_index.json").exists())
    changed_docs, unlabeled = [], []
    for doc_id in docs:
        path = tree_root / doc_id / "tree_index.json"
        tree = json.loads(path.read_text(encoding="utf-8"))
        before = json.dumps(tree, ensure_ascii=False)
        labels = MinerUDocument(mineru_root / doc_id).page_labels()
        if not labels:
            unlabeled.append(doc_id)
        set_page_labels(tree, labels)
        applied = apply_fixes(tree, fixes)
        if applied:
            print(f"[FIX] {doc_id}: {'; '.join(applied)}")
        if json.dumps(tree, ensure_ascii=False) != before:
            changed_docs.append(doc_id)
            if not args.dry_run:
                path.write_text(json.dumps(tree, ensure_ascii=False), encoding="utf-8")

    if unlabeled:
        print(f"[WARN] no logical_pages.json labels for {len(unlabeled)} docs: {unlabeled[:5]}")
    print(f"{'would change' if args.dry_run else 'changed'} {len(changed_docs)} of {len(docs)} trees")

    if not args.dry_run and changed_docs:
        info_path = tree_root / "run_info.json"
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {"dataset": args.dataset}
        info.setdefault("patches", []).append({
            "time": now_iso(),
            "argv": redacted_argv(),
            "git": git_state(),
            "what": "page_label from logical_pages.json + manual fixes",
            "fixes": str(fixes_file),
            "fixes_sha256": hashlib.sha256(fixes_file.read_bytes()).hexdigest() if fixes_file.exists() else None,
            "changed_docs": len(changed_docs),
        })
        write_json(info_path, info)


if __name__ == "__main__":
    main()
