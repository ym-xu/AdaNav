#!/usr/bin/env python3
"""Fill page summaries, section cards and the root summary into built trees.

Usage:
    # all documents, two vLLM replicas
    python scripts/summarize_doctree.py --dataset mmlongbench \
        --model Qwen/Qwen3-VL-30B-A3B-Instruct \
        --base-url http://127.0.0.1:8011/v1 --base-url http://127.0.0.1:8012/v1

    python scripts/summarize_doctree.py --dataset papertab --doc 1503.00841 --model ...
    python scripts/summarize_doctree.py --dataset mmlongbench --section-only --force --model ...
    python scripts/summarize_doctree.py --dataset mmlongbench --refresh-refs   # no model calls

Complete trees are skipped unless --force or --section-only; incomplete ones are
redone, with page summaries served from <doctree_dir>/page_summaries.json (see --cache),
so re-running the same command retries only failed requests.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adanav.config import dataset_paths, load_config
from adanav.doctree.summarize import PageSummaryCache, Summarizer, refresh_visual_refs


def is_complete(tree: dict) -> bool:
    """True when the root, every section and every page have a summary."""
    return bool(tree.get("root_summary")) and all(
        s.get("summary") and all(p.get("summary") for p in s["pages"]) for s in tree["sections"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--doc", action="append", help="only these doc ids (repeatable)")
    ap.add_argument("--first", type=int, help="only the first N documents")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--page-only", action="store_true", help="page summaries only")
    mode.add_argument("--section-only", action="store_true", help="regenerate section cards from existing page summaries")
    mode.add_argument("--refresh-refs", action="store_true", help="only recompute VISUAL_REFS blocks (offline)")
    ap.add_argument("--force", action="store_true", help="re-summarise trees that already have section cards")
    ap.add_argument("--model", help="model name served at --base-url")
    ap.add_argument("--base-url", action="append", dest="base_urls", help="OpenAI-compatible endpoint (repeatable)")
    ap.add_argument("--api-key")
    ap.add_argument("--cache", type=Path, help="page summary cache file")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--workers", type=int, default=4, help="concurrent page requests per document")
    ap.add_argument("--doc-workers", type=int, default=2, help="documents processed concurrently")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    config = load_config(args.config) if args.config else load_config()
    ds = dataset_paths(args.dataset, config)
    tree_root, image_root = ds["doctree_dir"], ds["mineru_root"]

    doc_ids = args.doc or sorted(p.parent.name for p in tree_root.glob("*/tree_index.json"))
    missing = [d for d in doc_ids if not (tree_root / d / "tree_index.json").exists()]
    if missing:
        print(f"[SKIP] no tree for {missing}; run build_doctree.py first")
        doc_ids = [d for d in doc_ids if d not in missing]
    if args.first:
        doc_ids = doc_ids[: args.first]

    def load(doc_id):
        return json.loads((tree_root / doc_id / "tree_index.json").read_text(encoding="utf-8"))

    def save(tree):
        (tree_root / tree["doc_id"] / "tree_index.json").write_text(
            json.dumps(tree, ensure_ascii=False), encoding="utf-8")

    if args.refresh_refs:
        changed = 0
        for doc_id in doc_ids:
            tree = load(doc_id)
            if refresh_visual_refs(tree):
                save(tree)
                changed += 1
        print(f"refreshed visual refs in {changed}/{len(doc_ids)} trees")
        return

    if not args.model:
        ap.error("--model is required")
    from adanav.llm import EndpointPool

    cache = None if args.no_cache else PageSummaryCache(args.cache or tree_root / "page_summaries.json")
    summarizer = Summarizer(args.model, EndpointPool(args.base_urls or [None], args.api_key),
                            cache=cache, page_workers=args.workers)

    def process(doc_id: str) -> str:
        tree = load(doc_id)
        if not (args.force or args.section_only) and is_complete(tree):
            return "skip"
        if not args.section_only:
            summarizer.summarize_pages(tree, image_root / doc_id)
        if not args.page_only:
            summarizer.summarize_sections(tree)
        save(tree)
        return "ok"

    counts = {"ok": 0, "skip": 0, "fail": 0}
    with ThreadPoolExecutor(max_workers=args.doc_workers) as pool:
        futures = {pool.submit(process, d): d for d in doc_ids}
        for future in as_completed(futures):
            try:
                status = future.result()
            except Exception as e:
                print(f"[FAIL] {futures[future]}: {e!r}")
                status = "fail"
            counts[status] += 1
            if status == "ok":
                print(f"[OK] {futures[future]}")
    if cache is not None:
        cache.save()
    print(counts)


if __name__ == "__main__":
    main()
