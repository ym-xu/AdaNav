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

Besides the trees, <doctree_dir> receives
    run_info.json          the page / section config the trees were made with, and one
                           entry per run (command, git commit, counts, truncations)
    raw_responses.jsonl    every model response, unparsed, one JSON object per call
A run whose config differs from the one recorded in run_info.json stops before calling
the model, so one directory never mixes trees from two configs: give each config its own
doctree_dir (point --config at a copy of configs/datasets.yaml).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adanav.config import dataset_paths, load_config
from adanav.doctree.summarize import PageSummaryCache, Summarizer, config_conflict, refresh_visual_refs
from adanav.runs import git_state, now_iso, redacted_argv, write_json

RUN_INFO = "run_info.json"
RAW_LOG = "raw_responses.jsonl"


def is_complete(tree: dict) -> bool:
    """True when the root, every section and every page have a summary."""
    return bool(tree.get("root_summary")) and all(
        s.get("summary") and all(p.get("summary") for p in s["pages"]) for s in tree["sections"])


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)  # progress lines reach a redirected log as they happen
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--doc", action="append", help="only these doc ids (repeatable)")
    ap.add_argument("--first", type=int, help="only the first N documents")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--page-only", action="store_true", help="page summaries only")
    mode.add_argument("--section-only", action="store_true", help="regenerate section cards from existing page summaries")
    mode.add_argument("--refresh-refs", action="store_true", help="only recompute VISUAL_REFS blocks (offline)")
    ap.add_argument("--force", action="store_true", help="re-summarise trees that are already complete")
    ap.add_argument("--model", help="model name served at --base-url")
    ap.add_argument("--base-url", action="append", dest="base_urls", help="OpenAI-compatible endpoint (repeatable)")
    ap.add_argument("--api-key")
    ap.add_argument("--max-tokens", type=int, default=1024, help="per response (default 1024, as in the paper)")
    ap.add_argument("--temperature", type=float, default=0.1, help="default 0.1, as in the paper")
    sampling = ap.add_argument_group("sampling (not sent unless given, so the endpoint default applies)")
    sampling.add_argument("--top-p", type=float)
    sampling.add_argument("--top-k", type=int, help="vLLM extension")
    sampling.add_argument("--presence-penalty", type=float)
    sampling.add_argument("--repetition-penalty", type=float, help="vLLM extension")
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
                            cache=cache, page_workers=args.workers, max_tokens=args.max_tokens,
                            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                            presence_penalty=args.presence_penalty, repetition_penalty=args.repetition_penalty,
                            raw_log=tree_root / RAW_LOG)
    runs_pages, runs_sections = not args.section_only, not args.page_only

    # Refuse to mix configs within one doctree_dir.
    info_path = tree_root / RUN_INFO
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}

    def recorded(key):  # stored configs carry their fingerprint; compare without it
        return {k: v for k, v in info[key].items() if k != "fingerprint"} if info.get(key) else None

    conflicts = []
    if runs_pages:
        diff = config_conflict(recorded("page_config"), summarizer.page_config)
        if diff:
            conflicts.append(f"page config: {diff}")
    if runs_sections:
        diff = config_conflict(recorded("section_config"), summarizer.section_config)
        if diff:
            conflicts.append(f"section config: {diff}")
    if conflicts:
        sys.exit(f"[ERROR] {info_path} records trees made with a different config\n  "
                 + "\n  ".join(conflicts)
                 + "\nWrite this config's trees to another doctree_dir (a copy of configs/datasets.yaml "
                   "passed with --config); with an unchanged page config, --cache can point at this "
                   "directory's page_summaries.json to reuse its page summaries.")

    if cache is not None:
        same, other = cache.count(summarizer.page_fp)
        print(f"page cache {cache.path}: {same} entries for this config"
              + (f", {other} from other configs (not used)" if other else ""))

    info["dataset"] = args.dataset
    if runs_pages:
        info["page_config"] = {**summarizer.page_config, "fingerprint": summarizer.page_fp}
    if runs_sections:
        info["section_config"] = {**summarizer.section_config, "fingerprint": summarizer.section_fp}
    run = {"started": now_iso(), "finished": None, "status": "running", "argv": redacted_argv(),
           "git": git_state(), "base_urls": args.base_urls or [], "documents": len(doc_ids)}
    info.setdefault("runs", []).append(run)
    tree_root.mkdir(parents=True, exist_ok=True)
    write_json(info_path, info)

    def process(doc_id: str) -> str:
        tree = load(doc_id)
        if not (args.force or args.section_only) and is_complete(tree):
            return "skip"
        if not args.section_only:
            summarizer.summarize_pages(tree, image_root / doc_id)
        if not args.page_only:
            summarizer.summarize_sections(tree)
        save(tree)
        return "ok" if args.page_only or is_complete(tree) else "incomplete"

    counts = {"ok": 0, "incomplete": 0, "skip": 0, "fail": 0}
    try:
        with ThreadPoolExecutor(max_workers=args.doc_workers) as pool:
            futures = {pool.submit(process, d): d for d in doc_ids}
            for future in as_completed(futures):
                try:
                    status = future.result()
                except Exception as e:
                    print(f"[FAIL] {futures[future]}: {e!r}")
                    status = "fail"
                counts[status] += 1
                if status in ("ok", "incomplete"):
                    print(f"[{status.upper()}] {futures[future]}")
        run["status"] = "done"
    except BaseException:
        run["status"] = "aborted"
        raise
    finally:
        if cache is not None:
            cache.save()
        run.update(finished=now_iso(), counts=counts, stats=dict(sorted(summarizer.stats.items())))
        write_json(info_path, info)
    print(counts)
    print(dict(sorted(summarizer.stats.items())))


if __name__ == "__main__":
    main()
