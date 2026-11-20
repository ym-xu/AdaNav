#!/usr/bin/env python3
"""Answer a benchmark's questions with AdaNav and record everything each question did.

Usage:
    # full system, single round (paper Algorithm 1), four vLLM replicas
    python scripts/run_adanav.py --dataset mmlongbench --run-dir results/mmlongbench/agentic \
        --model Qwen/Qwen3-VL-30B-A3B-Instruct \
        --base-url http://127.0.0.1:8011/v1 --base-url http://127.0.0.1:8012/v1 ...

    --preset base                  AdaNav (Base): no rule stage, no GeneralAgent
    --no-specific / --no-general   ablations of the agentic preset
    --iterations 3                 reasoner may ask for more pages (reflection prompt)
    --retrieval-only               stop after navigation (enough for Recall / All-Hit / F1)

<run-dir> receives
    predictions.jsonl   one record per question: route, sections, pages, answer, and the
                        trace (every model response, every fallback)
    run_info.json       agent config, model, tree provenance, and one entry per run
Re-running the same command redoes only the questions without a successful record. A run
whose config differs from the recorded one stops: use another --run-dir.
Score with scripts/evaluate_adanav.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adanav.agents import AdaNav, AdaNavConfig
from adanav.config import dataset_paths, load_config
from adanav.doctree.summarize import config_conflict
from adanav.eval.benchmark import load_samples
from adanav.runs import git_state, now_iso, redacted_argv, write_json

PREDICTIONS = "predictions.jsonl"
RUN_INFO = "run_info.json"


def read_predictions(path: Path) -> dict:
    """qid -> latest record (a re-run appends; the last line for a qid wins)."""
    records = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                records[r["qid"]] = r
    return records


def tree_provenance(doctree_dir: Path) -> dict:
    """Which summary config the trees were made with, and their latest page-label patch
    (from the run_info.json of summarize_doctree.py / patch_doctree.py)."""
    info_path = doctree_dir / "run_info.json"
    if not info_path.exists():
        return {"doctree_dir": str(doctree_dir)}
    info = json.loads(info_path.read_text(encoding="utf-8"))
    provenance = {"doctree_dir": str(doctree_dir),
                  "page_config": (info.get("page_config") or {}).get("fingerprint"),
                  "section_config": (info.get("section_config") or {}).get("fingerprint")}
    if info.get("patches"):
        last = info["patches"][-1]
        provenance["patch"] = {"time": last.get("time"), "fixes_sha256": last.get("fixes_sha256")}
    return provenance


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", action="append", dest="base_urls", help="OpenAI-compatible endpoint (repeatable)")
    ap.add_argument("--api-key")
    ap.add_argument("--preset", choices=sorted(AdaNavConfig.PRESETS), default="agentic")
    ap.add_argument("--no-specific", action="store_true", help="no rule stage: SPECIFIC questions go to the router LLM")
    ap.add_argument("--no-general", action="store_true", help="global questions use TreeNav over all sections")
    ap.add_argument("--no-refilter", action="store_true")
    ap.add_argument("--top-k", type=int, help="sections chosen by the router (K, default 3)")
    ap.add_argument("--page-navigator", choices=("vision", "text"), help="default vision")
    ap.add_argument("--vision-samples", type=int, help="vision navigator samples per chunk (default 3)")
    ap.add_argument("--b-nav", type=int, help="pages per section for TreeNav (default 4)")
    ap.add_argument("--b-gen", type=int, help="pages per section for the GeneralAgent (default 10)")
    ap.add_argument("--iterations", type=int, help="reasoner rounds (default 1)")
    ap.add_argument("--reasoner-prompt", help="prompt name under prompts/ (default per preset)")
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--doc", action="append", help="only questions on these doc ids (repeatable)")
    ap.add_argument("--first", type=int, help="only the first N questions")
    ap.add_argument("--qids-file", type=Path, help="only the question ids listed in this file (one per line)")
    ap.add_argument("--workers", type=int, default=16, help="questions processed concurrently")
    ap.add_argument("--page-workers", type=int, default=4, help="sections navigated concurrently per question")
    ap.add_argument("--config", default=None, help="datasets config (default configs/datasets[.local].yaml)")
    args = ap.parse_args()

    overrides = {"top_k": args.top_k, "b_nav": args.b_nav, "b_gen": args.b_gen, "iterations": args.iterations,
                 "reasoner_prompt": args.reasoner_prompt, "page_navigator": args.page_navigator,
                 "vision_samples": args.vision_samples}
    if args.no_specific:
        overrides["use_specific"] = False
    if args.no_general:
        overrides["use_general"] = False
    if args.no_refilter:
        overrides["refilter"] = False
    agent_config = AdaNavConfig.preset(args.preset, **overrides)

    ds = dataset_paths(args.dataset, load_config(args.config) if args.config else load_config())
    if not ds.get("samples"):
        sys.exit(f"[ERROR] no 'samples' path for {args.dataset} in the datasets config")
    samples = load_samples(ds["samples"])
    if args.doc:
        samples = [s for s in samples if s["doc_id"] in set(args.doc)]
    if args.qids_file:
        wanted = set(args.qids_file.read_text().split())
        samples = [s for s in samples if s["qid"] in wanted]
    if args.first:
        samples = samples[: args.first]

    from adanav.llm import EndpointPool
    adanav = AdaNav(args.model, EndpointPool(args.base_urls or [None], args.api_key), agent_config,
                    page_workers=args.page_workers)

    # One run directory, one config.
    config = {"dataset": args.dataset, "model": args.model, "preset": args.preset,
              "retrieval_only": args.retrieval_only, "agent": agent_config.to_dict(),
              "trees": tree_provenance(ds["doctree_dir"])}
    info_path = args.run_dir / RUN_INFO
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    diff = config_conflict(info.get("config"), config)
    if diff:
        sys.exit(f"[ERROR] {info_path} records a different config: {diff}\nUse another --run-dir.")

    pred_path = args.run_dir / PREDICTIONS
    done = {q for q, r in read_predictions(pred_path).items() if "error" not in r}
    todo = [s for s in samples if s["qid"] not in done]
    print(f"{len(samples)} questions, {len(samples) - len(todo)} already done, {len(todo)} to run")

    info["config"] = config
    run = {"started": now_iso(), "finished": None, "status": "running", "argv": redacted_argv(),
           "git": git_state(), "base_urls": args.base_urls or [], "questions": len(todo)}
    info.setdefault("runs", []).append(run)
    write_json(info_path, info)

    trees, tree_lock, write_lock = {}, threading.Lock(), threading.Lock()

    def tree_for(doc_id):
        with tree_lock:
            if doc_id not in trees:
                trees[doc_id] = json.loads((ds["doctree_dir"] / doc_id / "tree_index.json").read_text(encoding="utf-8"))
            return trees[doc_id]

    def process(sample):
        record = {"qid": sample["qid"], "doc_id": sample["doc_id"], "question": sample["question"]}
        try:
            record.update(adanav.run(sample["question"], tree_for(sample["doc_id"]),
                                     ds["mineru_root"] / sample["doc_id"], retrieval_only=args.retrieval_only))
        except Exception as e:
            record["error"] = repr(e)
        with write_lock:
            with pred_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    counts = {"ok": 0, "error": 0}
    started = time.time()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process, s) for s in todo]
            for n, future in enumerate(as_completed(futures), 1):
                record = future.result()
                counts["error" if "error" in record else "ok"] += 1
                if "error" in record:
                    print(f"[ERROR] {record['qid']} {record['doc_id']}: {record['error'][:200]}")
                if n % 25 == 0 or n == len(todo):
                    rate = n / (time.time() - started)
                    print(f"{n}/{len(todo)} done ({counts['error']} errors), "
                          f"{rate * 60:.1f}/min, eta {(len(todo) - n) / rate / 60:.0f} min")
        run["status"] = "done"
    except BaseException:
        run["status"] = "aborted"
        raise
    finally:
        run.update(finished=now_iso(), counts=counts)
        write_json(info_path, info)
    print(counts)


if __name__ == "__main__":
    main()
