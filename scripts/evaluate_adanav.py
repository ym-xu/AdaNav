#!/usr/bin/env python3
"""Score a run of scripts/run_adanav.py: retrieval metrics, routing, and QA accuracy.

Usage:
    # accuracy judged by GPT-4.1 with SimpleDoc's binary prompt ($OPENAI_API_KEY)
    python scripts/evaluate_adanav.py --run-dir results/mmlongbench/agentic
    # retrieval and routing only
    python scripts/evaluate_adanav.py --run-dir results/mmlongbench/agentic --no-judge
    # another judge behind an OpenAI-compatible endpoint
    python scripts/evaluate_adanav.py --run-dir results/mmlongbench/agentic \
        --judge-model Qwen/Qwen3-VL-30B-A3B-Instruct --judge-base-url http://127.0.0.1:8011/v1

The judge answers 1/0 for (question, prediction, gold). scripts/judge_batch.py fills the
same cache through the OpenAI Batch API at half the price.
Every question of the dataset is scored; one without a successful prediction counts as
retrieving nothing and answering wrong (--subset scores only the questions that were run).
Judgments are cached in <run-dir>/judgments.jsonl per question, answer and judge model,
so re-scoring only calls the judge for new answers. Results go to <run-dir>/metrics.json
(GPT-4.1) or <run-dir>/metrics_simpledoc-<judge>.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adanav.config import dataset_paths, load_config
from adanav.eval.benchmark import load_samples
from adanav.eval.metrics import SOURCES, SPANS, qa_metrics, retrieval_metrics, routing
from adanav.runs import now_iso, write_json

JUDGMENTS = "judgments.jsonl"
PROTOCOL = "simpledoc"


def pct(x):
    return "  -  " if x is None else f"{100 * x:5.1f}"


def print_qa(title: str, qa: dict) -> None:
    cols = list(SOURCES) + list(SPANS) + ["ACC"]
    row = ([qa["by_source"][c]["acc"] for c in SOURCES] + [qa["by_span"][c]["acc"] for c in SPANS]
           + [qa["overall"]["acc"]])
    print(f"\n{title}\n  " + "  ".join(f"{c:>5}" for c in cols) + "\n  " + "  ".join(pct(x) for x in row))
    print("  by route: " + ", ".join(f"{c} {pct(v['acc'])} (n={v['n']})" for c, v in qa["by_category"].items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--judge-model", default="gpt-4.1")
    ap.add_argument("--judge-base-url", action="append", dest="judge_base_urls",
                    help="OpenAI-compatible endpoint serving --judge-model (repeatable; default: OpenAI API)")
    ap.add_argument("--api-key")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--judge-workers", type=int, default=16)
    ap.add_argument("--subset", action="store_true", help="score only questions present in predictions.jsonl")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    info = json.loads((args.run_dir / "run_info.json").read_text(encoding="utf-8"))
    dataset = info["config"]["dataset"]
    ds = dataset_paths(dataset, load_config(args.config) if args.config else load_config())
    samples = load_samples(ds["samples"])

    preds = {}
    for line in (args.run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            preds[r["qid"]] = r
    ok = {q: r for q, r in preds.items() if "error" not in r}
    if args.subset:
        samples = [s for s in samples if s["qid"] in preds]
    missing = [s["qid"] for s in samples if s["qid"] not in ok]
    print(f"{dataset}: {len(samples)} questions scored, {len(missing)} without a successful prediction")

    metrics = {"run_dir": str(args.run_dir), "scored_at": now_iso(), "questions": len(samples),
               "without_prediction": len(missing), "routing": routing(samples, ok),
               "retrieval": retrieval_metrics(samples, ok)}

    answered = [s for s in samples if s["qid"] in ok and "answer" in ok[s["qid"]]]
    judge_on = answered and not args.no_judge
    if judge_on:
        from adanav.eval.judge import Judge
        from adanav.llm import EndpointPool
        judge = Judge(args.judge_model, EndpointPool(args.judge_base_urls or [None], args.api_key))

        def score_one(s, answer):
            score, how = judge.score(s["question"], answer, s["answer"])
            return {"score": score, "how": how}

        protocol = PROTOCOL
        cache_path = args.run_dir / JUDGMENTS
        cache = {}
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                j = json.loads(line)
                cache[(j["qid"], j.get("protocol", "simpledoc"), j["model"], j["answer"])] = j
        lock = threading.Lock()

        def judged(s):
            answer = ok[s["qid"]]["answer"]
            key = (s["qid"], protocol, args.judge_model, answer)
            if key in cache:
                return cache[key]
            try:
                result = score_one(s, answer)
            except Exception as e:  # left out of the cache; the next run retries it
                print(f"[JUDGE ERROR] {s['qid']}: {e!r}"[:300])
                return None
            j = {"qid": s["qid"], "protocol": protocol, "model": args.judge_model, "answer": answer,
                 "gold": s["answer"], **result}
            with lock:
                with cache_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(j, ensure_ascii=False) + "\n")
            return j

        with ThreadPoolExecutor(max_workers=args.judge_workers) as pool_:
            results = [j for j in pool_.map(judged, answered) if j]
        judge_failed = len(answered) - len(results)
        qa = {"protocol": protocol, "judge": args.judge_model, "judge_failed": judge_failed,
              **qa_metrics(samples, ok, {j["qid"]: j["score"] for j in results})}
        metrics["qa"] = qa
        if judge_failed:
            print(f"[WARN] {judge_failed} answers could not be judged and count as wrong; re-run to retry")

    suffix = "" if not judge_on or args.judge_model == "gpt-4.1" else \
        "_" + re.sub(r"[^A-Za-z0-9.]+", "-", f"{protocol}_{args.judge_model.split('/')[-1]}")
    write_json(args.run_dir / f"metrics{suffix}.json", metrics)

    r = metrics["retrieval"]["overall"]
    print(f"\nrouting: {metrics['routing']}")
    print(f"retrieval (n={r['n']}): Recall {pct(r['recall'])}  All-Hit {pct(r['hit'])}  "
          f"Prec {pct(r['precision'])}  F1 {pct(r['f1'])}  Avg.Pages {r['pages'] or 0:.2f} "
          f"(all questions {metrics['retrieval']['avg_pages_all_questions'] or 0:.2f})")
    if "qa" in metrics:
        print_qa(f"QA ({metrics['qa']['protocol']}, judge {args.judge_model})", metrics["qa"])


if __name__ == "__main__":
    main()
