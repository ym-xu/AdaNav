#!/usr/bin/env python3
"""Judge run answers with SimpleDoc's binary prompt through the OpenAI Batch API (half price).

Collects every answer of the given runs that has no judgment yet under
(protocol "simpledoc", --model), sends each distinct (question, answer) once, and
appends the results to each run's judgments.jsonl in the format evaluate_adanav.py
caches. evaluate_adanav.py --judge-model <model> then scores
the runs without calling the API (it still judges, synchronously, any answer whose
batch response could not be parsed).

Usage (key from $OPENAI_API_KEY):
    python scripts/judge_batch.py --run-dir results/mmlongbench/agentic --run-dir results/mmlongbench/base
    python scripts/judge_batch.py --run-dir ... --dry-run          # count requests and estimate tokens

The state (batch ids, request and response files) is kept in --state-dir, so an
interrupted call resumes the batches it already submitted instead of paying twice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adanav.config import dataset_paths, load_config
from adanav.eval.benchmark import load_samples
from adanav.eval.judge import PROMPT, normalize, parse_score
from adanav.runs import now_iso, write_json

PROTOCOL = "simpledoc"
JUDGMENTS = "judgments.jsonl"
DONE = ("completed", "failed", "expired", "cancelled")


def latest_predictions(run_dir: Path) -> dict:
    preds = {}
    for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            preds[r["qid"]] = r
    return {q: r for q, r in preds.items() if "error" not in r and "answer" in r}


def cached_keys(run_dir: Path, model: str) -> set:
    path = run_dir / JUDGMENTS
    keys = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            j = json.loads(line)
            if j.get("protocol", "simpledoc") == PROTOCOL and j["model"] == model:
                keys.add((j["qid"], j["answer"]))
    return keys


def custom_id(qid: str, answer: str) -> str:
    return f"{qid}-{hashlib.sha1(answer.encode('utf-8')).hexdigest()[:16]}"


def append_judgments(run_dir: Path, records: list) -> None:
    if records:
        with (run_dir / JUDGMENTS).open("a", encoding="utf-8") as f:
            for j in records:
                f.write(json.dumps(j, ensure_ascii=False) + "\n")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, action="append", required=True, dest="run_dirs")
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--state-dir", type=Path, help="default: <first run dir's parent>/judge_batches/<model>")
    ap.add_argument("--max-batch-tokens", type=int, default=800_000,
                    help="estimated input tokens per batch (keep under the account's batch queue limit)")
    ap.add_argument("--poll", type=int, default=60, help="seconds between status checks")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    state_dir = args.state_dir or args.run_dirs[0].parent / "judge_batches" / args.model
    state_path = state_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"model": args.model, "batches": []}
    config = load_config(args.config) if args.config else load_config()

    # ---- what needs judging ------------------------------------------------
    requests, wanted = {}, {}          # custom_id -> request body;  custom_id -> [(run_dir, record)]
    exact = 0
    for run_dir in args.run_dirs:
        info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
        samples = {s["qid"]: s for s in load_samples(dataset_paths(info["config"]["dataset"], config)["samples"])}
        have = cached_keys(run_dir, args.model)
        exact_records = []
        for qid, pred in latest_predictions(run_dir).items():
            answer = pred["answer"]
            if qid not in samples or (qid, answer) in have:
                continue
            s = samples[qid]
            record = {"qid": qid, "protocol": PROTOCOL, "model": args.model, "answer": answer, "gold": s["answer"]}
            if normalize(str(answer)) == normalize(str(s["answer"])):
                exact_records.append({**record, "score": 1, "how": "exact"})
                continue
            cid = custom_id(qid, str(answer))
            requests[cid] = {"model": args.model, "temperature": 0.0, "max_tokens": 512, "messages": [
                {"role": "user", "content": PROMPT.format(question=s["question"], pred=answer, gt=s["answer"])}]}
            wanted.setdefault(cid, []).append((run_dir, record))
        exact += len(exact_records)
        if not args.dry_run:
            append_judgments(run_dir, exact_records)
    est = {cid: len(r["messages"][0]["content"]) // 3 + 1 for cid, r in requests.items()}
    print(f"{len(args.run_dirs)} runs: {sum(len(v) for v in wanted.values())} answers to judge, "
          f"{len(requests)} distinct requests (~{sum(est.values()):,} input tokens), {exact} exact matches")
    if args.dry_run:
        return

    from openai import OpenAI
    client = OpenAI()

    def save_state():
        write_json(state_path, state)

    def collect(batch: dict) -> None:
        """Download a finished batch and write its judgments into every run that wants them."""
        remote = client.batches.retrieve(batch["id"])
        batch["status"], batch["counts"] = remote.status, remote.request_counts.model_dump()
        if remote.status not in DONE:
            return
        out_lines = []
        for file_id, name in ((remote.output_file_id, "output"), (remote.error_file_id, "errors")):
            if file_id:
                text = client.files.content(file_id).text
                (state_dir / f"{batch['name']}_{name}.jsonl").write_text(text, encoding="utf-8")
                out_lines += [json.loads(l) for l in text.splitlines() if l.strip()]
        per_run, failed = {}, 0
        for line in out_lines:
            cid, response = line["custom_id"], line.get("response") or {}
            if response.get("status_code") != 200:
                failed += 1
                continue
            text = (response["body"]["choices"][0]["message"]["content"] or "").strip()
            try:
                score = parse_score(text)
            except ValueError:
                failed += 1
                continue
            for run_dir, record in wanted.get(cid, []):
                per_run.setdefault(run_dir, []).append({**record, "score": score, "how": text})
        for run_dir, records in per_run.items():
            append_judgments(run_dir, records)
        batch.update(collected=now_iso(), failed=failed, written=sum(len(r) for r in per_run.values()))
        print(f"[{batch['name']}] {remote.status}: {batch['counts']}; wrote {batch['written']} judgments, "
              f"{failed} failed or unparseable")

    def wait(batch: dict) -> None:
        while "collected" not in batch:
            collect(batch)
            save_state()
            if "collected" not in batch:
                print(f"[{batch['name']}] {batch['status']} {batch['counts']}")
                time.sleep(args.poll)

    # ---- resume batches submitted earlier ---------------------------------
    for batch in state["batches"]:
        if "collected" not in batch:
            wait(batch)
            for cid in batch["custom_ids"]:
                requests.pop(cid, None)

    # ---- submit the rest, one batch at a time -------------------------------
    chunks, chunk, size = [], [], 0
    for cid in requests:
        if chunk and size + est[cid] > args.max_batch_tokens:
            chunks.append(chunk)
            chunk, size = [], 0
        chunk.append(cid)
        size += est[cid]
    if chunk:
        chunks.append(chunk)

    state_dir.mkdir(parents=True, exist_ok=True)
    for cids in chunks:
        name = f"batch_{len(state['batches']):03d}"
        path = state_dir / f"{name}_input.jsonl"
        path.write_text("".join(json.dumps({"custom_id": c, "method": "POST", "url": "/v1/chat/completions",
                                            "body": requests[c]}, ensure_ascii=False) + "\n" for c in cids),
                        encoding="utf-8")
        with path.open("rb") as f:
            file_id = client.files.create(file=f, purpose="batch").id
        remote = client.batches.create(input_file_id=file_id, endpoint="/v1/chat/completions",
                                       completion_window="24h", metadata={"source": "adanav judge_batch"})
        batch = {"name": name, "id": remote.id, "input_file": file_id, "submitted": now_iso(),
                 "requests": len(cids), "est_tokens": sum(est[c] for c in cids), "custom_ids": cids,
                 "runs": [str(r) for r in args.run_dirs]}
        state["batches"].append(batch)
        save_state()
        print(f"[{name}] submitted {remote.id}: {len(cids)} requests, ~{batch['est_tokens']:,} tokens")
        wait(batch)
        if batch["status"] != "completed":
            sys.exit(f"[{name}] ended {batch['status']}; see {state_path}")
    print("done")


if __name__ == "__main__":
    main()
