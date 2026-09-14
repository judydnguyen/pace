#!/usr/bin/env python3
"""Batch-score a new model on the 8 lm_eval generic benchmarks.

`score_new_model.py` drives lm-evaluation-harness ONE instance per call and
serializes those calls behind a lock (the harness mutates global state), so the
~20k lm_eval items of the full generic pool would take days per model. This
runner keeps the exact same scoring semantics — it reuses `eval_call` /
`extract` / `standardized_id_to_doc_id` from score_new_model — but invokes
lm_eval ONE TIME per subtask with all wanted doc_ids and `num_concurrent`
parallel API calls, then fans the logged samples back out into the same
standardized CSVs.

    python evaluations/batch_score_lm_eval.py \
        --model claude-opus-4.7 --model-name-out Claude-Opus-4.7 \
        --base-url http://127.0.0.1:4141 --api-key ... \
        --selections scripts/pacebench/selections/full_pool/selections_full_generic.csv \
        --benchmarks gpqa,mmlu_cot --num-concurrent 16

Resumable: rows already present in the output CSVs are skipped, and CSVs are
flushed after every subtask. Covers ONLY the lm_eval family (acp_gen, aime25,
gpqa, humaneval_chat, ifeval, logiqa, mbpp_chat, mmlu_cot); run infobench /
lifbench / livecodebench through score_new_model.py.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from evaluations._env import load_env  # noqa: E402
from evaluations.score_new_model import eval_call, extract, _STD  # noqa: E402
from evaluations.handlers.lm_eval import (  # noqa: E402
    LM_EVAL_DEFAULT_SUBTASK, LM_EVAL_TASK_PATHS, standardized_id_to_doc_id)

LM_EVAL_BENCHMARKS = ["acp_gen", "aime25", "gpqa", "humaneval_chat",
                      "ifeval", "logiqa", "mbpp_chat", "mmlu_cot"]


def load_existing(out_csv):
    if not out_csv.exists():
        return {}
    return {r["id"]: (r["score"], r["metric_name"])
            for r in csv.DictReader(open(out_csv))}


def flush(out_name, per_leaf):
    """per_leaf: {(benchmark, subdir): [(id, score, metric), ...]}"""
    for (b, subdir), triples in per_leaf.items():
        if not triples:
            continue
        out_dir = _STD / b / subdir if subdir else _STD / b
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"{out_name}.csv"
        existing = load_existing(out_csv)
        for iid, score, metric in triples:
            existing[iid] = (score, metric)
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "score", "metric_name"])
            for iid, (score, metric) in existing.items():
                w.writerow([iid, score, metric])
        triples.clear()


def main():
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--selections", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-name-out", default=None)
    ap.add_argument("--base-url", default=os.environ.get("BASE_URL"))
    ap.add_argument("--api-key", default=os.environ.get("API_KEY"))
    ap.add_argument("--benchmarks", default=",".join(LM_EVAL_BENCHMARKS),
                    help="comma-separated subset of the lm_eval benchmarks")
    ap.add_argument("--num-concurrent", type=int, default=16)
    ap.add_argument("--max-gen-toks", type=int, default=16384)
    ap.add_argument("--limit-docs", type=int, default=0,
                    help="only first N docs per subtask (debug)")
    args = ap.parse_args()
    if not (args.api_key and args.base_url):
        ap.error("need --api-key and --base-url")
    out_name = args.model_name_out or args.model
    wanted_benchmarks = [b for b in args.benchmarks.split(",") if b]
    bad = set(wanted_benchmarks) - set(LM_EVAL_BENCHMARKS)
    if bad:
        ap.error(f"not lm_eval benchmarks: {sorted(bad)} "
                 f"(use score_new_model.py for those)")

    # selections rows -> per (benchmark, subtask): {doc_id: [(subdir, row_id)]}
    plan = defaultdict(lambda: defaultdict(list))
    for row in csv.DictReader(open(args.selections)):
        b, subdir, iid = row["benchmark"], row["subdir"], row["instance_id"]
        if b not in wanted_benchmarks:
            continue
        subtask, call_iid = eval_call(b, subdir, iid)
        if subtask is None:
            subtask = LM_EVAL_DEFAULT_SUBTASK[b]
        doc_id = standardized_id_to_doc_id(b, subtask, call_iid)
        plan[(b, subtask)][doc_id].append((subdir, iid))

    # drop rows already scored (resume)
    done_cache = {}
    for (b, subtask), docs in plan.items():
        for doc_id in list(docs):
            rows = docs[doc_id]
            missing = []
            for subdir, iid in rows:
                out_csv = (_STD / b / subdir if subdir else _STD / b) / f"{out_name}.csv"
                key = (b, subdir)
                if key not in done_cache:
                    done_cache[key] = set(load_existing(out_csv))
                if iid not in done_cache[key]:
                    missing.append((subdir, iid))
            if missing:
                docs[doc_id] = missing
            else:
                del docs[doc_id]

    os.environ["OPENAI_API_KEY"] = args.api_key
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    chat_url = args.base_url.rstrip("/") + "/v1/chat/completions"

    from evaluations.handlers.lm_eval import LM_EVAL_HARNESS_DIR
    if LM_EVAL_HARNESS_DIR not in sys.path:
        sys.path.insert(0, LM_EVAL_HARNESS_DIR)
    from lm_eval.tasks import TaskManager
    from lm_eval.evaluator import simple_evaluate

    total_subtasks = sum(1 for docs in plan.values() if docs)
    n_sub = 0
    for (b, subtask), docs in sorted(plan.items()):
        if not docs:
            continue
        n_sub += 1
        doc_ids = sorted(docs)
        if args.limit_docs:
            doc_ids = doc_ids[:args.limit_docs]
        print(f"[{n_sub}/{total_subtasks}] {b}/{subtask}: {len(doc_ids)} docs",
              flush=True)
        tm = TaskManager(include_path=LM_EVAL_TASK_PATHS[b],
                         include_defaults=False)

        def _evaluate(gen_kwargs):
            return simple_evaluate(
                model="openai-chat-completions",
                model_args=(f"model={args.model},base_url={chat_url},"
                            f"num_concurrent={args.num_concurrent},"
                            f"disable_seed=true"),
                tasks=[subtask],
                samples={subtask: doc_ids},
                log_samples=True,
                apply_chat_template=True,
                gen_kwargs=gen_kwargs,
                bootstrap_iters=0,
                task_manager=tm,
                confirm_run_unsafe_code=True,
            )

        try:
            results = _evaluate(f"max_gen_toks={args.max_gen_toks},temperature=0.0")
        except Exception as exc:
            if "temperature" not in str(exc).lower():
                raise
            results = _evaluate(f"max_gen_toks={args.max_gen_toks}")

        samples = results.get("samples", {}).get(subtask, [])
        by_doc = defaultdict(list)   # doc_id -> [sample per filter]
        for s in samples:
            by_doc[s.get("doc_id")].append(s)

        per_leaf = defaultdict(list)
        n_written = n_missing = 0
        for doc_id in doc_ids:
            matched = by_doc.get(doc_id)
            if not matched:
                n_missing += 1
                continue
            for subdir, iid in docs[doc_id]:
                ex = extract(b, subdir, matched, instance_id=iid)
                if ex is None or ex[1] is None:
                    n_missing += 1
                    continue
                metric, score = ex
                per_leaf[(b, subdir)].append((iid, score, metric))
                n_written += 1
        flush(out_name, per_leaf)
        for key in per_leaf:
            done_cache.pop(key, None)   # invalidate; reloaded on next resume check
        print(f"    wrote {n_written} rows"
              + (f", {n_missing} unscorable" if n_missing else ""), flush=True)

    print(f"done: {n_sub} subtasks -> {_STD}/<benchmark>/.../{out_name}.csv")


if __name__ == "__main__":
    main()
