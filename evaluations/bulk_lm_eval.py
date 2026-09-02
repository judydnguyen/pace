#!/usr/bin/env python3
"""Bulk lm-eval runner for adding a NEW model's FULL columns.

Runs one lm_eval benchmark (ALL its tasks, ALL docs) for one model against an
OpenAI-compatible endpoint, writing --log_samples output under
results/raw_results/<bench>/<out_dir>/ in the exact layout the
scripts/standardize/standardize_*.py scripts consume.

Mirrors handlers/lm_eval.py's evaluation config (chat template, temperature 0,
max_gen_toks 16384, disable_seed) but evaluates every doc in one call with
concurrent requests, instead of one doc per process.

    python evaluations/bulk_lm_eval.py \
        --model claude-opus-4.7 --base-url http://127.0.0.1:4141 \
        --benchmark aime25 [--tasks aime25] [--out-dir claude-opus-4.7] \
        [--num-concurrent 8]
"""
import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
from evaluations.handlers.lm_eval import (  # noqa: E402
    LM_EVAL_HARNESS_DIR, LM_EVAL_TASK_PATHS,
)

if LM_EVAL_HARNESS_DIR not in sys.path:
    sys.path.insert(0, LM_EVAL_HARNESS_DIR)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "dummy"))
    ap.add_argument("--benchmark", required=True, choices=sorted(LM_EVAL_TASK_PATHS))
    ap.add_argument("--tasks", default=None,
                    help="comma-separated task names (default: every task in the "
                         "benchmark's include_path)")
    ap.add_argument("--out-dir", default=None,
                    help="dir name under raw_results/<bench>/ (default: --model)")
    ap.add_argument("--num-concurrent", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N docs")
    args = ap.parse_args()

    from lm_eval.tasks import TaskManager
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.loggers import EvaluationTracker

    os.environ["OPENAI_API_KEY"] = args.api_key
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    task_dir = LM_EVAL_TASK_PATHS[args.benchmark]
    tm = TaskManager(include_path=task_dir, include_defaults=False)
    tasks = (args.tasks.split(",") if args.tasks else sorted(tm.all_tasks))
    for t in tasks:
        assert t in tm.all_tasks, f"{t} not in {sorted(tm.all_tasks)}"
    print(f"benchmark={args.benchmark} tasks={tasks}", flush=True)

    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    results = simple_evaluate(
        model="openai-chat-completions",
        model_args=(f"model={args.model},base_url={url},"
                    f"num_concurrent={args.num_concurrent},disable_seed=true"),
        tasks=tasks,
        log_samples=True,
        apply_chat_template=True,
        gen_kwargs="max_gen_toks=16384,temperature=0.0",
        bootstrap_iters=0,
        task_manager=tm,
        confirm_run_unsafe_code=True,
        limit=args.limit,
    )

    out_root = _REPO / "results" / "raw_results" / args.benchmark / (args.out_dir or args.model)
    samples = results.pop("samples")
    tracker = EvaluationTracker(output_path=str(out_root.parent))
    # write in the standardizers' expected layout ourselves — one jsonl per task
    out_root.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    stamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S.%f")
    (out_root / f"results_{stamp}.json").write_text(
        json.dumps({k: v for k, v in results.items() if k != "config"} , default=str, indent=1))
    for task, rows in samples.items():
        p = out_root / f"samples_{task}_{stamp}.jsonl"
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
        print(f"wrote {p} ({len(rows)} rows)", flush=True)
    for task in tasks:
        agg = results.get("results", {}).get(task, {})
        print(f"AGG {task}: { {k: round(v,4) for k,v in agg.items() if isinstance(v,(int,float))} }", flush=True)


if __name__ == "__main__":
    main()
