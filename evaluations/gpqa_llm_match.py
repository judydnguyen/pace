#!/usr/bin/env python3
"""LLM answer-matching for gpqa raw runs whose responses defeat regex filters.

Coco-served Claude models answer GPQA in idiosyncratic formats (own choice
relabeling, bare computed values), so lm-eval's filters — and even a combined
letter/choice-text matcher — under-credit them. This maps each response to a
choice letter with gpt-4o (azure proxy) and writes per-doc scores:

    results/raw_results/gpqa/<model>/llm_matched_<subset>.json
        {"accuracy": float, "n": int, "docs": {doc_id: 0|1}, "unmatched": int}

    python evaluations/gpqa_llm_match.py <model-dir-name> [--glob-dir _corrupt]
"""
import argparse
import glob
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

_REPO = Path(__file__).resolve().parents[1]
AZURE = "https://dwip-openai-ehe0b4f3cdctbfbp.westus2-01.azurewebsites.net/v1"


def azure_key() -> str:
    env = Path("/home/azureuser/workspace/unified-cyber-taxonomy/unified-cyber/suite/.env")
    for line in env.read_text().splitlines():
        if line.startswith("OPENAI_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("no azure key")


KEY = azure_key()
client = OpenAI(api_key=KEY, base_url=AZURE, default_headers={"X-API-Key": KEY})

PROMPT = """A model was asked a multiple-choice question. Decide which choice its response selects.

Question: {q}

Choices:
(A) {a}
(B) {b}
(C) {c}
(D) {d}

Model response (may use its own labels or state a value instead of a letter — match by MEANING):
---
{resp}
---

Reply with exactly one token: A, B, C, D, or NONE."""


def match_one(r):
    doc = r["doc"]
    choices = [doc[f"choice{i}"] for i in range(1, 5)]
    resp = (r["resps"][0][0] if r.get("resps") else "") or ""
    if not resp.strip():
        return r["doc_id"], None
    q = doc.get("Question", "")[:1500]
    msg = PROMPT.format(q=q, a=choices[0], b=choices[1], c=choices[2],
                        d=choices[3], resp=resp[-1200:])
    for _ in range(3):
        try:
            out = client.chat.completions.create(
                model="gpt-4o", temperature=0, max_tokens=5,
                messages=[{"role": "user", "content": msg}])
            ans = (out.choices[0].message.content or "").strip().upper()
            m = re.search(r"\b([A-D])\b", ans)
            if m:
                return r["doc_id"], f"({m.group(1)})"
            if "NONE" in ans:
                return r["doc_id"], None
        except Exception:
            continue
    return r["doc_id"], None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--glob-dir", default="",
                    help="subdir of the model dir holding the samples files")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    base = _REPO / "results" / "raw_results" / "gpqa" / args.model
    src = base / args.glob_dir if args.glob_dir else base
    for task in ("diamond", "extended", "main"):
        fs = sorted(glob.glob(str(src / f"samples_gpqa_{task}_cot_zeroshot_*.jsonl")))
        if not fs:
            print(f"{task}: no samples file under {src}")
            continue
        rows, seen = [], set()
        for line in open(fs[-1]):
            r = json.loads(line)
            if r["doc_id"] not in seen:
                seen.add(r["doc_id"])
                rows.append(r)
        targets = {r["doc_id"]: r["target"] for r in rows}
        with ThreadPoolExecutor(args.workers) as ex:
            results = list(ex.map(match_one, rows))
        docs = {str(d): (1 if let == targets[d] else 0) for d, let in results}
        unmatched = sum(1 for _, let in results if let is None)
        acc = sum(docs.values()) / len(docs)
        out = base / f"llm_matched_{task}.json"
        out.write_text(json.dumps(
            {"accuracy": acc, "n": len(docs), "unmatched": unmatched,
             "source": fs[-1], "docs": docs}, indent=1))
        print(f"{args.model} {task}: llm_matched_acc={acc:.4f} n={len(docs)} "
              f"unmatched={unmatched}")


if __name__ == "__main__":
    main()
