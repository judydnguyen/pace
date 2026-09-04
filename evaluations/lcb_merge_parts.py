#!/usr/bin/env python3
"""Merge split-column parts (Model__pN) back into the model's real column.

A slow model's remaining instances get split across processes to parallelise
them. Those processes must NOT share an output file -- concurrent appends to
one CSV interleave rows -- so each writes Model__pN.csv and this merges them.

Rows are keyed by instance id; an id already present in the base column wins
(it was scored first), and the parts fill the gaps. Part files are removed on
success so they cannot be mistaken for real panel models.

    python evaluations/lcb_merge_parts.py Gemini-3.7-Flash
"""
import csv, glob, os, sys

BASE = "results/standardized_results/livecodebench"
SUBS = ["codeexecution", "codegeneration", "selfrepair", "testoutputprediction"]


def merge(model):
    for sub in SUBS:
        parts = sorted(glob.glob(f"{BASE}/{sub}/{model}__p*.csv"))
        if not parts:
            continue
        main = f"{BASE}/{sub}/{model}.csv"
        rows, metric = {}, "pass@1"
        if os.path.exists(main):
            for r in csv.DictReader(open(main)):
                rows[r["id"]] = r["score"]
                metric = r["metric_name"]
        added = 0
        for p in parts:
            for r in csv.DictReader(open(p)):
                if r["id"] not in rows:          # base column wins
                    rows[r["id"]] = r["score"]
                    added += 1
                metric = r["metric_name"]
        with open(main, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "score", "metric_name"])
            for i in sorted(rows, key=lambda x: (len(x), x)):
                w.writerow([i, rows[i], metric])
        for p in parts:
            os.remove(p)
        print(f"  {sub}: merged {len(parts)} parts, +{added} rows -> {len(rows)} total")


if __name__ == "__main__":
    for m in sys.argv[1:]:
        print(f"{m}:")
        merge(m)
