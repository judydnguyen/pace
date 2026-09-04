#!/usr/bin/env python3
"""Sanity-gate new livecodebench columns before they are published.

Filling these columns needed many concurrent clients on the coco gateway, and
that gateway is known to swap responses between concurrent clients (it silently
corrupted gpqa seven times). LCB is safe to run that way only because the
corruption is LOUD here: every instance is scored by executing the returned code
against that problem's own tests, so a response belonging to a different problem
fails its tests. Crossing therefore shows up as a collapsed pass@1, not a
plausible-but-wrong number.

This compares each new column against the distribution of the already-published
columns for the same sub-leaf and flags anything implausibly low.

    python evaluations/lcb_validate.py Claude-5-Opus Grok-4.6 ...
"""
import csv, glob, os, statistics, sys

BASE = "results/standardized_results/livecodebench"
SUBS = ["codeexecution", "codegeneration", "selfrepair", "testoutputprediction"]
EXPECT = {"codeexecution": 100, "codegeneration": 1050,
          "selfrepair": 1055, "testoutputprediction": 100}


def col_mean(path):
    rows = [float(r["score"]) for r in csv.DictReader(open(path))]
    return (sum(rows) / len(rows), len(rows)) if rows else (0.0, 0)


def main(models):
    bad = False
    for sub in SUBS:
        pub = {}
        for f in glob.glob(f"{BASE}/{sub}/*.csv"):
            m = os.path.basename(f)[:-4]
            if m in models:
                continue
            mean, n = col_mean(f)
            if n:
                pub[m] = mean
        lo = min(pub.values()) if pub else 0.0
        med = statistics.median(pub.values()) if pub else 0.0
        print(f"\n{sub}: published n={len(pub)} min={lo:.3f} median={med:.3f}")
        for m in models:
            p = f"{BASE}/{sub}/{m}.csv"
            if not os.path.exists(p):
                print(f"  {m:18s} MISSING")
                bad = True
                continue
            mean, n = col_mean(p)
            want = EXPECT[sub]
            # a crossed run scores far below every real model; half the weakest
            # published column is a deliberately generous floor
            flags = []
            if n != want:
                flags.append(f"INCOMPLETE {n}/{want}")
            if n and mean < lo * 0.5:
                flags.append(f"IMPLAUSIBLE (<50% of weakest published {lo:.3f})")
            status = "  ".join(flags) if flags else "ok"
            if flags:
                bad = True
            print(f"  {m:18s} mean={mean:.3f} n={n:4d}  {status}")
    print("\nVALIDATION:", "FAILED" if bad else "PASSED")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
