#!/usr/bin/env bash
# =============================================================================
# Run ONE model over the full 11-benchmark generic pool (27,957 items).
#
#   bash evaluations/run_full_generic.sh <api-model-id> <OutName> <base-url> <api-key-source>
#
#   api-key-source: "coco" (read ~/.coco/daemon-auth.json), "none" (proxy that
#                   needs no key, e.g. azure_key_proxy), or a literal key.
#
# Examples (proxies must already be up — see NOTES):
#   bash evaluations/run_full_generic.sh claude-opus-4.7 Claude-Opus-4.7 http://127.0.0.1:4142 coco
#   bash evaluations/run_full_generic.sh gpt-4.1        GPT-4.1         http://127.0.0.1:4321 none
#
# NOTES
# - coco models go through coco_index_proxy (:4142) — raw coco omits choices[].index
#   and lm_eval then silently scores items 0. Azure models go through
#   azure_key_proxy (:4321) — the dwip gateway wants X-API-Key, not bearer.
#     python ../unified-cyber/pace_cyber/scripts/coco_index_proxy.py --port 4142 &
#     python ../unified-cyber/pace_cyber/scripts/azure_key_proxy.py  --port 4321 &
# - Everything is resumable: rerun the same command to fill gaps.
# - Stages: [1] 8 lm_eval benchmarks batched (dominant cost, num_concurrent api
#   calls); [2] infobench+lifbench threaded; [3] livecodebench in 4 sharded
#   processes (its handler is process-global, so parallelism = processes),
#   merged into the final CSV at the end.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."          # pace repo root

MODEL="${1:?api model id}"
OUT="${2:?output model name}"
BASE_URL="${3:?base url}"
KEYSRC="${4:?coco|none|<key>}"

PY="${PYTHON:-$HOME/miniconda3/envs/pace-eval/bin/python}"
SEL=scripts/pacebench/selections/full_pool
export HF_TOKEN="${HF_TOKEN:-$(grep '^HF_TOKEN' ../unified-cyber/.env | cut -d= -f2)}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
# Isolated cache: the shared ~/.cache/huggingface/datasets holds entries written
# by newer `datasets` versions (feature type 'List') that the pinned 3.6.0
# cannot read (seen on google/IFEval).
export HF_DATASETS_CACHE="$HOME/.cache/huggingface/datasets-pace"

case "$KEYSRC" in
  coco) KEY=$(python3 -c "import json;print(json.load(open('$HOME/.coco/daemon-auth.json'))['authToken'])");;
  none) KEY=unused;;
  *)    KEY="$KEYSRC";;
esac

echo "== [1/3] lm_eval benchmarks (batched) =="
"$PY" evaluations/batch_score_lm_eval.py \
    --selections $SEL/selections_lm_eval.csv \
    --model "$MODEL" --model-name-out "$OUT" \
    --base-url "$BASE_URL" --api-key "$KEY" --num-concurrent 12

echo "== [2/3] infobench + lifbench (threaded) =="
"$PY" evaluations/score_new_model.py \
    --selections $SEL/selections_info_lif.csv \
    --model "$MODEL" --model-name-out "$OUT" \
    --base-url "$BASE_URL" --api-key "$KEY" --workers 12

echo "== [3/3] livecodebench (4 sharded processes) =="
pids=()
for i in 0 1 2 3; do
    "$PY" evaluations/score_new_model.py \
        --selections $SEL/selections_lcb_shard$i.csv \
        --model "$MODEL" --model-name-out "$OUT.lcbshard$i" \
        --base-url "$BASE_URL" --api-key "$KEY" --workers 1 &
    pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done

"$PY" - "$OUT" <<'PYEOF'
import csv, sys
from pathlib import Path
out = sys.argv[1]
for leaf in Path("results/standardized_results/livecodebench").iterdir():
    if not leaf.is_dir():
        continue
    final = leaf / f"{out}.csv"
    merged = {}
    if final.exists():
        merged = {r["id"]: (r["score"], r["metric_name"])
                  for r in csv.DictReader(open(final))}
    n0 = len(merged)
    for i in range(4):
        shard = leaf / f"{out}.lcbshard{i}.csv"
        if shard.exists():
            for r in csv.DictReader(open(shard)):
                merged[r["id"]] = (r["score"], r["metric_name"])
    if len(merged) > n0:
        with open(final, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "score", "metric_name"])
            for iid, (s, m) in merged.items():
                w.writerow([iid, s, m])
    print(f"{leaf.name}: {len(merged)} rows ({len(merged)-n0} new)")
PYEOF
[ $rc -ne 0 ] && echo "WARN: an lcb shard exited non-zero (rerun to resume)"

echo "== coverage for $OUT =="
"$PY" - "$OUT" <<'PYEOF'
import csv, sys
from pathlib import Path
out = sys.argv[1]
sel = Path("scripts/pacebench/selections/full_pool/selections_full_generic.csv")
want = {}
for r in csv.DictReader(open(sel)):
    want.setdefault((r["benchmark"], r["subdir"]), set()).add(r["instance_id"])
total = have_total = 0
for (b, sub), ids in sorted(want.items()):
    p = Path("results/standardized_results") / b / sub / f"{out}.csv" if sub \
        else Path("results/standardized_results") / b / f"{out}.csv"
    have = {r["id"] for r in csv.DictReader(open(p))} if p.exists() else set()
    got = len(ids & have)
    total += len(ids); have_total += got
    flag = "" if got == len(ids) else "   <-- INCOMPLETE"
    print(f"{b}/{sub or '-'}: {got}/{len(ids)}{flag}")
print(f"TOTAL: {have_total}/{total}")
PYEOF
