#!/usr/bin/env bash
# =============================================================================
# One-shot supervisor for finishing the 5-model generic-pool eval — NO cron.
#
#   setsid nohup bash evaluations/generic_supervisor.sh >> .runlogs/supervisor.log 2>&1 &
#
# Why it exists: running all five models' livecodebench stages at once (20
# torch-loaded shard processes) trips the kernel OOM killer on this 122GB box —
# that, not the restarts themselves, is what kept killing the runs. So this
# walks the models in PAIRS (bounded memory), restarting a pair's runs until
# that pair's coverage stops improving, then moves on. Exits when everything
# is complete (writes .runlogs/ALL_DONE) or after MAX_ROUNDS full sweeps.
#
# It keeps the coco daemon and both proxies alive itself (launched from a real
# shell so `coco` is on PATH — the cron attempt failed on exactly that).
# =============================================================================
set -u
export PATH="$HOME/.local/bin:$PATH"
REPO=/home/t-dungnguyen/workspace/code/unified-cyber-taxonomy
LOG=$REPO/pace/.runlogs
PY=$HOME/miniconda3/envs/pace-eval/bin/python
cd "$REPO/pace"

ensure_endpoints() {
    if ! curl -s -m 5 -o /dev/null http://127.0.0.1:4141/v1/models; then
        echo "[sup] coco daemon down -> restart"
        COCO_ENABLE_OPENAI_ROUTES=1 coco restart
        sleep 3
    fi
    pgrep -f "coco_index_proxy.py --port 4142" >/dev/null || {
        echo "[sup] start coco index proxy"
        setsid nohup python3 "$REPO/unified-cyber/pace_cyber/scripts/coco_index_proxy.py" --port 4142 >> "$LOG/coco_proxy.log" 2>&1 &
    }
    pgrep -f "azure_key_proxy.py --port 4321" >/dev/null || {
        echo "[sup] start azure key proxy"
        setsid nohup python3 "$REPO/unified-cyber/pace_cyber/scripts/azure_key_proxy.py" --port 4321 >> "$LOG/azure_proxy.log" 2>&1 &
    }
    sleep 2
}

# missing-count for one model (includes unmerged lcb shard credit)
missing() {
    "$PY" - "$1" <<'PYEOF'
import csv, sys
from pathlib import Path
m = sys.argv[1]
want = {}
for r in csv.DictReader(open("scripts/pacebench/selections/full_pool/selections_full_generic.csv")):
    want.setdefault((r["benchmark"], r["subdir"]), set()).add(r["instance_id"])
have_total = 0
total = sum(len(v) for v in want.values())
for (b, sub), ids in want.items():
    p = Path("results/standardized_results")/b/sub/f"{m}.csv" if sub else Path("results/standardized_results")/b/f"{m}.csv"
    have = {r["id"] for r in csv.DictReader(open(p))} if p.exists() else set()
    have_total += len(ids & have)
for sub in Path("results/standardized_results/livecodebench").iterdir():
    if sub.is_dir():
        fin = sub/f"{m}.csv"
        have_fin = {r["id"] for r in csv.DictReader(open(fin))} if fin.exists() else set()
        for i in range(4):
            sp = sub/f"{m}.lcbshard{i}.csv"
            if sp.exists():
                have_fin |= {r["id"] for r in csv.DictReader(open(sp))}
        # count shard-only ids once
        want_ids = want.get(("livecodebench", sub.name), set())
        have_total += len((want_ids & have_fin) - ({r["id"] for r in csv.DictReader(open(fin))} if fin.exists() else set()))
print(total - have_total)
PYEOF
}

run_pair() {  # up to 2 x "api_id|OutName|base|keysrc|tag"
    local pids=()
    for spec in "$@"; do
        IFS='|' read -r api out base key tag <<< "$spec"
        echo "[sup] $(date '+%T') launch $tag (missing $(missing "$out"))"
        bash evaluations/run_full_generic.sh "$api" "$out" "$base" "$key" >> "$LOG/full_$tag.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
}

M1="claude-opus-4.7|Claude-Opus-4.7|http://127.0.0.1:4142|coco|opus47"
M2="claude-opus-4.8|Claude-Opus-4.8|http://127.0.0.1:4142|coco|opus48"
M3="anthropic-gemini-3.7-flash|Gemini-3.7-Flash|http://127.0.0.1:4142|coco|gem37"
M4="gpt-4.1|GPT-4.1|http://127.0.0.1:4321|none|gpt41"
M5="gpt-5.4|GPT-5.4|http://127.0.0.1:4321|none|gpt54"

MAX_ROUNDS=6
for round in $(seq 1 $MAX_ROUNDS); do
    echo "[sup] ===== round $round $(date '+%F %T')"
    ensure_endpoints
    left_total=0
    for spec in "$M1" "$M2" "$M3" "$M4" "$M5"; do
        IFS='|' read -r _ out _ _ tag <<< "$spec"
        left=$(missing "$out")
        left_total=$((left_total + left))
        echo "[sup] $tag missing $left"
    done
    if [ "$left_total" -eq 0 ]; then
        echo "[sup] ALL COMPLETE $(date '+%F %T')"
        touch "$LOG/ALL_DONE"
        exit 0
    fi
    # pairs: memory-bounded (max 2 concurrent models)
    run_pair "$M1" "$M4"
    ensure_endpoints
    run_pair "$M2" "$M5"
    ensure_endpoints
    run_pair "$M3"
done
echo "[sup] max rounds reached, still missing items — inspect logs"
exit 1
