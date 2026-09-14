#!/usr/bin/env bash
# =============================================================================
# Cron watchdog for the 5-model generic-pool eval (see run_full_generic.sh).
#
# Claude Code session resets kill even setsid-detached children, so a crontab
# entry re-launches whatever is missing every 10 minutes:
#
#   */10 * * * * /home/t-dungnguyen/workspace/code/unified-cyber-taxonomy/pace/evaluations/generic_watchdog.sh
#
# Idempotent: checks each proxy / model run and starts only what is down.
# Every run is resumable, so restarts never lose scored items. Remove the
# crontab line (crontab -e) once all runs report full coverage.
# =============================================================================
set -u
REPO=/home/t-dungnguyen/workspace/code/unified-cyber-taxonomy
LOG=$REPO/pace/.runlogs
mkdir -p "$LOG"
exec >> "$LOG/watchdog.log" 2>&1
echo "--- watchdog $(date '+%F %T')"

# done-marker: stop everything once the final coverage confirms completion
[ -f "$LOG/ALL_DONE" ] && { echo "ALL_DONE marker present; exiting"; exit 0; }

# 1. coco daemon
if ! curl -s -m 5 -o /dev/null http://127.0.0.1:4141/v1/models; then
    echo "coco daemon down -> restart"
    COCO_ENABLE_OPENAI_ROUTES=1 coco restart
    sleep 3
fi

# 2. proxies
if ! pgrep -f "coco_index_proxy.py --port 4142" >/dev/null; then
    echo "coco index proxy down -> start"
    setsid nohup python3 "$REPO/unified-cyber/pace_cyber/scripts/coco_index_proxy.py" --port 4142 \
        >> "$LOG/coco_proxy.log" 2>&1 &
fi
if ! pgrep -f "azure_key_proxy.py --port 4321" >/dev/null; then
    echo "azure key proxy down -> start"
    setsid nohup python3 "$REPO/unified-cyber/pace_cyber/scripts/azure_key_proxy.py" --port 4321 \
        >> "$LOG/azure_proxy.log" 2>&1 &
fi
sleep 2

# 3. model runs
cd "$REPO/pace"
run_if_dead() {  # api_id out_name base_url keysrc tag
    if pgrep -f "run_full_generic.sh $1 " >/dev/null; then
        return
    fi
    echo "run $5 down -> relaunch"
    setsid nohup bash evaluations/run_full_generic.sh "$1" "$2" "$3" "$4" \
        >> "$LOG/full_$5.log" 2>&1 &
}
run_if_dead claude-opus-4.7            Claude-Opus-4.7  http://127.0.0.1:4142 coco opus47
run_if_dead claude-opus-4.8            Claude-Opus-4.8  http://127.0.0.1:4142 coco opus48
run_if_dead anthropic-gemini-3.7-flash Gemini-3.7-Flash http://127.0.0.1:4142 coco gem37
run_if_dead gpt-4.1                    GPT-4.1          http://127.0.0.1:4321 none gpt41
run_if_dead gpt-5.4                    GPT-5.4          http://127.0.0.1:4321 none gpt54
echo "alive after pass: $(pgrep -fc run_full_generic.sh || echo 0)"
