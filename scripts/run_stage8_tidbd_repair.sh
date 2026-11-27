#!/usr/bin/env bash
# Stage 8A: does normalising AutoTIDBD's step sizes locally (only the weights in the trace) instead of globally (all
# 134M, as published) stop the collapse that froze stage 8's arm F, and does the network then play better?
# Everything but that one rule is arm F's recipe: 8×6, downgrading from 16384, late-game starts from 8192 in a fifth
# of games, seed 42, 8 threads, initial rate 0.01, θ 0.01, τ 10000.
#
# Usage (long runs go through caffeinate -i; keep the lid open and the power connected):
#   bash scripts/run_stage8_tidbd_repair.sh test                      # both rules' correctness tests
#   caffeinate -i bash scripts/run_stage8_tidbd_repair.sh diagnose    # replay from arm F's 1M checkpoint, attributing each normalisation
#   bash scripts/run_stage8_tidbd_repair.sh start CHECKPOINT          # name the shared pre-collapse checkpoint for the fork
#   caffeinate -i bash scripts/run_stage8_tidbd_repair.sh fork        # A1: control (global) and repair (local), 500k games each from it
#   caffeinate -i bash scripts/run_stage8_tidbd_repair.sh main        # A2: the local rule from scratch, to the next decision point (2M, 5M, 10M)
#   bash scripts/run_stage8_tidbd_repair.sh report                    # A1's tables
#
# Each command appends to experiments/stage8-tidbd-repair/logs/<command>.log, ending with its exit code. A run only
# resumes from its own checkpoints (the fork's first session excepted, which starts both arms from the shared one), and
# the trainer refuses a checkpoint trained under the other rule, so nothing restarts from zero or switches rule by
# accident. Stage 8's own checkpoints are read, never written.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=experiments/stage8-tidbd-repair
CHECKPOINTS=checkpoints/ntuple/stage8-tidbd-repair
MIN_FREE_GB=${MIN_FREE_GB:-8}
FORK_GAMES=500000
FORK_EVERY=100000
MAIN_STOPS=(100000 500000 1000000 1500000 2000000 3000000 5000000 7500000 10000000)
DECISION_STOPS=(2000000 5000000 10000000)
COMMAND=${1:-}
[ $# -gt 0 ] && shift

mkdir -p "$OUT/logs"
LOG=$OUT/logs/${COMMAND:-usage}.log
finish() {
    code=$?
    echo "[$(date -u +%FT%TZ)] $COMMAND exited with code $code"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
exec > >(tee -a "$LOG") 2>&1
echo "[$(date -u +%FT%TZ)] $COMMAND $* (caffeinated: $(pmset -g assertions 2>/dev/null | grep -c 'caffeinate' || true) assertions, commit $(git rev-parse --short HEAD))"

check_disk() {
    free=$(df -g . | awk 'NR == 2 { print $4 }')
    if [ "$free" -lt "$MIN_FREE_GB" ]; then
        echo "only $free GB of disk free: at least $MIN_FREE_GB GB is needed before training" >&2
        exit 1
    fi
}

recipe() {    # normalisation: arm F's recipe with this rule
    export NTUPLE_NETWORK=8x6 NTUPLE_DOWNGRADE_TILE=16384 NTUPLE_LATE_START_FRACTION=0.2 NTUPLE_LATE_START_TILE=8192 NTUPLE_SEED=42
    export NTUPLE_THREADS_AFTER=10000 NTUPLE_THREADS=${THREADS:-8}
    export NTUPLE_OPTIMIZER=autotidbd NTUPLE_TIDBD_INITIAL_ALPHA=0.01 NTUPLE_TIDBD_META_RATE=0.01 NTUPLE_TIDBD_TAU=10000 NTUPLE_TIDBD_NORMALISATION=$1
}

newest() {    # run → its latest.npz while a session is unfinished, else its last snapshot, else nothing
    if [ -f "$CHECKPOINTS/$1/latest.npz" ]; then
        echo "$CHECKPOINTS/$1/latest.npz"
    else
        { ls "$CHECKPOINTS/$1"/games_*.npz 2>/dev/null || true; } | sort -t_ -k2 -n | tail -1
    fi
}

games_in() {    # checkpoint → games played
    uv run python -c "
import json, sys
import numpy as np
with np.load(sys.argv[1]) as data: print(json.loads(str(data['metadata']))['games_played'])" "$1"
}

train_to() {    # run, normalisation, stop, then the checkpoint and flags to begin from if the run has none of its own
    run=$1 rule=$2 stop=$3
    shift 3
    recipe "$rule"
    from=$(newest "$run")
    begin=(--resume "$from")
    if [ -z "$from" ]; then
        if [ -d "$CHECKPOINTS/$run" ] && [ -n "$(ls "$CHECKPOINTS/$run")" ]; then
            echo "$CHECKPOINTS/$run isn't empty but holds no checkpoint to resume from: not starting it afresh" >&2
            exit 1
        fi
        begin=(${@+"$@"})
    fi
    if [ -z "$from" ] || [ "$(games_in "$from")" -lt "$stop" ]; then
        check_disk
        uv run python -u scripts/train_ntuple.py --run "stage8-tidbd-repair/$run" ${begin[@]+"${begin[@]}"} --games "$stop" --snapshot-at "$stop"
        if [ -f "$CHECKPOINTS/$run/games_$stop.npz" ]; then rm -f "$CHECKPOINTS/$run/latest.npz"; fi
    fi
}

evaluate() {    # checkpoint, name, depth
    uv run python scripts/stage8.py evaluate "$1" --name "$2" --depth "$3" --split dev | tr '\r' '\n'
}

case $COMMAND in
test)
    uv run pytest tests/test_autotidbd.py tests/test_autotidbd_local.py -q
    ;;
diagnose)
    uv run python -u scripts/stage8_repair.py diagnose "${CHECKPOINT:-checkpoints/ntuple/stage8-tidbd/arm-F/games_1000000.npz}" --games "${GAMES:-40000}" --single-games "${SINGLE_GAMES:-4000}"
    ;;
start)
    [ -f "${1:-}" ] || { echo "usage: start CHECKPOINT" >&2; exit 2; }
    echo "$1 $(shasum -a 256 "$1" | cut -d' ' -f1)" > "$OUT/start"
    chmod a-w "$1"
    echo "the fork starts from $1 ($(games_in "$1") games)"
    ;;
fork)
    [ -f "$OUT/start" ] || { echo "no starting checkpoint named: run 'start CHECKPOINT' first" >&2; exit 1; }
    read -r start sha < "$OUT/start"
    [ "$(shasum -a 256 "$start" | cut -d' ' -f1)" = "$sha" ] || { echo "$start has changed since it was named as the start" >&2; exit 1; }
    first=$(games_in "$start")
    evaluate "$start" "repair-start-$first" 1
    evaluate "$start" "repair-start-$first" 3
    for ((stop = first + FORK_EVERY; stop <= first + FORK_GAMES; stop += FORK_EVERY)); do
        train_to control global "$stop" --resume "$start"
        evaluate "$CHECKPOINTS/control/games_$stop.npz" "repair-control-$stop" 1
        train_to local local "$stop" --resume "$start" --fork-normalisation
        evaluate "$CHECKPOINTS/local/games_$stop.npz" "repair-local-$stop" 1
        uv run python scripts/stage8_repair.py compare
    done
    # Exact depth 3 at the end of the continuation, for both arms (DEPTH3_AT names other stops)
    for stop in ${DEPTH3_AT:-$((first + FORK_GAMES))}; do
        evaluate "$CHECKPOINTS/control/games_$stop.npz" "repair-control-$stop" 3
        evaluate "$CHECKPOINTS/local/games_$stop.npz" "repair-local-$stop" 3
    done
    uv run python scripts/stage8_repair.py compare
    ;;
main)
    from=$(newest scratch)
    done_games=0
    if [ -n "$from" ]; then done_games=$(games_in "$from"); fi
    target=${TARGET:-}
    if [ -z "$target" ]; then
        for stop in "${DECISION_STOPS[@]}"; do
            if [ "$done_games" -lt "$stop" ]; then target=$stop; break; fi
        done
    fi
    [ -n "$target" ] || { echo "the scratch run has reached the 10M budget cap: nothing more to train"; exit 0; }
    echo "local normalisation from scratch: $done_games games so far, training to $target"
    for stop in "${MAIN_STOPS[@]}"; do
        if [ "$stop" -gt "$target" ]; then break; fi
        train_to scratch local "$stop"
        evaluate "$CHECKPOINTS/scratch/games_$stop.npz" "repair-scratch-$stop" 1
        if [ "$stop" -ge 500000 ]; then evaluate "$CHECKPOINTS/scratch/games_$stop.npz" "repair-scratch-$stop" 3; fi
    done
    uv run python scripts/stage8_repair.py compare
    ;;
report)
    uv run python scripts/stage8_repair.py compare
    ;;
*)
    sed -n '2,20p' "$0"
    exit 2
    ;;
esac
