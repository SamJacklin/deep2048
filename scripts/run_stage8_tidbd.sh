#!/usr/bin/env bash
# Stage 8: train the 8×6 network from scratch with AutoStep TIDBD(0) in place of temporal coherence, and compare it
# with the temporal coherence champion. Everything else is the long run's recipe: downgrading from 16384, late-game
# starts from 8192 in a fifth of games, seed 42, the first 10,000 games single-threaded.
#
# Usage (long training goes through caffeinate -i, so the Mac doesn't sleep while idle):
#   bash scripts/run_stage8_tidbd.sh test                    # the correctness tests, which training must pass first
#   caffeinate -i bash scripts/run_stage8_tidbd.sh bench     # throughput and stability on 1, 4 and 8 threads, against TC
#   caffeinate -i bash scripts/run_stage8_tidbd.sh screen    # round 1: arms A to F to 100k games, evaluated at 25k, 50k, 100k
#   caffeinate -i bash scripts/run_stage8_tidbd.sh extend C D    # round 2: two arms on to 300k games
#   bash scripts/run_stage8_tidbd.sh select D                # name the arm for the main run
#   caffeinate -i bash scripts/run_stage8_tidbd.sh main      # the selected arm, to the next decision point: 2M, 5M, then 10M
#   caffeinate -i bash scripts/run_stage8_tidbd.sh resume    # the same: an interrupted run carries on from its last checkpoint
#   caffeinate -i bash scripts/run_stage8_tidbd.sh baseline  # the champion and its earlier snapshots on the development seeds
#   bash scripts/run_stage8_tidbd.sh report                  # tables and figures from what's been evaluated
#
# THREADS sets the training threads (default: the benchmark's choice, in experiments/stage8-tidbd/threads, else 8).
# Each command appends to experiments/stage8-tidbd/logs/<command>.log, with its exit code on the last line. Training
# only ever resumes from an arm's own latest.npz, and the trainer refuses a checkpoint without the optimiser's state or
# with other settings, so a failed resume stops rather than starting afresh. Ctrl-C (or kill) saves a checkpoint first.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=experiments/stage8-tidbd
CHECKPOINTS=checkpoints/ntuple/stage8-tidbd
MIN_FREE_GB=${MIN_FREE_GB:-6}
SCREEN_STOPS=(25000 50000 100000)
EXTEND_STOPS=(150000 200000 250000 300000)
MAIN_STOPS=(500000 1000000 1500000 2000000 3000000 5000000 7500000 10000000)
DEPTH3_STOPS=" 500000 1000000 1500000 2000000 3000000 5000000 7500000 10000000 "
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
# Everything below goes to the log as well as the terminal. pipefail and the trap keep the real exit code.
exec > >(tee -a "$LOG") 2>&1
echo "[$(date -u +%FT%TZ)] $COMMAND $* (caffeinated: $(pmset -g assertions 2>/dev/null | grep -c 'caffeinate' || true) assertions, commit $(git rev-parse --short HEAD))"

rates() {    # arm → "initial rate, meta rate"
    case $1 in
        A) echo "0.00125 0.003" ;; B) echo "0.00125 0.01" ;; C) echo "0.004 0.003" ;;
        D) echo "0.004 0.01" ;; E) echo "0.01 0.003" ;; F) echo "0.01 0.01" ;;
        *) echo "unknown arm $1 (A to F)" >&2; exit 2 ;;
    esac
}

check_disk() {
    free=$(df -g . | awk 'NR == 2 { print $4 }')
    if [ "$free" -lt "$MIN_FREE_GB" ]; then
        echo "only $free GB of disk free: at least $MIN_FREE_GB GB is needed before training" >&2
        exit 1
    fi
}

recipe() {    # the champion's stage 1 recipe, with nothing but the learning rule changed
    export NTUPLE_NETWORK=8x6 NTUPLE_DOWNGRADE_TILE=16384 NTUPLE_LATE_START_FRACTION=0.2 NTUPLE_LATE_START_TILE=8192 NTUPLE_SEED=42
    export NTUPLE_THREADS_AFTER=10000 NTUPLE_THREADS=${THREADS:-$(cat "$OUT/threads" 2>/dev/null || echo 8)}
}

tidbd() {    # arm
    recipe
    set -- $(rates "$1")
    export NTUPLE_OPTIMIZER=autotidbd NTUPLE_TIDBD_INITIAL_ALPHA=$1 NTUPLE_TIDBD_META_RATE=$2 NTUPLE_TIDBD_TAU=10000
}

train_to() {    # run name, total games, snapshot game counts...
    run=$1 games=$2
    shift 2
    check_disk
    resume=()
    from=$(newest "$run")
    if [ -n "$from" ]; then
        resume=(--resume "$from")
    elif [ -d "$CHECKPOINTS/$run" ] && [ -n "$(ls "$CHECKPOINTS/$run")" ]; then
        echo "$CHECKPOINTS/$run isn't empty but holds no checkpoint to resume from: not starting it afresh" >&2
        exit 1
    fi
    # The odd expansion keeps an empty array working under set -u in macOS's bash 3.2
    uv run python -u scripts/train_ntuple.py --run "stage8-tidbd/$run" ${resume[@]+"${resume[@]}"} --games "$games" --snapshot-at "$@"
}

evaluate() {    # run name, games, depth, split
    uv run python scripts/stage8.py evaluate "$CHECKPOINTS/$1/games_$2.npz" --name "$1-$2" --depth "$3" --split "$4" | tr '\r' '\n'
}

# A run's newest checkpoint: latest.npz while a training session is unfinished, else its last snapshot. Once a stop's
# snapshot is saved, latest.npz holds the same state uncompressed (2.6 GB), so it's deleted and the snapshot resumes the run.
newest() {    # run name → path, or nothing
    if [ -f "$CHECKPOINTS/$1/latest.npz" ]; then
        echo "$CHECKPOINTS/$1/latest.npz"
    else
        ls "$CHECKPOINTS/$1"/games_*.npz 2>/dev/null | sort -t_ -k2 -n | tail -1
    fi
}

games_done() {    # run name → games in its newest checkpoint, 0 if there's none
    from=$(newest "$1")
    if [ -n "$from" ]; then
        uv run python -c "
import json, sys
import numpy as np
with np.load(sys.argv[1]) as data: print(json.loads(str(data['metadata']))['games_played'])" "$from"
    else
        echo 0
    fi
}

train_arm() {    # arm, split to evaluate greedily on, stops...
    arm=$1 split=$2
    shift 2
    tidbd "$arm"
    done_games=$(games_done "arm-$arm")
    for stop in "$@"; do
        if [ "$done_games" -lt "$stop" ]; then
            train_to "arm-$arm" "$stop" "$stop"
            [ -f "$CHECKPOINTS/arm-$arm/games_$stop.npz" ] && rm -f "$CHECKPOINTS/arm-$arm/latest.npz"
        fi
        evaluate "arm-$arm" "$stop" 1 "$split"
        if [ "$split" = dev ] && [[ $DEPTH3_STOPS == *" $stop "* ]]; then
            evaluate "arm-$arm" "$stop" 3 dev
        fi
    done
}

case $COMMAND in
test)
    uv run pytest tests/test_autotidbd.py tests/test_ntuple_train.py tests/test_ntuple_network.py tests/test_ntuple_search.py -q
    ;;
bench)
    # One TIDBD arm to 10k games single-threaded (the recipe's own start, and the single-thread benchmark), then three
    # branches from that checkpoint of 10k more games on 1, 4 and 8 threads, and temporal coherence the same way on 1 and 8
    bench_run() {    # name, branch-from name or "", total games, threads
        check_disk
        export NTUPLE_OUTPUT_DIR=$OUT/bench/$1 NTUPLE_CHECKPOINT_DIR=$CHECKPOINTS/bench/$1 NTUPLE_THREADS=$4 NTUPLE_THREADS_AFTER=10000
        [ -f "$NTUPLE_OUTPUT_DIR/manifest.json" ] && [ -f "$NTUPLE_CHECKPOINT_DIR/games_$3.npz" ] && return 0
        from=()
        [ -n "$2" ] && from=(--resume "$CHECKPOINTS/bench/$2/games_10000.npz")
        rm -rf "$NTUPLE_OUTPUT_DIR"
        uv run python -u scripts/train_ntuple.py ${from[@]+"${from[@]}"} --games "$3" --snapshot-at "$3"
        rm -f "$NTUPLE_CHECKPOINT_DIR/latest.npz"
    }
    BENCH_ARM=${BENCH_ARM:-D}
    tidbd "$BENCH_ARM"
    bench_run tidbd-start "" 10000 1
    for threads in 1 4 8 10; do
        bench_run "tidbd-${threads}-threads" tidbd-start 20000 "$threads"
    done
    unset NTUPLE_OPTIMIZER
    export NTUPLE_LEARNING_RATE=0.02 NTUPLE_TEMPORAL_COHERENCE=true
    bench_run tc-start "" 10000 1
    for threads in 1 8; do
        bench_run "tc-${threads}-threads" tc-start 20000 "$threads"
    done
    uv run python scripts/stage8.py bench | tee "$OUT/bench/summary.md"
    ;;
screen)
    for arm in ${ARMS:-A B C D E F}; do
        # An arm that fails (a non-finite update stops training) is recorded and the screen moves on to the next
        # (its own process, because bash ignores set -e inside an if's condition)
        if ! bash "$0" arm "$arm" screen "${SCREEN_STOPS[@]}"; then
            echo "arm $arm FAILED: see above" | tee -a "$OUT/screen-failures.txt"
        fi
    done
    uv run python scripts/stage8.py screen
    ;;
arm)    # internal: arm, split, stops...
    train_arm "$@"
    ;;
extend)
    [ $# -ge 1 ] || { echo "usage: extend ARM [ARM]" >&2; exit 2; }
    for arm in "$@"; do
        train_arm "$arm" screen "${EXTEND_STOPS[@]}"
    done
    uv run python scripts/stage8.py screen
    ;;
select)
    rates "$1" > /dev/null
    echo "$1" > "$OUT/selected-arm"
    echo "arm $1 selected for the main run"
    ;;
main | resume)
    [ -f "$OUT/selected-arm" ] || { echo "no arm selected: run 'select ARM' after the screen" >&2; exit 1; }
    arm=$(cat "$OUT/selected-arm")
    [ -f "$CHECKPOINTS/arm-$arm/latest.npz" ] || { echo "arm $arm has no checkpoint to continue from" >&2; exit 1; }
    done_games=$(games_done "arm-$arm")
    target=${TARGET:-}
    if [ -z "$target" ]; then
        for stop in "${DECISION_STOPS[@]}"; do
            if [ "$done_games" -lt "$stop" ]; then target=$stop; break; fi
        done
    fi
    [ -n "$target" ] || { echo "arm $arm has reached the 10M budget cap: nothing more to train"; exit 0; }
    echo "arm $arm: $done_games games so far, training to $target"
    stops=()
    for stop in 100000 250000 "${MAIN_STOPS[@]}"; do
        if [ "$stop" -le "$target" ]; then stops+=("$stop"); fi
    done
    train_arm "$arm" dev "${stops[@]}"
    uv run python scripts/stage8.py curve "$arm"
    ;;
baseline)
    uv run python scripts/stage8.py baseline | tr '\r' '\n'
    ;;
report)
    uv run python scripts/stage8.py screen
    if [ -f "$OUT/selected-arm" ]; then uv run python scripts/stage8.py curve "$(cat "$OUT/selected-arm")"; fi
    ;;
*)
    sed -n '2,20p' "$0"
    exit 2
    ;;
esac
