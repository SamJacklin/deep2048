#!/usr/bin/env bash
# Start a new run from any saved snapshot, leaving the run it came from untouched.
#
# Usage:
#   caffeinate -i bash scripts/branch_run.sh SNAPSHOT NEW_RUN GAMES [reset]
#   caffeinate -i bash scripts/branch_run.sh checkpoints/ntuple/optimistic-8x6-500/games_1500000.npz optimistic-8x6-500-reset 2000000 reset
#
# Trains NEW_RUN from SNAPSHOT until it has played GAMES games in total (the snapshot's games count towards that),
# with the long run's recipe. With `reset` as the last word the temporal coherence sums start again first, as round
# 1 did to the long run at 1.5M. Snapshots are saved at every 500,000 games passed and at the end, and each is
# evaluated greedily on the usual seeds once training finishes. The history starts at the snapshot's game count in
# experiments/NEW_RUN, and scripts/watch_against_best.py draws any run named optimistic-8x6-<start>-<anything>.
#
# Rerunning the same command after an interruption resumes NEW_RUN from its own last checkpoint, without resetting
# again. To stop it: pkill -f -- "--run NEW_RUN --"
set -euo pipefail
cd "$(dirname "$0")/.."

SNAPSHOT=${1:?usage: branch_run.sh SNAPSHOT NEW_RUN GAMES [reset]}
RUN=${2:?usage: branch_run.sh SNAPSHOT NEW_RUN GAMES [reset]}
GAMES=${3:?usage: branch_run.sh SNAPSHOT NEW_RUN GAMES [reset]}
RESET=${4:-}
EVAL_GAMES=${EVAL_GAMES:-1000}
PYTHON=${PYTHON:-uv run python}
CHECKPOINTS=checkpoints/ntuple/$RUN

if pgrep -f -- "--run $RUN --" > /dev/null; then
    echo "$RUN is already training; stop it first or wait for it to finish" >&2
    exit 1
fi

arguments=()
if [ -f "$CHECKPOINTS/latest.npz" ]; then
    echo "Resuming $RUN from its own last checkpoint"
    arguments=(--resume "$CHECKPOINTS/latest.npz")
else
    if [ ! -f "$SNAPSHOT" ]; then
        echo "$SNAPSHOT is missing" >&2
        exit 1
    fi
    arguments=(--resume "$SNAPSHOT")
    if [ "$RESET" = reset ]; then
        echo "Starting $RUN from $SNAPSHOT with the temporal coherence sums reset"
        arguments+=(--reset-coherence)
    else
        echo "Starting $RUN from $SNAPSHOT"
    fi
fi

# Snapshots: each 500,000 above the starting snapshot's games, and the end
start=${SNAPSHOT##*games_}
start=${start%.npz}
case $start in *[!0-9]* | "") start=0 ;; esac
SNAPSHOTS=()
for ((games = 500000; games < GAMES; games += 500000)); do
    if [ "$games" -gt "$start" ] && [ ! -f "$CHECKPOINTS/games_$games.npz" ]; then SNAPSHOTS+=("$games"); fi
done
SNAPSHOTS+=("$GAMES")

export NTUPLE_NETWORK=${NETWORK:-8x6} NTUPLE_LEARNING_RATE=${LEARNING_RATE:-0.02} NTUPLE_TEMPORAL_COHERENCE=true NTUPLE_THREADS=${THREADS:-8}
export NTUPLE_DOWNGRADE_TILE=${DOWNGRADE_TILE:-16384} NTUPLE_LATE_START_FRACTION=0.2 NTUPLE_LATE_START_TILE=8192
mkdir -p "experiments/$RUN"

echo "Training $RUN to $GAMES games, snapshots at ${SNAPSHOTS[*]}"
$PYTHON -u scripts/train_ntuple.py --run "$RUN" "${arguments[@]}" --games "$GAMES" --snapshot-at "${SNAPSHOTS[@]}" 2>&1 | tee -a "experiments/$RUN/train.log"

for games in "${SNAPSHOTS[@]}"; do
    if [ -f "$CHECKPOINTS/games_$games.npz" ]; then
        $PYTHON scripts/evaluate.py "$CHECKPOINTS/games_$games.npz" --agent ntuple -n "$EVAL_GAMES" --experiment "experiments/$RUN" | tr '\r' '\n' | grep -E "^N-tuple" || true
    fi
done
