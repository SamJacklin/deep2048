#!/usr/bin/env bash
# Continue a finished n-tuple run to a higher game count, with the long run's recipe and no reset.
#
# Usage:
#   caffeinate -i bash scripts/continue_run.sh optimistic-8x6-1000 2500000
#
# Resumes checkpoints/ntuple/RUN/latest.npz and trains on until RUN has played GAMES games in total, saving a snapshot
# at every 500,000 games passed on the way and at the end, then evaluates each new snapshot greedily on the usual
# seeds. The history and log carry on in experiments/RUN, so scripts/watch_against_best.py keeps drawing the run.
# Training runs in the foreground: Ctrl-C stops the trainer, which saves a checkpoint first, and the same command
# picks it up again.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN=${1:?usage: continue_run.sh RUN GAMES}
GAMES=${2:?usage: continue_run.sh RUN GAMES}
EVAL_GAMES=${EVAL_GAMES:-1000}
PYTHON=${PYTHON:-uv run python}
CHECKPOINTS=checkpoints/ntuple/$RUN

if [ ! -f "$CHECKPOINTS/latest.npz" ]; then
    echo "$CHECKPOINTS/latest.npz is missing: nothing to continue" >&2
    exit 1
fi
if pgrep -f -- "--run $RUN --" > /dev/null; then
    echo "$RUN is already training; stop it first or wait for it to finish" >&2
    exit 1
fi

# Snapshots: each 500,000 above the largest one already saved, and the end
saved=0
for file in "$CHECKPOINTS"/games_*.npz; do
    [ -f "$file" ] || continue
    games=${file##*games_}
    games=${games%.npz}
    if [ "$games" -gt "$saved" ]; then saved=$games; fi
done
if [ "$GAMES" -le "$saved" ]; then
    echo "$RUN already has a snapshot at $saved games" >&2
    exit 1
fi
SNAPSHOTS=()
for ((games = 500000; games < GAMES; games += 500000)); do
    if [ "$games" -gt "$saved" ]; then SNAPSHOTS+=("$games"); fi
done
SNAPSHOTS+=("$GAMES")

export NTUPLE_NETWORK=${NETWORK:-8x6} NTUPLE_LEARNING_RATE=${LEARNING_RATE:-0.02} NTUPLE_TEMPORAL_COHERENCE=true NTUPLE_THREADS=${THREADS:-8}
export NTUPLE_DOWNGRADE_TILE=${DOWNGRADE_TILE:-16384} NTUPLE_LATE_START_FRACTION=0.2 NTUPLE_LATE_START_TILE=8192

echo "Continuing $RUN to $GAMES games, snapshots at ${SNAPSHOTS[*]}"
$PYTHON -u scripts/train_ntuple.py --run "$RUN" --resume "$CHECKPOINTS/latest.npz" --games "$GAMES" --snapshot-at "${SNAPSHOTS[@]}" 2>&1 | tee -a "experiments/$RUN/train.log"

for games in "${SNAPSHOTS[@]}"; do
    if [ -f "$CHECKPOINTS/games_$games.npz" ]; then
        $PYTHON scripts/evaluate.py "$CHECKPOINTS/games_$games.npz" --agent ntuple -n "$EVAL_GAMES" --experiment "experiments/$RUN" | tr '\r' '\n' | grep -E "^N-tuple" || true
    fi
done
