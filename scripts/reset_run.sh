#!/usr/bin/env bash
# Reset a finished n-tuple run's temporal coherence sums and train it on, as round 1 did to the long run at 1.5M.
#
# Usage:
#   caffeinate -i bash scripts/reset_run.sh optimistic-8x6-500 2000000
#
# The weights are kept and every weight's learning-rate sums start again, which is what took the long run from
# about 300k to 316-320k greedy between 1.5M and 2M games (experiments/decision-8x6-tc-reset), so a run reset here
# at 1.5M and trained to 2M is the like-for-like comparison with the best system. Scores dip for the first 50,000
# games or so after a reset before they recover.
#
# Resumes checkpoints/ntuple/RUN/latest.npz and trains on until RUN has played GAMES games in total, saving a snapshot
# at every 500,000 games passed on the way and at the end, then evaluates each new snapshot greedily on the usual
# seeds. The history and log carry on in experiments/RUN, so scripts/watch_against_best.py keeps drawing the run.
# Training runs in the foreground: Ctrl-C stops the trainer, which saves a checkpoint first, and the same command
# picks it up again without resetting a second time.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN=${1:?usage: reset_run.sh RUN GAMES}
GAMES=${2:?usage: reset_run.sh RUN GAMES}
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

# The marker is older than any checkpoint saved since the reset, so an interrupted run resumes without resetting again
MARKER=$CHECKPOINTS/reset-to-$GAMES-started
reset=()
if [ -f "$MARKER" ] && [ "$CHECKPOINTS/latest.npz" -nt "$MARKER" ]; then
    echo "Resuming $RUN to $GAMES games (already reset), snapshots at ${SNAPSHOTS[*]}"
else
    echo "Resetting the temporal coherence sums of $RUN and training on to $GAMES games, snapshots at ${SNAPSHOTS[*]}"
    reset=(--reset-coherence)
    touch "$MARKER"
    sleep 1
fi
$PYTHON -u scripts/train_ntuple.py --run "$RUN" --resume "$CHECKPOINTS/latest.npz" ${reset[@]+"${reset[@]}"} --games "$GAMES" --snapshot-at "${SNAPSHOTS[@]}" 2>&1 | tee -a "experiments/$RUN/train.log"

for games in "${SNAPSHOTS[@]}"; do
    if [ -f "$CHECKPOINTS/games_$games.npz" ]; then
        $PYTHON scripts/evaluate.py "$CHECKPOINTS/games_$games.npz" --agent ntuple -n "$EVAL_GAMES" --experiment "experiments/$RUN" | tr '\r' '\n' | grep -E "^N-tuple" || true
    fi
done
