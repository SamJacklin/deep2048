#!/usr/bin/env bash
# The long run: train the strongest configuration for 1.5M games on several threads, then evaluate it
# greedily at every snapshot and with depth-2 and depth-3 search at the end.
#
# Usage:
#   caffeinate -i scripts/run_long_run.sh                     # 8×6, TC 0.02, 8 threads, 1.5M games
#   GAMES=3000 EVAL_GAMES=20 NTUPLE_THREADS_AFTER=1000 scripts/run_long_run.sh   # smoke test, as a smoke-* run
#
# Rerunning the same command after an interruption resumes from the last checkpoint. Ingredients
# can be switched with DOWNGRADE=0/1 and LATE_STARTS=0/1, and the run name with RUN.
set -euo pipefail
cd "$(dirname "$0")/.."

GAMES=${GAMES:-1500000}
EVAL_GAMES=${EVAL_GAMES:-1000}
DOWNGRADE=${DOWNGRADE:-1}
LATE_STARTS=${LATE_STARTS:-0}
RUN=${RUN:-long-8x6}
if [ "$GAMES" -lt 1500000 ]; then
    RUN=smoke-$RUN
    SNAPSHOTS=("$((GAMES / 3))" "$GAMES")
else
    SNAPSHOTS=(100000 250000 500000 1000000 "$GAMES")
fi
CHECKPOINTS=checkpoints/ntuple/$RUN

if pgrep -f -- "--run $RUN --" > /dev/null; then
    echo "$RUN is already training; stop it first or wait for it to finish" >&2
    exit 1
fi

export NTUPLE_NETWORK=8x6 NTUPLE_LEARNING_RATE=0.02 NTUPLE_TEMPORAL_COHERENCE=true NTUPLE_THREADS=${THREADS:-8}
if [ "$DOWNGRADE" = 1 ]; then
    export NTUPLE_DOWNGRADE_TILE=16384
fi
if [ "$LATE_STARTS" = 1 ]; then
    export NTUPLE_LATE_START_FRACTION=0.2 NTUPLE_LATE_START_TILE=8192
fi

if [ ! -f "$CHECKPOINTS/games_$GAMES.npz" ]; then
    mkdir -p "experiments/$RUN"
    resume=()
    if [ -f "$CHECKPOINTS/latest.npz" ]; then
        echo "Resuming $RUN from $CHECKPOINTS/latest.npz"
        resume=(--resume "$CHECKPOINTS/latest.npz")
    else
        : > "experiments/$RUN/train.log"
    fi
    echo "Training $RUN to $GAMES games; log in experiments/$RUN/train.log"
    # The odd expansion keeps an empty array working under set -u in macOS's bash 3.2
    uv run python -u scripts/train_ntuple.py --run "$RUN" ${resume[@]+"${resume[@]}"} --games "$GAMES" --snapshot-at "${SNAPSHOTS[@]}" \
        >> "experiments/$RUN/train.log" 2>&1
fi

for games in "${SNAPSHOTS[@]}"; do
    uv run python scripts/evaluate.py "$CHECKPOINTS/games_$games.npz" --agent ntuple -n "$EVAL_GAMES" --experiment "experiments/$RUN" \
        | tr '\r' '\n' | grep -E "^N-tuple"
done
for depth in 2 3; do
    uv run python scripts/evaluate.py "$CHECKPOINTS/games_$GAMES.npz" --agent ntuple_search --depth "$depth" -n "$EVAL_GAMES" --experiment "experiments/$RUN" \
        | tr '\r' '\n' | grep -E "^N-tuple"
done
