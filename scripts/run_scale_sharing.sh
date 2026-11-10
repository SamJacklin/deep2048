#!/usr/bin/env bash
# Train the scale-sharing test (control, C, B and A) in parallel, then evaluate every snapshot.
#
# Usage:
#   scripts/run_scale_sharing.sh                       # 300k games each, 1,000-game evaluations
#   GAMES=3000 EVAL_GAMES=20 scripts/run_scale_sharing.sh   # quick smoke test
#
# Each run records to experiments/<run>/ and logs to experiments/<run>/train.log.
# Watch progress with: make watch-ntuple ARGS="--run sharing-4x6-control sharing-4x6-late-starts sharing-4x6-downgrade-scale sharing-4x6-downgrade"
set -euo pipefail
cd "$(dirname "$0")/.."

GAMES=${GAMES:-300000}
EVAL_GAMES=${EVAL_GAMES:-1000}
SNAPSHOTS=("$((GAMES / 3))" "$((2 * GAMES / 3))" "$GAMES")

export NTUPLE_NETWORK=4x6 NTUPLE_LEARNING_RATE=0.08 NTUPLE_TEMPORAL_COHERENCE=true

train() {
    local run=$1
    shift
    mkdir -p "experiments/$run"
    env "$@" uv run python -u scripts/train_ntuple.py --run "$run" --games "$GAMES" --snapshot-at "${SNAPSHOTS[@]}" \
        > "experiments/$run/train.log" 2>&1 &
}

train sharing-4x6-control
train sharing-4x6-late-starts NTUPLE_LATE_START_FRACTION=0.2 NTUPLE_LATE_START_TILE=8192
train sharing-4x6-downgrade-scale NTUPLE_DOWNGRADE_TILE=16384 NTUPLE_SCALE_FEATURE=true
train sharing-4x6-downgrade NTUPLE_DOWNGRADE_TILE=16384
echo "Training 4 runs to $GAMES games; logs in experiments/sharing-4x6-*/train.log"
wait

for run in sharing-4x6-control sharing-4x6-late-starts sharing-4x6-downgrade-scale sharing-4x6-downgrade; do
    for games in "${SNAPSHOTS[@]}"; do
        uv run python scripts/evaluate.py "checkpoints/ntuple/$run/games_$games.npz" --agent ntuple -n "$EVAL_GAMES" --experiment "experiments/$run" \
            | tr '\r' '\n' | grep -E "^N-tuple"
    done
done
