#!/usr/bin/env bash
# Train the scale-sharing test (control, C, B and A) in parallel, then evaluate every snapshot.
#
# Usage:
#   scripts/run_scale_sharing.sh                       # 300k games each, 1,000-game evaluations
#   GAMES=3000 EVAL_GAMES=20 scripts/run_scale_sharing.sh   # quick smoke test, as smoke-* runs
#
# Each run records to experiments/<run>/ and logs to experiments/<run>/train.log. Runs with fewer
# than 300k games are named smoke-* (and gitignored) so they can't overwrite the real ones.
# Watch progress with: make watch-ntuple ARGS="--run sharing-4x6-control sharing-4x6-late-starts sharing-4x6-downgrade-scale sharing-4x6-downgrade"
set -euo pipefail
cd "$(dirname "$0")/.."

GAMES=${GAMES:-300000}
EVAL_GAMES=${EVAL_GAMES:-1000}
SNAPSHOTS=("$((GAMES / 3))" "$((2 * GAMES / 3))" "$GAMES")
PREFIX=sharing-4x6
if [ "$GAMES" -lt 300000 ]; then
    PREFIX=smoke-sharing-4x6
fi
RUNS=("$PREFIX-control" "$PREFIX-late-starts" "$PREFIX-downgrade-scale" "$PREFIX-downgrade")

# Launching truncates each run's log before training can check its lock, so refuse up front
for run in "${RUNS[@]}"; do
    if pgrep -f -- "--run $run --games" > /dev/null; then
        echo "$run is already training; stop it first or wait for it to finish" >&2
        exit 1
    fi
done

export NTUPLE_NETWORK=4x6 NTUPLE_LEARNING_RATE=0.08 NTUPLE_TEMPORAL_COHERENCE=true

train() {
    local run=$1
    shift
    mkdir -p "experiments/$run"
    env "$@" uv run python -u scripts/train_ntuple.py --run "$run" --games "$GAMES" --snapshot-at "${SNAPSHOTS[@]}" \
        > "experiments/$run/train.log" 2>&1 &
}

train "${RUNS[0]}"
train "${RUNS[1]}" NTUPLE_LATE_START_FRACTION=0.2 NTUPLE_LATE_START_TILE=8192
train "${RUNS[2]}" NTUPLE_DOWNGRADE_TILE=16384 NTUPLE_SCALE_FEATURE=true
train "${RUNS[3]}" NTUPLE_DOWNGRADE_TILE=16384
echo "Training 4 runs to $GAMES games; logs in experiments/$PREFIX-*/train.log"
wait

for run in "${RUNS[@]}"; do
    for games in "${SNAPSHOTS[@]}"; do
        uv run python scripts/evaluate.py "checkpoints/ntuple/$run/games_$games.npz" --agent ntuple -n "$EVAL_GAMES" --experiment "experiments/$run" \
            | tr '\r' '\n' | grep -E "^N-tuple"
    done
done
