#!/usr/bin/env bash
# Measure how much single-threaded training varies between seeds, then evaluate every snapshot.
#
# Usage:
#   scripts/run_seed_check.sh                                # seeds 7 and 123, 200k games each
#   SEEDS="1 2 3" scripts/run_seed_check.sh                  # other seeds
#   GAMES=3000 EVAL_GAMES=20 scripts/run_seed_check.sh       # quick smoke test, as smoke-* runs
#
# Trains the same configuration as screen-4x6-tc0.08 (seed 42): the 4×6 network with temporal
# coherence at base rate 0.08, single-threaded, one run per seed in parallel.
set -euo pipefail
cd "$(dirname "$0")/.."

GAMES=${GAMES:-200000}
EVAL_GAMES=${EVAL_GAMES:-1000}
read -r -a SEED_LIST <<< "${SEEDS:-7 123}"
SNAPSHOTS=("$((GAMES / 2))" "$GAMES")
PREFIX=seed-4x6-tc0.08
if [ "$GAMES" -lt 200000 ]; then
    PREFIX=smoke-seed-4x6-tc0.08
fi

for seed in "${SEED_LIST[@]}"; do
    if pgrep -f -- "--run $PREFIX-s$seed --games" > /dev/null; then
        echo "$PREFIX-s$seed is already training; stop it first or wait for it to finish" >&2
        exit 1
    fi
done

export NTUPLE_NETWORK=4x6 NTUPLE_LEARNING_RATE=0.08 NTUPLE_TEMPORAL_COHERENCE=true

for seed in "${SEED_LIST[@]}"; do
    run=$PREFIX-s$seed
    mkdir -p "experiments/$run"
    NTUPLE_SEED=$seed uv run python -u scripts/train_ntuple.py --run "$run" --games "$GAMES" --snapshot-at "${SNAPSHOTS[@]}" \
        > "experiments/$run/train.log" 2>&1 &
done
echo "Training seeds ${SEED_LIST[*]} to $GAMES games; logs in experiments/$PREFIX-s*/train.log"
wait

for seed in "${SEED_LIST[@]}"; do
    run=$PREFIX-s$seed
    for games in "${SNAPSHOTS[@]}"; do
        uv run python scripts/evaluate.py "checkpoints/ntuple/$run/games_$games.npz" --agent ntuple -n "$EVAL_GAMES" --experiment "experiments/$run" \
            | tr '\r' '\n' | grep -E "^N-tuple"
    done
done
