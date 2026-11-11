#!/usr/bin/env bash
# Test the scale-sharing ingredients across several seeds, then summarise each arm's mean and spread.
#
# Usage:
#   scripts/run_ingredient_test.sh                             # 5 arms × seeds 1 2 3, 300k games each
#   GAMES=3000 EVAL_GAMES=20 NTUPLE_THREADS_AFTER=1000 scripts/run_ingredient_test.sh   # smoke test, as smoke-* runs
#
# Every run is the 4×6 network with temporal coherence at base rate 0.08 on THREADS threads,
# differing only in the arm's features and the seed. Runs train one after another, since each
# uses most of the machine, and each final snapshot is evaluated on EVAL_GAMES seeded games.
set -euo pipefail
cd "$(dirname "$0")/.."

GAMES=${GAMES:-300000}
EVAL_GAMES=${EVAL_GAMES:-1000}
THREADS=${THREADS:-8}
read -r -a SEED_LIST <<< "${SEEDS:-1 2 3}"
PREFIX=ingredients-4x6
if [ "$GAMES" -lt 300000 ]; then
    PREFIX=smoke-ingredients-4x6
fi

ARMS=(control downgrade late-starts downgrade-late-starts downgrade-scale-when-downgraded)
arm_settings() {
    case $1 in
        control) ;;
        downgrade) echo NTUPLE_DOWNGRADE_TILE=16384 ;;
        late-starts) echo NTUPLE_LATE_START_FRACTION=0.2 NTUPLE_LATE_START_TILE=8192 ;;
        downgrade-late-starts) echo NTUPLE_DOWNGRADE_TILE=16384 NTUPLE_LATE_START_FRACTION=0.2 NTUPLE_LATE_START_TILE=8192 ;;
        downgrade-scale-when-downgraded) echo NTUPLE_DOWNGRADE_TILE=16384 NTUPLE_SCALE_FEATURE=true NTUPLE_SCALE_WHEN_DOWNGRADED=true ;;
    esac
}

if pgrep -f -- "--run $PREFIX-" > /dev/null; then
    echo "an $PREFIX run is already training; stop it first or wait for it to finish" >&2
    exit 1
fi

export NTUPLE_NETWORK=4x6 NTUPLE_LEARNING_RATE=0.08 NTUPLE_TEMPORAL_COHERENCE=true NTUPLE_THREADS=$THREADS

for seed in "${SEED_LIST[@]}"; do
    for arm in "${ARMS[@]}"; do
        run=$PREFIX-$arm-s$seed
        if [ -f "experiments/$run/evaluations.json" ]; then
            echo "$run already trained and evaluated, skipping"
            continue
        fi
        if [ ! -f "checkpoints/ntuple/$run/games_$GAMES.npz" ]; then
            echo "Training $run"
            mkdir -p "experiments/$run"
            # shellcheck disable=SC2046  # the arm's settings are deliberately split into separate variables
            env NTUPLE_SEED="$seed" $(arm_settings "$arm") uv run python -u scripts/train_ntuple.py --run "$run" --games "$GAMES" --snapshot-at "$GAMES" \
                > "experiments/$run/train.log" 2>&1
        fi
        uv run python scripts/evaluate.py "checkpoints/ntuple/$run/games_$GAMES.npz" --agent ntuple -n "$EVAL_GAMES" --experiment "experiments/$run" \
            | tr '\r' '\n' | grep -E "^N-tuple"
    done
done

uv run python - "$PREFIX" "${SEED_LIST[@]}" <<'EOF'
import json
import sys
from pathlib import Path

import numpy as np

prefix, seeds = sys.argv[1], sys.argv[2:]
arms = ["control", "downgrade", "late-starts", "downgrade-late-starts", "downgrade-scale-when-downgraded"]
means = {}
print(f"\n{'Arm':<34}{'Mean of seeds':>15}{'Spread (sd)':>13}   Per seed")
for arm in arms:
    scores = []
    for seed in seeds:
        records = json.loads(Path(f"experiments/{prefix}-{arm}-s{seed}/evaluations.json").read_text())
        scores.append(np.mean(records[-1]["result"]["scores"]))
    means[arm] = np.array(scores)
    print(f"{arm:<34}{means[arm].mean():>15,.0f}{means[arm].std(ddof=1):>13,.0f}   " + ", ".join(f"{s:,.0f}" for s in scores))
print("\nDifference from control, paired by seed (mean ± sd of the per-seed differences):")
for arm in arms[1:]:
    differences = means[arm] - means["control"]
    print(f"  {arm:<32}{differences.mean():>+10,.0f} ± {differences.std(ddof=1):,.0f}")
EOF
