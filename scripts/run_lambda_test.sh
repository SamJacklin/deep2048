#!/usr/bin/env bash
# Test TD(λ) variants across several seeds on the best fast recipe, then summarise against TD(0).
#
# Usage:
#   scripts/run_lambda_test.sh                                 # 3 arms × seeds 1 2 3, 300k games each
#   GAMES=3000 EVAL_GAMES=20 NTUPLE_THREADS_AFTER=1000 scripts/run_lambda_test.sh   # smoke test, as smoke-* runs
#
# Every run is the 4×6 network with temporal coherence at base rate 0.08, downgrading and late-game
# starts (A + C), on THREADS threads. The TD(0) control is the ingredient test's A + C arm, which has
# exactly these settings, so it isn't retrained.
set -euo pipefail
cd "$(dirname "$0")/.."

GAMES=${GAMES:-300000}
EVAL_GAMES=${EVAL_GAMES:-1000}
THREADS=${THREADS:-8}
read -r -a SEED_LIST <<< "${SEEDS:-1 2 3}"
PREFIX=lambda-4x6
CONTROL=ingredients-4x6-downgrade-late-starts
if [ "$GAMES" -lt 300000 ]; then
    PREFIX=smoke-lambda-4x6
fi

ARMS=(jaskowski long annealed)
arm_settings() {
    case $1 in
        jaskowski) echo NTUPLE_LAMBDA=0.5 NTUPLE_LAMBDA_HORIZON=3 ;;
        long) echo NTUPLE_LAMBDA=0.75 NTUPLE_LAMBDA_HORIZON=6 ;;
        annealed) echo NTUPLE_LAMBDA=0.75 NTUPLE_LAMBDA_END=0 NTUPLE_LAMBDA_HORIZON=6 ;;
    esac
}

if pgrep -f -- "--run $PREFIX-" > /dev/null; then
    echo "a $PREFIX run is already training; stop it first or wait for it to finish" >&2
    exit 1
fi

export NTUPLE_NETWORK=4x6 NTUPLE_LEARNING_RATE=0.08 NTUPLE_TEMPORAL_COHERENCE=true NTUPLE_THREADS=$THREADS
export NTUPLE_DOWNGRADE_TILE=16384 NTUPLE_LATE_START_FRACTION=0.2 NTUPLE_LATE_START_TILE=8192

for seed in "${SEED_LIST[@]}"; do
    for arm in "${ARMS[@]}"; do
        run=$PREFIX-$arm-s$seed
        if [ -f "checkpoints/ntuple/$run/games_$GAMES.npz" ]; then
            echo "$run already trained, skipping"
            continue
        fi
        echo "Training $run"
        mkdir -p "experiments/$run"
        # shellcheck disable=SC2046  # the arm's settings are deliberately split into separate variables
        env NTUPLE_SEED="$seed" $(arm_settings "$arm") uv run python -u scripts/train_ntuple.py --run "$run" --games "$GAMES" --snapshot-at "$GAMES" \
            > "experiments/$run/train.log" 2>&1
        uv run python scripts/evaluate.py "checkpoints/ntuple/$run/games_$GAMES.npz" --agent ntuple -n "$EVAL_GAMES" --experiment "experiments/$run" \
            | tr '\r' '\n' | grep -E "^N-tuple"
    done
done

uv run python - "$PREFIX" "$CONTROL" "$GAMES" "${SEED_LIST[@]}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

prefix, control, games, seeds = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
runs = {"TD(0) (A + C control)": control} if games == 300000 else {}
runs |= {arm: f"{prefix}-{arm}" for arm in ("jaskowski", "long", "annealed")}
means = {}
print(f"\n{'Arm':<24}{'Mean of seeds':>15}{'Spread (sd)':>13}   Per seed")
for name, run in runs.items():
    scores = [np.mean(json.loads(Path(f"experiments/{run}-s{seed}/evaluations.json").read_text())[-1]["result"]["scores"]) for seed in seeds]
    means[name] = np.array(scores)
    print(f"{name:<24}{means[name].mean():>15,.0f}{means[name].std(ddof=1):>13,.0f}   " + ", ".join(f"{s:,.0f}" for s in scores))
if "TD(0) (A + C control)" in means:
    print("\nDifference from TD(0), paired by seed (mean ± sd of the per-seed differences):")
    for name in ("jaskowski", "long", "annealed"):
        differences = means[name] - means["TD(0) (A + C control)"]
        print(f"  {name:<22}{differences.mean():>+10,.0f} ± {differences.std(ddof=1):,.0f}")
PY
