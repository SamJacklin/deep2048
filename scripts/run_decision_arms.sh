#!/usr/bin/env bash
# The decision arms for the final training run: branch every arm from one checkpoint, train each a further
# 500k games, then compare them game for game with the round's first arm.
#
# Usage:
#   caffeinate -i scripts/run_decision_arms.sh                         # round 1: 4 arms, 1.5M → 2M games, about 4–5 hours on the M4
#   ROUND=2 caffeinate -i scripts/run_decision_arms.sh                 # round 2: 3 arms, 2M → 2.5M games from round 1's tc-reset
#   GAMES=1503000 EVAL_GAMES=20 scripts/run_decision_arms.sh           # smoke test, as smoke-* runs (any GAMES short of the round's)
#
# Every arm is the long run's setup (8×6, temporal coherence at 0.02, downgrading from 16384,
# late-game starts from 8192, 8 threads) except for the one thing it tests.
# Round 1, from the long run's 1.5M-game network:
#   control       unchanged
#   tc-reset      temporal coherence sums reset at the branch point, so every weight starts back at the full rate
#   constant      a constant rate of 0.0016 per weight (Guei's and 2048-rl's 0.1 per board) instead of temporal coherence
#   late-share    half of all games start from late-game positions instead of a fifth, for more experience on the way to 32768
#                 (starts from 16384 positions aren't possible: doubled, they could need two 32768s merged, which the bitboard can't do)
# Round 2, from round 1's tc-reset at 2M games, to see whether resets keep paying:
#   continue      unchanged, so the first reset's cycle carries on
#   reset-again   the sums reset a second time
#   gentle-reset  the sums scaled by 0.1: rates stay as they were but adapt to new errors sooner, avoiding the full reset's dip
# Arms train one after another, since each uses most of the machine. Rerunning resumes an interrupted
# arm from its own latest checkpoint and skips finished ones (evaluations come from the cache).
# Each snapshot is evaluated greedily, and the final one also at exact depth 3, on EVAL_GAMES seeded games.
set -euo pipefail
cd "$(dirname "$0")/.."

case ${ROUND:-1} in
    1) BASE=checkpoints/ntuple/long-8x6/games_1500000.npz FULL_GAMES=2000000 PREFIX=decision-8x6 ARMS=(control tc-reset constant late-share) ;;
    2) BASE=checkpoints/ntuple/decision-8x6-tc-reset/games_2000000.npz FULL_GAMES=2500000 PREFIX=decision2-8x6 ARMS=(continue reset-again gentle-reset) ;;
    *) echo "ROUND must be 1 or 2" >&2 && exit 1 ;;
esac
GAMES=${GAMES:-$FULL_GAMES}
EVAL_GAMES=${EVAL_GAMES:-1000}
if [ "$GAMES" -lt "$FULL_GAMES" ]; then
    PREFIX=smoke-$PREFIX
fi
START=$(uv run python -c "from deep_2048.agents.ntuple import NTupleNetwork; print(NTupleNetwork.load('$BASE')[1].games_played)")
SNAPSHOTS=("$(((START + GAMES) / 2))" "$GAMES")

arm_settings() {
    case $1 in
        constant) echo NTUPLE_TEMPORAL_COHERENCE=false NTUPLE_LEARNING_RATE=0.0016 ;;
        late-share) echo NTUPLE_LATE_START_FRACTION=0.5 ;;
    esac
}
# Applied only when an arm branches from BASE, never when it resumes from its own checkpoint
arm_flags() {
    case $1 in
        tc-reset | reset-again) echo --reset-coherence ;;
        gentle-reset) echo --coherence-scale 0.1 ;;
    esac
}

if pgrep -f -- "--run $PREFIX-" > /dev/null; then
    echo "a $PREFIX run is already training; stop it first or wait for it to finish" >&2
    exit 1
fi

export NTUPLE_NETWORK=8x6 NTUPLE_LEARNING_RATE=0.02 NTUPLE_TEMPORAL_COHERENCE=true NTUPLE_THREADS=${THREADS:-8}
export NTUPLE_DOWNGRADE_TILE=16384 NTUPLE_LATE_START_FRACTION=0.2 NTUPLE_LATE_START_TILE=8192

for arm in "${ARMS[@]}"; do
    run=$PREFIX-$arm
    checkpoints=checkpoints/ntuple/$run
    if [ ! -f "$checkpoints/games_$GAMES.npz" ]; then
        mkdir -p "experiments/$run"
        # A fresh arm branches from BASE; an interrupted one resumes from its own latest checkpoint, without resetting again
        if [ -f "$checkpoints/latest.npz" ]; then
            start=(--resume "$checkpoints/latest.npz")
        else
            # shellcheck disable=SC2207  # the flags are deliberately split into separate arguments
            start=(--resume "$BASE" $(arm_flags "$arm"))
        fi
        echo "Training $run from ${start[1]}"
        # shellcheck disable=SC2046  # the arm's settings are deliberately split into separate variables
        env $(arm_settings "$arm") uv run python -u scripts/train_ntuple.py --run "$run" "${start[@]}" --games "$GAMES" --snapshot-at "${SNAPSHOTS[@]}" \
            >> "experiments/$run/train.log" 2>&1
        # latest.npz now duplicates the final snapshot, and at 1.5 GB per arm the disk space matters
        rm -f "$checkpoints/latest.npz"
    fi
    for games in "${SNAPSHOTS[@]}"; do
        uv run python scripts/evaluate.py "$checkpoints/games_$games.npz" --agent ntuple -n "$EVAL_GAMES" --experiment "experiments/$run" \
            | tr '\r' '\n' | grep -E "^N-tuple"
    done
    uv run python scripts/evaluate.py "$checkpoints/games_$GAMES.npz" --agent ntuple_search --depth 3 -n "$EVAL_GAMES" --experiment "experiments/$run" \
        | tr '\r' '\n' | grep -E "^N-tuple"
done

uv run python - "$PREFIX" "${ARMS[@]}" <<'EOF'
import json
import sys
from pathlib import Path

import numpy as np

prefix, arms = sys.argv[1], sys.argv[2:]


def evaluations(arm: str) -> list[dict]:
    return json.loads(Path(f"experiments/{prefix}-{arm}/evaluations.json").read_text())


reference = arms[0]
control = evaluations(reference)
print(f"\n{'Arm':<14}{'Evaluation':<34}{'Score':>10}{'vs ' + reference:>20}{'≥32768':>9}")
for arm in arms:
    for record, baseline in zip(evaluations(arm), control):
        scores, tiles = np.array(record["result"]["scores"]), np.array(record["result"]["max_tiles"])
        difference = scores - np.array(baseline["result"]["scores"])
        interval = 1.96 * difference.std(ddof=1) / np.sqrt(len(difference)) if arm != reference else 0
        label = record["key"]["label"]
        what = label[label.index("(") + 1 : label.index(" games")] + " games" + (" + depth 3" if "depth=3" in label else ", greedy")
        paired = f"{difference.mean():+,.0f} ± {interval:,.0f}" if arm != reference else "—"
        print(f"{arm:<14}{what:<34}{scores.mean():>10,.0f}{paired:>20}{(tiles >= 32768).mean() * 100:>8.1f}%")
EOF
