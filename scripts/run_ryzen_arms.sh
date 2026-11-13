#!/usr/bin/env bash
# Long from-scratch runs for the Ryzen: this project's recipe against the same with Guei's optimistic recipe on top.
#
# Usage:
#   scripts/run_ryzen_arms.sh recipe       # TC at 0.02 from scratch, its sums reset once at 1.5M games, to TOTAL games
#   scripts/run_ryzen_arms.sh optimism     # every weight starting at 5,000 (320k per board), a fixed rate of 0.0016
#                                          # with TC off to 90% of TOTAL, then TC at 0.02 (from fresh sums) to TOTAL
#   TOTAL=40000 EVAL_EVERY=10000 EVAL_GAMES=20 scripts/run_ryzen_arms.sh recipe    # smoke test, as a smoke-* run
#
# Both arms use the 8x6 network with downgrading from 16384 and late-game starts from 8192 in a fifth of games.
# A snapshot every EVAL_EVERY games is evaluated greedily on EVAL_GAMES seeded games while training goes on, and
# deleted afterwards unless it's a kept point (the reset or switch, half way, and the end), which also get an exact
# depth-3 evaluation at the end. Rerunning the same command after an interruption resumes where it stopped.
set -euo pipefail
cd "$(dirname "$0")/.."

ARM=${1:?usage: scripts/run_ryzen_arms.sh recipe|optimism}
TOTAL=${TOTAL:-10000000}
EVAL_EVERY=${EVAL_EVERY:-250000}
EVAL_GAMES=${EVAL_GAMES:-1000}
RUN=ryzen-$ARM
if [ "$TOTAL" -lt 10000000 ]; then
    RUN=smoke-$RUN
fi
CHECKPOINTS=checkpoints/ntuple/$RUN

export NTUPLE_NETWORK=8x6 NTUPLE_THREADS=${THREADS:-8} NTUPLE_DOWNGRADE_TILE=16384 NTUPLE_LATE_START_FRACTION=0.2 NTUPLE_LATE_START_TILE=8192
case $ARM in
    recipe) BOUNDARY=$((TOTAL * 3 / 20)) ;;  # 1.5M of 10M: where this recipe's temporal coherence stalled
    optimism) BOUNDARY=$((TOTAL * 9 / 10)) ;;  # Guei's best: temporal coherence for the last 10%
    *) echo "ARM must be recipe or optimism" >&2 && exit 1 ;;
esac
KEEP=("$BOUNDARY" "$((TOTAL / 2))" "$TOTAL")

if pgrep -f -- "--run $RUN " > /dev/null; then
    echo "$RUN is already training; stop it first or wait for it to finish" >&2
    exit 1
fi
mkdir -p "experiments/$RUN"

games_played() {
    if [ -f "$CHECKPOINTS/latest.npz" ]; then
        uv run python -c "from deep_2048.agents.ntuple import NTupleNetwork; print(NTupleNetwork.load('$CHECKPOINTS/latest.npz')[1].games_played)"
    else
        echo 0
    fi
}

is_kept() {
    case " ${KEEP[*]} " in *" $1 "*) return 0 ;; *) return 1 ;; esac
}

# Evaluate each snapshot greedily once it's saved (saves are atomic), then delete it unless it's a kept point
EVALUATED=" "
evaluate_new_snapshots() {
    for ((games = EVAL_EVERY; games <= TOTAL; games += EVAL_EVERY)); do
        case $EVALUATED in *" $games "*) continue ;; esac
        if [ -f "$CHECKPOINTS/games_$games.npz" ] \
            && uv run python scripts/evaluate.py "$CHECKPOINTS/games_$games.npz" --agent ntuple -n "$EVAL_GAMES" --experiment "experiments/$RUN" \
                | tr '\r' '\n' | grep -E "^N-tuple"; then
            EVALUATED="$EVALUATED$games "
            is_kept "$games" || rm "$CHECKPOINTS/games_$games.npz"
        fi
    done
}

# Train from the last checkpoint to `end` games with the given settings and flags, evaluating snapshots as they come
train_to() {
    local end=$1 settings=$2
    shift 2
    local start resume=()
    start=$(games_played)
    [ "$start" -ge "$end" ] && return 0
    [ "$start" -gt 0 ] && resume=(--resume "$CHECKPOINTS/latest.npz")
    local snapshots=()
    for ((games = EVAL_EVERY; games <= end; games += EVAL_EVERY)); do
        [ "$games" -gt "$start" ] && snapshots+=("$games")
    done
    for games in "${KEEP[@]}"; do
        if [ "$games" -gt "$start" ] && [ "$games" -le "$end" ]; then snapshots+=("$games"); fi
    done
    echo "Training $RUN from $start to $end games ($settings $*)"
    # shellcheck disable=SC2086  # the settings are deliberately split into separate variables
    env $settings uv run python -u scripts/train_ntuple.py --run "$RUN" ${resume[@]+"${resume[@]}"} --games "$end" --snapshot-at "${snapshots[@]}" "$@" \
        >> "experiments/$RUN/train.log" 2>&1 &
    local training=$!
    while kill -0 "$training" 2> /dev/null; do
        evaluate_new_snapshots
        sleep 60
    done
    wait "$training"
    evaluate_new_snapshots
}

if [ -f "$CHECKPOINTS/games_$TOTAL.npz" ]; then
    echo "$RUN already trained to $TOTAL games"
elif [ "$ARM" = recipe ]; then
    train_to "$BOUNDARY" "NTUPLE_TEMPORAL_COHERENCE=true NTUPLE_LEARNING_RATE=0.02"
    # The reset applies only when the second phase starts from the boundary, not when it resumes later
    if [ "$(games_played)" -eq "$BOUNDARY" ]; then
        train_to "$TOTAL" "NTUPLE_TEMPORAL_COHERENCE=true NTUPLE_LEARNING_RATE=0.02" --reset-coherence
    else
        train_to "$TOTAL" "NTUPLE_TEMPORAL_COHERENCE=true NTUPLE_LEARNING_RATE=0.02"
    fi
else
    # Optimism applies only to a fresh network: every weight starts at 5,000, so a board starts at 320,000
    if [ "$(games_played)" -eq 0 ]; then
        train_to "$BOUNDARY" "NTUPLE_TEMPORAL_COHERENCE=false NTUPLE_LEARNING_RATE=0.0016" --optimistic-unvisited 5000
    else
        train_to "$BOUNDARY" "NTUPLE_TEMPORAL_COHERENCE=false NTUPLE_LEARNING_RATE=0.0016"
    fi
    # The constant-rate phase saves no temporal coherence sums, so the finish starts every weight at the full rate
    train_to "$TOTAL" "NTUPLE_TEMPORAL_COHERENCE=true NTUPLE_LEARNING_RATE=0.02"
fi
# latest.npz duplicates the final snapshot by now
rm -f "$CHECKPOINTS/latest.npz"

for games in "${KEEP[@]}"; do
    uv run python scripts/evaluate.py "$CHECKPOINTS/games_$games.npz" --agent ntuple_search --depth 3 -n "$EVAL_GAMES" --experiment "experiments/$RUN" \
        | tr '\r' '\n' | grep -E "^N-tuple"
done
echo "$RUN finished"
