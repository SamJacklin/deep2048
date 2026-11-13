#!/usr/bin/env bash
# The long run: train the strongest configuration for 1.5M games on several threads, then evaluate it
# greedily at every snapshot and with depth-2 and depth-3 search at the end.
#
# Usage:
#   caffeinate -i scripts/run_long_run.sh                     # 8×6, TC 0.02, A + C, 8 threads, 1.5M games
#   GAMES=3000 EVAL_GAMES=20 NTUPLE_THREADS_AFTER=1000 scripts/run_long_run.sh   # smoke test, as a smoke-* run
#   RUN=long-8x6-dg32768 DOWNGRADE_TILE=32768 EVAL_EVERY=250000 caffeinate -i scripts/run_long_run.sh
#   RUN=long-8x6-redundant NETWORK=8x6-redundant LEARNING_RATE=0.0107 caffeinate -i scripts/run_long_run.sh
#
# Rerunning the same command after an interruption resumes from the last checkpoint. Ingredients
# default to downgrading and late-game starts (the ingredient test's best, A + C); switch them with
# DOWNGRADE=0/1 and LATE_STARTS=0/1, the downgrading threshold with DOWNGRADE_TILE, the network with NETWORK
# (with LEARNING_RATE, the base rate), and the run name with RUN.
# With EVAL_EVERY, a snapshot is also saved every that many games and evaluated greedily while training
# goes on; the ones that aren't usual snapshots are deleted once evaluated, to save disk space.
set -euo pipefail
cd "$(dirname "$0")/.."

GAMES=${GAMES:-1500000}
EVAL_GAMES=${EVAL_GAMES:-1000}
DOWNGRADE=${DOWNGRADE:-1}
DOWNGRADE_TILE=${DOWNGRADE_TILE:-16384}
EVAL_EVERY=${EVAL_EVERY:-0}
LATE_STARTS=${LATE_STARTS:-1}
RUN=${RUN:-long-8x6}
if [ "$GAMES" -lt 1500000 ]; then
    RUN=smoke-$RUN
    SNAPSHOTS=("$((GAMES / 3))" "$GAMES")
else
    SNAPSHOTS=(100000 250000 500000 1000000 "$GAMES")
fi
CHECKPOINTS=checkpoints/ntuple/$RUN
EXTRA_SNAPSHOTS=()
if [ "$EVAL_EVERY" -gt 0 ]; then
    for ((games = EVAL_EVERY; games < GAMES; games += EVAL_EVERY)); do
        case " ${SNAPSHOTS[*]} " in *" $games "*) ;; *) EXTRA_SNAPSHOTS+=("$games") ;; esac
    done
fi

evaluate_greedy() {
    uv run python scripts/evaluate.py "$CHECKPOINTS/games_$1.npz" --agent ntuple -n "$EVAL_GAMES" --experiment "experiments/$RUN" \
        | tr '\r' '\n' | grep -E "^N-tuple"
}

# Evaluate each snapshot greedily once it's saved (saves are atomic), and delete the extra ones afterwards
EVALUATED=" "
evaluate_new_snapshots() {
    for games in "${SNAPSHOTS[@]}" ${EXTRA_SNAPSHOTS[@]+"${EXTRA_SNAPSHOTS[@]}"}; do
        case $EVALUATED in *" $games "*) continue ;; esac
        if [ -f "$CHECKPOINTS/games_$games.npz" ] && evaluate_greedy "$games"; then
            EVALUATED="$EVALUATED$games "
            case " ${EXTRA_SNAPSHOTS[*]-} " in *" $games "*) rm "$CHECKPOINTS/games_$games.npz" ;; esac
        fi
    done
}

if pgrep -f -- "--run $RUN --" > /dev/null; then
    echo "$RUN is already training; stop it first or wait for it to finish" >&2
    exit 1
fi

export NTUPLE_NETWORK=${NETWORK:-8x6} NTUPLE_LEARNING_RATE=${LEARNING_RATE:-0.02} NTUPLE_TEMPORAL_COHERENCE=true NTUPLE_THREADS=${THREADS:-8}
if [ "$DOWNGRADE" = 1 ]; then
    export NTUPLE_DOWNGRADE_TILE=$DOWNGRADE_TILE
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
    uv run python -u scripts/train_ntuple.py --run "$RUN" ${resume[@]+"${resume[@]}"} --games "$GAMES" \
        --snapshot-at "${SNAPSHOTS[@]}" ${EXTRA_SNAPSHOTS[@]+"${EXTRA_SNAPSHOTS[@]}"} >> "experiments/$RUN/train.log" 2>&1 &
    training=$!
    while kill -0 "$training" 2> /dev/null; do
        evaluate_new_snapshots
        sleep 30
    done
    wait "$training"
fi
evaluate_new_snapshots
for depth in 2 3; do
    uv run python scripts/evaluate.py "$CHECKPOINTS/games_$GAMES.npz" --agent ntuple_search --depth "$depth" -n "$EVAL_GAMES" --experiment "experiments/$RUN" \
        | tr '\r' '\n' | grep -E "^N-tuple"
done
