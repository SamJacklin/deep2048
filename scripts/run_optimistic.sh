#!/usr/bin/env bash
# Optimistic start: the long run's recipe from scratch, with every weight starting high and one early reset.
#
# Usage:
#   caffeinate -i scripts/run_optimistic.sh                        # 8×6, TC 0.02, A + C, 8 threads, 1.5M games
#   OPTIMISM=1000 RUN=optimistic-8x6-1000 caffeinate -i scripts/run_optimistic.sh
#   OPTIMISM=100 RESET=0 RUN=optimistic-8x6-100 caffeinate -i scripts/run_optimistic.sh   # a small value needs no reset
#   GAMES=4000 RESET_AT=2000 EVAL_GAMES=20 NTUPLE_THREADS_AFTER=1000 scripts/run_optimistic.sh   # smoke test
#
# Every weight starts at OPTIMISM (5,000 a weight is 320,000 a board on the 8×6, the level Guei et al. used), so
# greedy play is drawn to patterns it hasn't learned yet. A weight's first update from there is a huge negative
# error, which dominates its temporal coherence sums and would hold it at the full rate long after, so the sums are
# reset once, at RESET_AT games, when the first optimism has drained. Everything else is scripts/run_long_run.sh's
# recipe (downgrading from 16384, late-game starts, rate 0.02), and snapshots are evaluated greedily at the same
# game counts on the same seeds, so experiments/long-8x6 is the control. Round 3 added optimism to a trained network
# and found nothing; this is the from-scratch test it left open.
#
# Rerunning the same command after an interruption resumes from the last checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."

GAMES=${GAMES:-1500000}
OPTIMISM=${OPTIMISM:-5000}
RESET_AT=${RESET_AT:-250000}
# RESET=0 trains straight through RESET_AT without resetting the sums: the long run's recipe with only the start changed
RESET=${RESET:-1}
EVAL_GAMES=${EVAL_GAMES:-1000}
RUN=${RUN:-optimistic-8x6}
PYTHON=${PYTHON:-uv run python}
if [ "$GAMES" -lt 1500000 ]; then
    RUN=smoke-$RUN
    SNAPSHOTS=("$RESET_AT" "$GAMES")
else
    SNAPSHOTS=(100000 250000 500000 1000000 "$GAMES")
    case " ${SNAPSHOTS[*]} " in *" $RESET_AT "*) ;; *) SNAPSHOTS+=("$RESET_AT") ;; esac
fi
CHECKPOINTS=checkpoints/ntuple/$RUN
EARLY=() LATER=()
for games in "${SNAPSHOTS[@]}"; do
    if [ "$games" -le "$RESET_AT" ]; then EARLY+=("$games"); else LATER+=("$games"); fi
done

EVALUATED=" "
evaluate_new_snapshots() {
    for games in "${SNAPSHOTS[@]}"; do
        case $EVALUATED in *" $games "*) continue ;; esac
        if [ -f "$CHECKPOINTS/games_$games.npz" ] && $PYTHON scripts/evaluate.py "$CHECKPOINTS/games_$games.npz" --agent ntuple -n "$EVAL_GAMES" \
            --experiment "experiments/$RUN" | tr '\r' '\n' | grep -E "^N-tuple"; then
            EVALUATED="$EVALUATED$games "
        fi
    done
}

# Train in the background with the given arguments, evaluating snapshots as they are saved
train_and_evaluate() {
    $PYTHON -u scripts/train_ntuple.py --run "$RUN" "$@" >> "experiments/$RUN/train.log" 2>&1 &
    training=$!
    while kill -0 "$training" 2> /dev/null; do
        evaluate_new_snapshots
        sleep 30
    done
    wait "$training"
}

if pgrep -f -- "--run $RUN --" > /dev/null; then
    echo "$RUN is already training; stop it first or wait for it to finish" >&2
    exit 1
fi

export NTUPLE_NETWORK=${NETWORK:-8x6} NTUPLE_LEARNING_RATE=${LEARNING_RATE:-0.02} NTUPLE_TEMPORAL_COHERENCE=true NTUPLE_THREADS=${THREADS:-8}
export NTUPLE_DOWNGRADE_TILE=${DOWNGRADE_TILE:-16384} NTUPLE_LATE_START_FRACTION=0.2 NTUPLE_LATE_START_TILE=8192
mkdir -p "experiments/$RUN"

# Phase 1: from optimistic weights to RESET_AT games
if [ ! -f "$CHECKPOINTS/games_$RESET_AT.npz" ]; then
    if [ -f "$CHECKPOINTS/latest.npz" ]; then
        echo "Resuming $RUN from $CHECKPOINTS/latest.npz"
        train_and_evaluate --resume "$CHECKPOINTS/latest.npz" --games "$RESET_AT" --snapshot-at "${EARLY[@]}"
    else
        : > "experiments/$RUN/train.log"
        echo "Training $RUN from weights of $OPTIMISM to $RESET_AT games; log in experiments/$RUN/train.log"
        train_and_evaluate --optimistic-unvisited "$OPTIMISM" --games "$RESET_AT" --snapshot-at "${EARLY[@]}"
    fi
fi

# Phase 2: reset the temporal coherence sums once, then on to GAMES. The marker is older than any checkpoint the
# second phase has saved, so an interrupted second phase resumes without resetting again.
if [ ! -f "$CHECKPOINTS/games_$GAMES.npz" ]; then
    if [ -f "$CHECKPOINTS/reset-started" ] && [ "$CHECKPOINTS/latest.npz" -nt "$CHECKPOINTS/reset-started" ]; then
        echo "Resuming $RUN from $CHECKPOINTS/latest.npz"
        train_and_evaluate --resume "$CHECKPOINTS/latest.npz" --games "$GAMES" --snapshot-at "${LATER[@]}"
    else
        reset=()
        if [ "$RESET" = 1 ]; then
            echo "Resetting the temporal coherence sums at $RESET_AT games and training $RUN on to $GAMES"
            reset=(--reset-coherence)
        else
            echo "Training $RUN on from $RESET_AT games to $GAMES without a reset"
        fi
        touch "$CHECKPOINTS/reset-started"
        sleep 1
        train_and_evaluate --resume "$CHECKPOINTS/games_$RESET_AT.npz" ${reset[@]+"${reset[@]}"} --games "$GAMES" --snapshot-at "${LATER[@]}"
    fi
fi
evaluate_new_snapshots
echo "Done. Compare the greedy evaluations in experiments/$RUN with experiments/long-8x6 at the same game counts."
