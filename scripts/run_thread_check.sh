#!/usr/bin/env bash
# Check that lock-free multi-threaded training learns as well as single-threaded, and measure its speed.
#
# Usage:
#   scripts/run_thread_check.sh                                # 200k games per run, 1,000-game evaluations
#   GAMES=6000 EVAL_GAMES=20 SPEED_GAMES=500 NTUPLE_THREADS_AFTER=1000 scripts/run_thread_check.sh   # smoke test, as smoke-* runs
#
# 1. Trains the 4×6 network at temporal coherence rate 0.08 to GAMES twice with THREADS threads,
#    to compare with the single-threaded screen-4x6-tc0.08 run and with each other.
# 2. From the first run's final snapshot, trains SPEED_GAMES more at 1, 2, 4 and 8 threads to
#    measure moves per second (written under output/thread-speed/, not committed).
# 3. Evaluates both runs' snapshots into their experiments/ folders.
set -euo pipefail
cd "$(dirname "$0")/.."

GAMES=${GAMES:-200000}
EVAL_GAMES=${EVAL_GAMES:-1000}
SPEED_GAMES=${SPEED_GAMES:-5000}
THREADS=${THREADS:-8}
SNAPSHOTS=("$((GAMES / 2))" "$GAMES")
PREFIX=threads-4x6-t$THREADS
if [ "$GAMES" -lt 200000 ]; then
    PREFIX=smoke-threads-4x6-t$THREADS
fi
RUNS=("$PREFIX-a" "$PREFIX-b")

for run in "${RUNS[@]}"; do
    if pgrep -f -- "--run $run --games" > /dev/null; then
        echo "$run is already training; stop it first or wait for it to finish" >&2
        exit 1
    fi
done

export NTUPLE_NETWORK=4x6 NTUPLE_LEARNING_RATE=0.08 NTUPLE_TEMPORAL_COHERENCE=true

for run in "${RUNS[@]}"; do
    echo "Training $run to $GAMES games on $THREADS threads"
    mkdir -p "experiments/$run"
    NTUPLE_THREADS=$THREADS uv run python -u scripts/train_ntuple.py --run "$run" --games "$GAMES" --snapshot-at "${SNAPSHOTS[@]}" \
        > "experiments/$run/train.log" 2>&1
done

echo "Measuring speed from ${RUNS[0]} at $GAMES games"
mkdir -p output/thread-speed
for threads in 1 2 4 8; do
    out=output/thread-speed/$PREFIX-speed-t$threads
    rm -rf "$out"
    NTUPLE_THREADS=$threads NTUPLE_THREADS_AFTER=0 NTUPLE_LOG_EVERY=$((SPEED_GAMES / 10)) NTUPLE_OUTPUT_DIR=$out NTUPLE_CHECKPOINT_DIR=$out/checkpoints \
        uv run python scripts/train_ntuple.py --resume "checkpoints/ntuple/${RUNS[0]}/games_$GAMES.npz" --games "$((GAMES + SPEED_GAMES))" > "$out.log" 2>&1
    uv run python - "$out/training_history.csv" "$threads" <<'EOF'
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1])))[1:]  # the first batch's moves include those before resuming
games = [int(row["games"]) for row in rows]
moves = [int(row["moves"]) for row in rows]
seconds = sum((b - a) / float(row["games_per_second"]) for a, b, row in zip(games, games[1:], rows[1:]))
print(f"  {sys.argv[2]} threads: {(moves[-1] - moves[0]) / seconds / 1e6:.2f}M moves/s")
EOF
done

for run in "${RUNS[@]}"; do
    for games in "${SNAPSHOTS[@]}"; do
        uv run python scripts/evaluate.py "checkpoints/ntuple/$run/games_$games.npz" --agent ntuple -n "$EVAL_GAMES" --experiment "experiments/$run" \
            | tr '\r' '\n' | grep -E "^N-tuple"
    done
done
