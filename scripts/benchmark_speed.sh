#!/usr/bin/env bash
# Speed benchmark for comparing machines, about 10 minutes. Nothing else should be running.
#
# Usage:
#   scripts/benchmark_speed.sh
#
# Trains the 8x6 network from scratch on 1, 8 and up to 16 threads (the long runs' settings), reporting training speed
# in moves/s over the games after the first 5,000. Then times a greedy and a depth-3 evaluation of the
# single-threaded network. Single-threaded training is fully seeded, so its evaluation scores should match
# another machine's exactly. The benchmark's runs are deleted at the end.
set -euo pipefail
cd "$(dirname "$0")/.."

export NTUPLE_NETWORK=8x6 NTUPLE_LEARNING_RATE=0.02 NTUPLE_TEMPORAL_COHERENCE=true NTUPLE_DOWNGRADE_TILE=16384
export NTUPLE_LATE_START_FRACTION=0.2 NTUPLE_LATE_START_TILE=8192

moves_per_second() {
    uv run python - "$1" <<'EOF'
import csv
import sys

rows = list(csv.DictReader(open(f"experiments/{sys.argv[1]}/training_history.csv")))
moves = seconds = 0.0
for previous, row in zip(rows, rows[1:]):
    if int(previous["games"]) >= 5000:
        games = int(row["games"]) - int(previous["games"])
        moves += int(row["moves"]) - int(previous["moves"])
        seconds += games / float(row["games_per_second"])
print(f"{moves / seconds / 1e6:.2f}M moves/s over {int(rows[-1]['games']) - 5000:,} games")
EOF
}

cpus=$(getconf _NPROCESSORS_ONLN 2> /dev/null || sysctl -n hw.ncpu)
echo "Machine: $(uname -sm), $cpus logical CPUs"
# Training can't use more threads than the machine has CPUs, so the last step is 16 or all of them, whichever is fewer
for threads in 1 8 $((cpus < 16 ? cpus : 16)); do
    games=$([ "$threads" = 1 ] && echo 20000 || echo 40000)
    run=smoke-bench-${threads}t
    rm -rf "checkpoints/ntuple/$run" "experiments/$run"
    NTUPLE_THREADS=$threads NTUPLE_THREADS_AFTER=0 uv run python scripts/train_ntuple.py --run "$run" --games "$games" > /dev/null
    echo "Training, $threads thread(s), $games games from scratch: $(moves_per_second "$run")"
done

checkpoint=checkpoints/ntuple/smoke-bench-1t/latest.npz
for search in "--agent ntuple -n 1000" "--agent ntuple_search --depth 3 -n 100"; do
    start=$SECONDS
    # shellcheck disable=SC2086  # the evaluation's options are deliberately split
    result=$(uv run python scripts/evaluate.py "$checkpoint" $search --rerun | tr '\r' '\n' | grep -E "^N-tuple" | awk '{for (i = 1; i <= NF; i++) if ($i ~ /^[0-9]+$/) {print $i; exit}}')
    echo "Evaluation ($search, all CPUs): average score $result in $((SECONDS - start)) s"
done

rm -rf checkpoints/ntuple/smoke-bench-* experiments/smoke-bench-*
