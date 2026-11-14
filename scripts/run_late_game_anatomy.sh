#!/usr/bin/env bash
# Stage 7 on the Ryzen: Parts 1 and 2, every step resuming where it stopped. Part 3 isn't run here (it waits for review).
# Usage: scripts/run_late_game_anatomy.sh            (SMOKE=1 for a quick end-to-end check)
set -euo pipefail
cd "$(dirname "$0")/.."
step() {
  echo "$(date '+%F %T') starting $*"
  uv run python -u scripts/late_game_anatomy.py "$@"
  echo "$(date '+%F %T') finished $*"
}
free_gb=$(df -Pk . | awk 'NR==2 {print int($4 / 1048576)}')
if [ "$free_gb" -lt 5 ]; then echo "Only ${free_gb} GB free: need 5"; exit 1; fi
step manifest
step p1-greedy
step p1-depth3
step p1-report
step p2-trajectories --phase pilot
step p2-run --phase pilot
step p2-report --phase pilot
step p2-plan
step p2-trajectories --phase expansion
step p2-run --phase expansion
step p2-report --phase final
echo "$(date '+%F %T') stage 7 finished"
