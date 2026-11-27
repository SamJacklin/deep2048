#!/usr/bin/env bash
# Stage 8A's further arms: the fork's shared 850k checkpoint continued under a variant of AutoTIDBD, beside
# run_stage8_tidbd_repair.sh's fork, whose control and local arms they're compared with: same start, same recipe, same
# stops and evaluations. An arm is named and set by the environment:
#
#   caffeinate -i bash scripts/run_stage8_tidbd_arm.sh                                    # "proportional": proportional normalisation (autotidbd.py)
#   ARM=theta0.1 META_RATE=0.1 caffeinate -i bash scripts/run_stage8_tidbd_arm.sh         # the same with the meta-rate θ at 0.1 from the fork on
#   ARM=mean NORMALISER_DECAY=0.01 caffeinate -i bash scripts/run_stage8_tidbd_arm.sh     # the same with the running-mean normaliser (stopped at 880k: see the README)
#
# STOPS lists the game counts to train to and evaluate greedily at (default: every 100k from 950k to 1.35M), and
# DEPTH3_AT those to evaluate at exact depth 3 afterwards (default: 1.35M), so an arm can be carried on further:
#   ARM=theta0.1 META_RATE=0.1 STOPS="1450000 1550000 1650000 1750000 1850000" DEPTH3_AT=1850000 caffeinate -i bash scripts/run_stage8_tidbd_arm.sh
#
# NORMALISATION (default proportional), META_RATE (default 0.01, arm F's) and NORMALISER_DECAY (default 0, the published
# normaliser) are the only things an arm changes. Rerunning resumes from the arm's newest checkpoint.
# Log: experiments/stage8-tidbd-repair/logs/<arm>.log.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=experiments/stage8-tidbd-repair
NAME=${ARM:-proportional}
ARM=checkpoints/ntuple/stage8-tidbd-repair/$NAME
mkdir -p "$OUT/logs"
trap 'echo "[$(date -u +%FT%TZ)] $NAME exited with code $?"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
exec > >(tee -a "$OUT/logs/$NAME.log") 2>&1
echo "[$(date -u +%FT%TZ)] $NAME (caffeinated: $(pmset -g assertions 2>/dev/null | grep -c 'caffeinate' || true) assertions, commit $(git rev-parse --short HEAD))"

export NTUPLE_NETWORK=8x6 NTUPLE_DOWNGRADE_TILE=16384 NTUPLE_LATE_START_FRACTION=0.2 NTUPLE_LATE_START_TILE=8192 NTUPLE_SEED=42
export NTUPLE_THREADS_AFTER=10000 NTUPLE_THREADS=${THREADS:-8}
export NTUPLE_OPTIMIZER=autotidbd NTUPLE_TIDBD_INITIAL_ALPHA=0.01 NTUPLE_TIDBD_META_RATE=${META_RATE:-0.01} NTUPLE_TIDBD_TAU=10000
export NTUPLE_TIDBD_NORMALISATION=${NORMALISATION:-proportional} NTUPLE_TIDBD_NORMALISER_DECAY=${NORMALISER_DECAY:-0}

read -r start sha < "$OUT/start"
[ "$(shasum -a 256 "$start" | cut -d' ' -f1)" = "$sha" ] || { echo "$start has changed since it was named as the start" >&2; exit 1; }
for stop in ${STOPS:-950000 1050000 1150000 1250000 1350000}; do
    if [ ! -f "$ARM/games_$stop.npz" ]; then
        [ "$(df -g . | awk 'NR == 2 { print $4 }')" -ge 8 ] || { echo "less than 8 GB of disk free" >&2; exit 1; }
        if [ -f "$ARM/latest.npz" ]; then
            begin=(--resume "$ARM/latest.npz")
        else
            newest=$({ ls "$ARM"/games_*.npz 2>/dev/null || true; } | sort -t_ -k2 -n | tail -1)
            if [ -n "$newest" ]; then begin=(--resume "$newest"); else begin=(--resume "$start" --fork-normalisation); fi
        fi
        uv run python -u scripts/train_ntuple.py --run "stage8-tidbd-repair/$NAME" "${begin[@]}" --games "$stop" --snapshot-at "$stop"
        rm -f "$ARM/latest.npz"
    fi
    uv run python scripts/stage8.py evaluate "$ARM/games_$stop.npz" --name "repair-$NAME-$stop" --depth 1 --split dev | tr '\r' '\n'
done
for stop in ${DEPTH3_AT:-1350000}; do
    uv run python scripts/stage8.py evaluate "$ARM/games_$stop.npz" --name "repair-$NAME-$stop" --depth 3 --split dev | tr '\r' '\n'
done
uv run python scripts/stage8_repair.py compare
