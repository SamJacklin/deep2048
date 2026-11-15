#!/usr/bin/env bash
# Stage 8A's fourth arm: the fork's shared 850k checkpoint continued under proportional normalisation with the
# per-weight normaliser as a running mean (decay NORMALISER_DECAY, default 0.01) in place of the published running
# maximum, so that meta-steps are a good part of θ and step sizes can adapt weight by weight (autotidbd.py). Compared
# with the proportional arm, which differs only in the normaliser, and with the fork's control and local arms.
#
#   caffeinate -i bash scripts/run_stage8_tidbd_mean.sh
#
# Rerunning resumes from the arm's newest checkpoint. Log: experiments/stage8-tidbd-repair/logs/mean.log.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=experiments/stage8-tidbd-repair
ARM=checkpoints/ntuple/stage8-tidbd-repair/mean
mkdir -p "$OUT/logs"
trap 'echo "[$(date -u +%FT%TZ)] mean exited with code $?"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
exec > >(tee -a "$OUT/logs/mean.log") 2>&1
echo "[$(date -u +%FT%TZ)] proportional (caffeinated: $(pmset -g assertions 2>/dev/null | grep -c 'caffeinate' || true) assertions, commit $(git rev-parse --short HEAD))"

export NTUPLE_NETWORK=8x6 NTUPLE_DOWNGRADE_TILE=16384 NTUPLE_LATE_START_FRACTION=0.2 NTUPLE_LATE_START_TILE=8192 NTUPLE_SEED=42
export NTUPLE_THREADS_AFTER=10000 NTUPLE_THREADS=${THREADS:-8}
export NTUPLE_OPTIMIZER=autotidbd NTUPLE_TIDBD_INITIAL_ALPHA=0.01 NTUPLE_TIDBD_META_RATE=0.01 NTUPLE_TIDBD_TAU=10000 NTUPLE_TIDBD_NORMALISATION=proportional NTUPLE_TIDBD_NORMALISER_DECAY=${NORMALISER_DECAY:-0.01}

read -r start sha < "$OUT/start"
[ "$(shasum -a 256 "$start" | cut -d' ' -f1)" = "$sha" ] || { echo "$start has changed since it was named as the start" >&2; exit 1; }
for stop in 950000 1050000 1150000 1250000 1350000; do
    if [ ! -f "$ARM/games_$stop.npz" ]; then
        [ "$(df -g . | awk 'NR == 2 { print $4 }')" -ge 8 ] || { echo "less than 8 GB of disk free" >&2; exit 1; }
        if [ -f "$ARM/latest.npz" ]; then
            begin=(--resume "$ARM/latest.npz")
        else
            newest=$({ ls "$ARM"/games_*.npz 2>/dev/null || true; } | sort -t_ -k2 -n | tail -1)
            if [ -n "$newest" ]; then begin=(--resume "$newest"); else begin=(--resume "$start" --fork-normalisation); fi
        fi
        uv run python -u scripts/train_ntuple.py --run stage8-tidbd-repair/mean "${begin[@]}" --games "$stop" --snapshot-at "$stop"
        rm -f "$ARM/latest.npz"
    fi
    uv run python scripts/stage8.py evaluate "$ARM/games_$stop.npz" --name "repair-mean-$stop" --depth 1 --split dev | tr '\r' '\n'
done
uv run python scripts/stage8.py evaluate "$ARM/games_1350000.npz" --name repair-mean-1350000 --depth 3 --split dev | tr '\r' '\n'
uv run python scripts/stage8_repair.py compare
