# Stage 7: the anatomy of the late game

A diagnostic, not a training experiment: why does round 1's TC-reset network at 2M games
(`checkpoints/ntuple/decision-8x6-tc-reset/games_2000000.npz`, sha256 `8f117f2b449ddba0c48e8a1d71b1afe4720b7b8d5c168b88ac9fbfe8b0dde513`)
convert so few first 16384s into a 32768? No weights change. Code: `scripts/late_game_anatomy.py`,
`deep_2048/agents/ntuple/anatomy.py`, run by `scripts/run_late_game_anatomy.sh`.

## Data and seeds

- **Positions:** stage 6's dev pool, 5,000 genuine first-16384 boards from distinct greedy games of that network
  (`experiments/late-network/positions.csv`, seeds 30,000,000 on). Stage 6's test pool is never read.
- **Seeds kept apart:** the games that made the positions; Part 1 continuation j uses rng([game seed, j]), so j = 0 and
  1 replay stage 6's control evaluation exactly (checked: `stage6-replay-check.json`); Part 2's trajectories use
  rng([game seed, 9000]); its continuations rng([game seed, move, j, 77]), shared by every policy, so policies are paired.
- **The depth-3 subset** (`depth3-subset.json`) was chosen from the boards alone, before any outcome, stratified by
  where the 16384 sits (corner, edge, interior) and by thirds of empty cells, at least 30 per stratum.

## Part 1: does the entry position matter?

64 greedy continuations from each of the 5,000 positions, and 8 at depth 3 from the 1,000-position subset.
- **Position effect:** a beta-binomial fit to each position's 32768 count gives rho, the share of the variance in single
  outcomes that lies between positions, and the spread of true conversion across positions; a one-way random-effects
  split does the same for score added. Bootstrap intervals resample positions.
- **Features** (13, interpretable: the top tile's place, empty cells, the 2nd and 3rd largest tiles and their distance
  from it, monotonicity, legal moves, equal neighbours, the descending chain from the top tile, tiles of 256+, other
  tiles' sum): L2-penalised logistic regression on the counts, out of fold over 5 folds grouped by source game, against
  a constant predictor, by Brier score and log loss, with a calibration table.
- **Best against worst:** positions grouped by out-of-fold predicted quality, not by their noisy observed rates.

## Part 2: when can failing games be rescued?

- **Pilot:** greedy games (`record_trajectory`) from entry positions in seed order, until 20 made a 32768 (kept up to it)
  and 20 didn't (kept to the end). Points: the first 16384, every 250 moves, the first board with at most one empty
  cell, and for failures 200 and 50 moves before the end. Each records moves since the 16384 and, looking back only,
  moves remaining.
- **Policies** from each point, 8 paired continuations each: A greedy; B depth 3 from here; C depth 3 for 200 moves,
  then greedy; D pruned depth 5 (cutoff 0.01) for 200 moves, then greedy (D against C asks whether deeper search finds
  what depth 3 misses over the same window).
- **Report:** each policy's paired difference from greedy in 32768 conversion and score added, by time since the 16384,
  source outcome, moves remaining and empty cells, with intervals from resampling whole trajectories, and cost.
- **Expansion, by a fixed rule** (`p2-plan`, written down with its reasons before it runs): a (source outcome, time)
  group is expanded if B or C could matter (upper bound +5,000 score or +2 points of conversion) and is still uncertain
  (score interval wider than ±2,000). Expanded groups get 32 continuations for A, B and C, at the pilot's points and at
  20 + 20 new trajectories' points; D only where its edge over C could reach +5,000. Continuations are halved until
  the estimate fits BUDGET_HOURS (default 10).

## Part 3

Not run yet: it waits for Parts 1 and 2 to be reviewed, then targets the situations they pick out.

## Results

Reports land here: `part1.md`, `part2-pilot.md`, `p2-plan.json`, `part2-final.md`, `costs.jsonl`, `manifest.json`.
