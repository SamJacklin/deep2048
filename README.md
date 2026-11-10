# Deep 2048

Reinforcement-learning agents that taught themselves to play 2048, benchmarked honestly against a classical search algorithm: a deep Q-network that loses to it, and an n-tuple network that beats it.

[**▶ Watch the trained agents play**](https://samjacklin.com/deep-2048)

The DQN is built with PyTorch and trained for ~1M steps on Apple Silicon. The n-tuple network beats the classical search after a minute of training on one CPU core, and its strongest version trains in under an hour. The trained agents can be watched in an interactive replay viewer (React + Vite, deployed to S3/CloudFront via GitHub Actions).

## Results

1,000 games per agent, seeded identically, with 95% confidence intervals:

| Agent | Avg score | Avg max tile | ≥1024 | ≥2048 | ≥8192 | ≥16384 |
|---|---:|---:|---:|---:|---:|---:|
| DQN (best checkpoint) | 5,689 ± 170 | 426 | 6.6% | 0% | 0% | 0% |
| Heuristic expectimax (depth 2) | 19,981 ± 597 | 1,366 | 86.2% | 37.2% | 0% | 0% |
| N-tuple TD (1.5M games, greedy) | 128,208 ± 2,640 | 7,071 | 99.8% | 98.6% | 74.2% | 0.7% |
| N-tuple TD + expectimax (depth 3) | **202,291 ± 3,457** | **11,637** | **100%** | **100%** | **98.9%** | **42.6%** |

Every result, with its training curves, settings and the commit that produced it, is recorded in [`experiments/`](experiments/README.md).

### The honest takeaway

The hand-crafted classical search beats the deep learning model by 3×. That result stays published on purpose.

The DQN was the far more interesting system to build — it learns corner strategy and tile-chaining from nothing but board states and rewards. But a ~130-line heuristic with expectimax search is the better *product* than the DQN: stronger, faster to build, fully explainable, and with no training cost. Knowing which of those two things a problem actually needs is the point of doing the comparison properly.

Learning itself wasn't the problem. An n-tuple network trained by temporal-difference learning (Szubert & Jaśkowski, 2014) already beats the search after 100,000 training games, which take under a minute. It's a large lookup table rather than a deep network, so each update takes nanoseconds. Given the same experience as the DQN it scores about 22% higher, and it can afford over 1,000 times more. Trained for 1.5M games with temporal coherence learning, it averages 128k playing greedily. Putting a depth-3 expectimax search on top, the same idea as the classical baseline, lifts it to 202k, reaching 16384 in 43% of games.

## How it works

### Game engine (`deep_2048/game/`)

Gym-style environment with seeded RNG for reproducibility. Only *move left* is implemented directly; the other three directions are derived by flip/transpose transformations. Rewards follow standard 2048 scoring (sum of merged tile values); new tiles spawn 2 (90%) or 4 (10%).

### State encoding

Boards are one-hot encoded over tile exponents into a `(16, 4, 4)` tensor — tile values are categorical to the network, not ordinal, so each power of two gets its own channel.

### The network (`deep_2048/agents/dqn/network.py`)

A CNN with three parallel paths matched to the game's structure:

- **3×3 convs** (×3 layers) — local tile interactions, effectively global at 4×4
- **1×4 row kernels** — full-row structure for LEFT/RIGHT merges
- **4×1 column kernels** — full-column structure for UP/DOWN merges

concatenated into a **dueling head** (`Q = V + (A − mean(A))`), so state value and per-action advantage are learned separately.

### Training (`deep_2048/agents/dqn/trainer.py`)

A Rainbow-lite DQN:

- **Double DQN** — online network selects the next action, target network evaluates it (counters Q overestimation)
- **Prioritized experience replay** — hand-rolled sum tree, O(log n) proportional sampling, importance-sampling weights with β annealed 0.4 → 1.0
- **N-step returns** (n = 5) — faster reward propagation through long episodes
- **8-fold symmetry augmentation** — random rotation/flip applied to sampled transitions, with actions remapped to match
- **Reward shaping** — `log1p` compression of merge scores plus first-time milestone bonuses per new max tile
- **Invalid-action masking**, gradient clipping, cosine LR decay, linear ε decay 1.0 → 0.01

All hyperparameters live in a pydantic-settings config overridable via `.env` — the same code runs a seconds-long smoke test or the full 1M-step run (see `.env.example`).

### The classical baseline (`deep_2048/agents/heuristic/`)

Depth-2 **expectimax** (max over player moves, expectation over the random tile spawns) with a four-signal board evaluator: monotonicity, merge potential, empty cells, and corner anchoring — all in log2 space. Both run in numba on the bitboard engine, with the evaluator's row signals precomputed for every possible row. The `evaluation.py` docstring records the design iterations, including a positional bonus that was removed because it actively penalised merging.

### The hybrid (`deep_2048/agents/dqn/search.py`)

Expectimax using the DQN's max-Q as the leaf evaluator. All leaf boards for a root action are evaluated in a single batched forward pass. Depth 1 provably adds nothing (the Q-values already contain the one-step spawn expectation from n-step training targets — see the module docstring); depth 2 genuinely extends the planning horizon. Available via `scripts/play.py --agent dqn_search` and `scripts/evaluate.py --agent dqn_search`.

### The n-tuple network (`deep_2048/agents/ntuple/`)

Four tuples (two straight lines of 4 cells and two 2×3 rectangles) are applied to all 8 rotations and reflections of the board. Each placement indexes a learned weight, and a board's value is the sum of the 32 weights it selects, out of 33.7M in total. The agent picks the move with the highest reward plus value of the board it leaves behind, before the new tile spawns (the *afterstate*). Training is TD(0) on afterstates: after each move, the previous afterstate's value moves towards the next move's reward plus the next afterstate's value.

Everything runs in numba on a bitboard: the whole board packed into one 64-bit integer, with moves done as row lookups. Training runs at about 3M moves per second, learning included. Tests check the bitboard against the reference game engine on every possible row.

Training can also use temporal coherence (Beal & Smith, 1999), which scales each weight's learning rate by how consistently its errors point the same way, or a learning rate that decays over the run. At play time, `ntuple_search` adds expectimax: each move is worth its reward plus the expected value over every tile that could spawn, down to a fixed depth, where the network values the afterstate. Depth 1 is the greedy agent.

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # full training config; a fast smoke-test block is included

make train-dqn         # train from scratch (~hours on Apple Silicon / CUDA)
make resume-dqn CHECKPOINT=checkpoints/dqn/step_500045.pt
make train-ntuple      # 100k games from scratch (~1 minute on one CPU core)
make resume-ntuple ARGS="--games 200000"
make evaluate          # every agent, 1,000 games each, across all CPU cores
make play ARGS="--agent heuristic --render"
make plot              # training curves from output/dqn/training_history.csv
make test
```

The n-tuple agents load `checkpoints/ntuple/best.npz` by default. To rebuild the benchmarked network, run the two training sessions in [`experiments/ntuple-td-1m`](experiments/ntuple-td-1m/NOTES.md) and [`experiments/ntuple-1m-cont-tc`](experiments/ntuple-1m-cont-tc/NOTES.md), then link the second run's 1.5M-game snapshot as `best.npz`. Training is deterministic, so this reproduces the same weights.

Frontend replay viewer:

```bash
make benchmark         # writes frontend/public/benchmark.json
make replays           # writes frontend/public/replays/*.json
cd frontend && npm install && npm run dev
```

Evaluation results are cached per game in `output/benchmark_cache/`, keyed by agent label, checkpoint hash and game count, so `make benchmark` only replays agents that changed. Pass `ARGS="--rerun"` after changing an agent's code.

The viewer is a static replay player — inference happens offline in `scripts/generate_replays.py`, which records every board state, chosen action, and the network's Q-values per move.

## Repo layout

```
deep_2048/
  game/           # environment: board, moves, encoding, game loop, numba bitboard engine (shared by every agent)
  agents/
    base.py       # Agent interface: act(board, valid_actions) -> Decision
    heuristic/    # hand-crafted evaluator + expectimax
    dqn/          # network, learner, PER buffer, trainer, hybrid search, play-time policies
    ntuple/       # n-tuple network, TD training, greedy agent
  bench/          # seeded game runner (parallel across processes) and result cache used by evaluate.py
  config.py       # shared game settings (pydantic-settings, .env-overridable)
experiments/      # one folder per run: manifest, training history, per-game evaluations, notes
scripts/          # train_dqn / train_ntuple / evaluate / play / watch_ntuple / plot / generate_replays
tests/            # pytest: engine parity, network, training, agents
frontend/         # React + Vite replay viewer
.github/          # frontend CI + S3/CloudFront deploy (OIDC, no stored keys)
```

## Deployment

Pushes to `main` touching `frontend/**` build the viewer and sync it to S3 behind CloudFront. Auth uses GitHub OIDC role assumption (no long-lived AWS credentials); hashed assets get immutable cache headers while `index.html` stays no-cache, with a CloudFront invalidation on deploy.
