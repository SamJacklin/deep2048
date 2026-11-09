# DQN, 1M steps

The Rainbow-lite DQN described in the README, trained for 1,000,000 steps (moves) over 5,427 episodes in March 2026 on Apple Silicon, with the full-run configuration in `.env.example`. The exact commit wasn't recorded: the run predates the package refactor and experiment manifests.

## Record

- `training_history.csv`: one row per episode, with periodic in-training evaluations and losses.
- `checkpoints.json`: the metadata saved with each checkpoint, including the in-training evaluation used to pick `best`.
- `training_results.png`: training curves from `make plot`.
- `best.pt` is the checkpoint with the best in-training evaluation: step 925,281 (5,226 episodes).

## Evaluation

1,000 seeded games per checkpoint (`evaluations.json`), played on the CPU, mean score with a 95% confidence interval:

| Checkpoint | Moves trained | Avg score | ≥256 | ≥512 | ≥1024 |
|---|---:|---:|---:|---:|---:|
| step_100083 | 0.10M | 2,228 ± 74 | 42.3% | 4.0% | 0% |
| step_350068 | 0.35M | 2,631 ± 87 | 54.4% | 9.3% | 0% |
| step_625066 | 0.63M | 3,534 ± 108 | 77.5% | 21.3% | 0.7% |
| best | 0.93M | 5,689 ± 170 | 92.9% | 57.1% | 6.6% |

The published 100-game benchmark (6,157) is within the spread of these seeds; the first 100 happen to favour this agent.
