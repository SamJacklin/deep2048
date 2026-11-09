# Heuristic expectimax, depth 2

The hand-crafted evaluator with depth-2 expectimax. There's no training. The numba version makes the same decisions as the original Python version (checked on 12,709 boards and the published benchmark), so these results apply to both.

## Evaluation

1,000 seeded games (`evaluations.json`): average score 19,981 ± 597 (95% CI), with 86.2% of games reaching 1024, 37.2% reaching 2048 and 1.7% reaching 4096.
