"""Build the late-game corpus: positions recorded from whole games played by greedy play and by depth-3 search.

Usage:
    uv run python scripts/late_game_corpus.py checkpoints/ntuple/decision-8x6-tc-reset/games_2000000.npz

Plays seeded games with each policy in batches until it has --first-16384 games that made a 16384. Each game's first
8192 and first 16384 boards go to experiments/late-game-corpus/positions.csv, to be committed, and every 50th board to
output/late-game-corpus/samples.csv, which is too large to commit but rebuilt exactly by rerunning. Each row has greedy
play's relative margin between its top two moves, and the game it came from. Game i of a policy uses seed
OFFSET + i (greedy 1,000,000, depth 3 2,000,000), clear of the 0-999 evaluation seeds. Positions are split into
train, dev and test sets by source game (seeds ending in 0-5, 6-7 and 8-9), so no game's positions span two sets:
later training may use only the train set.
"""

import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_2048.agents.ntuple import NTupleNetwork
from deep_2048.agents.ntuple.diagnostics import FIRST_8192, FIRST_16384, record_game
from deep_2048.bench import file_sha256

POLICIES = {"greedy": (1, 1_000_000), "depth3": (3, 2_000_000)}
KIND_NAMES = {0: "first-8192", 1: "first-16384", 2: "sample"}
OUTPUT = Path("experiments/late-game-corpus")
SAMPLES = Path("output/late-game-corpus")
COLUMNS = ("policy", "seed", "split", "kind", "move", "score", "board", "max_tile", "empty", "greedy_margin", "game_score", "game_max_tile")


def split_of(seed: int) -> str:
    return "train" if seed % 10 < 6 else "dev" if seed % 10 < 8 else "test"


def main() -> None:
    parser = argparse.ArgumentParser(description="Record late-game positions from greedy and depth-3 games")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--first-16384", type=int, default=500, help="Games that made a 16384 to collect per policy (default: 500)")
    parser.add_argument("--batch", type=int, default=100, help="Games per batch (default: 100)")
    args = parser.parse_args()

    network, metadata = NTupleNetwork.load(args.checkpoint)
    weights, layout = network.weights, network.layout
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        for policy, (depth, offset) in POLICIES.items():
            games = made = 0
            while made < args.first_16384:
                seeds = range(offset + games, offset + games + args.batch)
                results = list(pool.map(lambda s: (s, record_game(weights, layout, depth, 0.0, np.random.default_rng(s))), seeds))
                for seed, (game_score, game_tile, kinds, moves, scores, boards, margins) in results:
                    if made >= args.first_16384:
                        break
                    games += 1
                    made += int((kinds == FIRST_16384).any())
                    for kind, move, score, board, margin in zip(kinds, moves, scores, boards, margins):
                        cells = [(int(board) >> (4 * i)) & 0xF for i in range(16)]
                        rows.append((policy, seed, split_of(seed), KIND_NAMES[int(kind)], int(move), int(score), f"{int(board):016x}",
                                     1 << max(cells), cells.count(0), float(margin), int(game_score), int(game_tile)))
                print(f"{policy}: {games} games, {made} made a 16384")
            first = [r for r in rows if r[0] == policy]
            assert all(r[7] == 8192 for r in first if r[3] == KIND_NAMES[FIRST_8192])
            assert all(r[7] == 16384 for r in first if r[3] == KIND_NAMES[FIRST_16384])

    SAMPLES.mkdir(parents=True, exist_ok=True)
    for path, chosen in ((OUTPUT / "positions.csv", [r for r in rows if r[3] != "sample"]), (SAMPLES / "samples.csv", [r for r in rows if r[3] == "sample"])):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
            writer.writerows(chosen)
    (OUTPUT / "source.txt").write_text(f"{args.checkpoint} ({metadata.games_played:,} games)\nsha256 {file_sha256(args.checkpoint)}\n")
    print(f"Wrote {sum(r[3] != 'sample' for r in rows):,} milestone positions to {OUTPUT / 'positions.csv'} and {sum(r[3] == 'sample' for r in rows):,} samples to {SAMPLES / 'samples.csv'}")


if __name__ == "__main__":
    main()
