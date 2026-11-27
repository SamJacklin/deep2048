"""The methods batch: learning rules from scratch on the 8×6, each to 3M games, against the long run.

Every arm uses the long run's recipe (the 8×6, downgrading from 16384, late-game starts, 8 threads) and differs only
in the learning rule. Each is its own run, methods-8x6-<arm>, drawn by scripts/watch_against_best.py --prefix methods-8x6.

    uv run python scripts/methods.py tc-0.005        # temporal coherence from scratch at base rate 0.005
    uv run python scripts/methods.py tc-0.01
    uv run python scripts/methods.py tc-0.04
    uv run python scripts/methods.py td-0.1 --until 300000    # plain TD, a step of 0.1 a board: the opening only
    uv run python scripts/methods.py td-0.5 --until 300000    # the same at 0.5 a board
    uv run python scripts/methods.py td-0.5                   # carry the chosen opening on, on Guei's schedule
    uv run python scripts/methods.py td-yeh --from td-0.5     # its branch on Yeh's rule
    uv run python scripts/methods.py td-yeh --from td-0.5 --now   # the same, from where the opening stands now
    uv run python scripts/methods.py td-guei-tc --from td-0.5 # a TC finish for the Guei-schedule run
    uv run python scripts/methods.py td-yeh-tc                # a TC finish for the Yeh-rule run
    uv run python scripts/methods.py td-opt                   # the long run: TD 0.1 from optimistic weights, to 8M
    uv run python scripts/methods.py td-opt --optimism 2500  # another value: its own run, methods-8x6-td-opt-2500
    uv run python scripts/methods.py td-opt --optimism 1250 --total 15000000 --hold-until 8000000 --tag hold
    uv run python scripts/methods.py td-opt --optimism 1250 --total 15000000 --hold-until 8000000 --tag hold-nodg --no-downgrade
    uv run python scripts/methods.py best-long                # the best network (reset at 1.5M) carried on from 2M to 5M
    uv run python scripts/methods.py stage2 --optimism 1250   # a late-game second stage on the best network: methods-8x6-stage2-opt1250
    uv run python scripts/methods.py stage2 --optimism 0 --stage-downgrade 16384 --tc-only --keep-sums --total 4000000
    uv run python scripts/methods.py stage2-finish --source-run stage2-opt1250-dg16384 --rate 0.005   # another TC finish, from where it switched
    uv run python scripts/methods.py joint --source-run stage2-opt1250-dg16384-tc0.005   # both stages trained together, 2M more games
    uv run python scripts/methods.py stage1-rate --rate 0.005 --total 10000000   # best-long from 8M at TC 0.005
    uv run python scripts/methods.py stage1-scale                       # a large-tile count term on tc0.005's 10M network
    uv run python scripts/methods.py stage1-scale --scale-by-state      # the same, read from the board each move is made from
    uv run python scripts/methods.py scratch-scale                      # the best network's path from zero, with the count term from game 1
    uv run python scripts/methods.py scratch-stall                      # 8x6 from zero, moving to the next rate at stalls
    uv run python scripts/methods.py scratch-stall --network 12x6       # the same schedule for a larger network, at the same step a board
    NTUPLE_SEED=43 uv run python scripts/methods.py scratch-stall --network 12x6 --tag seed43   # another seed, its own run
    NTUPLE_SMALL_TABLES=true uv run python scripts/methods.py scratch-stall --tag small   # with the two small-tile tables, its own run (set it again to resume)
    uv run python scripts/methods.py second-reset --base FILE --network 12x6   # a network grown by add_tuples.py, on scratch-stall's schedule from its second phase
    uv run python scripts/methods.py second-reset                       # scratch-stall's 4.6M network: reset the sums again, then cut at the stall
    uv run python scripts/methods.py stage1-goal --goal-value 160000   # stage 1 tailored to building 16384
    uv run python scripts/methods.py stage1-focus             # the best network trained on from first-4096 boards, 2M more games

The plain TD arms
- Rates are given as the step a board's value takes (the papers' scale) and divided by the weights an update
  touches, 64 on the 8×6, for the trainer. Guei et al. (2021) and Yeh et al. (2016) both start near 0.1 a board, for
  runs of 5M to 100M games; 0.5 is this project's guess at what a 3M-game run needs. The first 300k games of each
  decide which opening carries on.
- An opening runs on Guei's schedule: its starting rate to halfway, a tenth of it to three quarters, a hundredth to
  the end. On the way it watches for its first stall (below), and when that comes it keeps a copy of the network.
- td-yeh starts from that copy and follows Yeh's rule: cut the rate to a tenth whenever training stalls, at most
  twice (so it ends where Guei's schedule ends). The two share every game up to the first stall.
- A stall is a gain of less than STALL_GAIN in the smoothed training score over the last STALL_WINDOW games, with
  at least that many games played at the current rate. It is checked every STEP games.
- Each TC finish takes its run's network at 2.5M games and trains the last 500k with temporal coherence at 0.02,
  its sums starting from zero. Guei gives TC the last 10%; a fresh TC start dips for about 150k games here, so it
  gets 500k.

The long run (td-opt)
- Plain TD at 0.1 a board from optimistic weights: every weight starts at --optimism (default 100, so a board starts
  at 64 times that). Under plain TD a weight loses about a tenth of its excess a visit, so optimism lasts far longer
  than it did under TC, whose full first step wipes it in a visit. This is Guei's optimistic TD.
- The rate is cut to a tenth at a long stall, twice at most (0.1, 0.01, 0.001 a board). A long stall is under
  LONG_GAIN in the smoothed score AND under LONG_GAIN_32768 percentage points in the 32768 rate over the last
  LONG_WINDOW games, all at the current rate: the batch showed the 32768 rate climbing for 500k games after the score
  looked flat.
- Each cut also has a latest point, Guei's schedule over the TD part of the budget (total - tc-games): the first by
  halfway, the second by three quarters (3M and 4.5M of the default 6M), and never sooner than LONG_WINDOW games after
  the cut before it. A run still creeping up then gets its cuts anyway.
- After the second cut, at the next long stall or once --total minus --tc-games is reached (whichever comes first),
  it switches to temporal coherence at 0.02 from zeroed sums for --tc-games games (default 2M), then stops.
- A copy of the network is kept before each cut and before the switch, for branching later.
- Compare it with td-0.1 at the same games: the same rate from plain zero weights.
- --hold-until GAMES keeps the opening rate to that many games with no stall checks (the first cut lands there unless
  the schedule's latest point is later), so optimism can wear off at the full rate for longer, as in Guei's runs, whose
  first cut comes at half the budget. --no-downgrade trains without reading boards from 16384 up as smaller ones (Guei
  has no downgrading), to see whether downgrading takes away what optimism is for. --tag names a variant's run:
  methods-8x6-td-opt-<value>-<tag>. A run's budget, hold and downgrading are fixed when it starts.
- The default value, 100, runs as methods-8x6-td-opt; any other value is its own run, methods-8x6-td-opt-<value>.
  Guei et al. spread 80k-320k a board over the weights; with 64 weights read a board, 2,500 is 160k a board.

The best recipe, carried on (best-long)
- decision-8x6-tc-reset's 2M network (the best so far: 320k greedy) trained on with its own recipe, temporal
  coherence at 0.02 with its sums kept, to --total (default 5M). Nothing about it changes: it is the baseline any
  longer method has to beat, and it shows whether the best recipe's own plateau is still learning slowly underneath.
- The 2M network is only read: the finisher patches are tied to it.

Stage 2 (stage2)
- The plan's second stage on the best network (decision-8x6-tc-reset at 2M), which stays frozen as stage 1. Boards
  whose real largest tile is 16384 or more read a second copy of the weights, copied from stage 1, which downgrades
  only from 32768 (stage 1 downgrades from 16384). Every second-stage weight still at 0 (all patterns holding a real
  16384, never used by stage 1, and anything else stage 1 never learned) starts at --optimism.
- Every training game starts from a pool of first-16384 boards from stage 1's own greedy games (built once, shared
  by every stage2 run), so only the second stage trains.
- The schedule is the long runs': TD at 0.1 a board held to --hold-until (default 5M) with no stall checks, cuts to a
  tenth at long stalls or the latest points, then the TC finish for --tc-games (default 2M), within --total (default
  10M). Evaluations every 500k are whole games on the usual seeds, stage 1 then stage 2: the number to compare with
  stage 1 alone (320.0k, 8.5% 32768). The training games' 32768 rate is the conversion from 16384.
- --stage-downgrade 16384 keeps stage 2 reading boards as stage 1 does, so its start is an exact copy that plays like
  stage 1 and only patterns stage 1 never saw start optimistic (run methods-8x6-stage2-opt<value>-dg16384). With the
  default 32768, every pattern holding a real 16384 is new to stage 2 and starts at the optimism value.
- stage2-finish --source-run <stage2 arm> --rate R branches a second TC finish at base rate R from the network the
  run kept when it switched to TC (waiting for the switch if need be), for the same TC games, as
  methods-8x6-<arm>-tc<R>: whether the default 0.02 jolts a tuned network more than a smaller rate.
- --tc-only --stage-downgrade 16384 --keep-sums --optimism 0 starts stage 2 as an exact copy of stage 1 in its sums
  too, so every weight carries on at the TC rate it had reached and only moves where its errors keep the same sign
  after 16384: stage 1's knowledge refined for the 16384 stage, never overwritten (run methods-8x6-stage2-opt0-dg16384-tc-sums).

Stage 1 focus (stage1-focus)
- The best network (decision-8x6-tc-reset at 2M) trained on with TC at 0.02, its sums reset to zero so every weight
  starts back at the full rate (as at that run's own reset at 1.5M, and the TC finishes), but every
  game starts from a pool of first-4096 boards from its own greedy games (built once), so all the training lands
  where 22.6% of its games die short of 16384. No search: ordinary TD learning, more of it in that stretch.
- --total is the games to add (default 2M). Every 250k games a greedy whole-game evaluation on the usual seeds, the
  number to compare with the best network's own (320.0k, 77.4% 16384). The training games' 16384 rate is the
  conversion from 4096.

Both stages together (joint)
- joint --source-run <finished stage2 arm> trains that run's network (stage 1 + stage 2) on whole games with the best
  recipe: TC at 0.02, late starts at 8192, --total more games (default 2M), a whole-game evaluation every 250k. Each
  weight carries on from its own TC sums (stage 1's from the best network, stage 2's from the run's finish), so
  nothing is jolted. Stage 2 plays every board from 16384 up, so stage 1's tables are freed from those boards and
  can specialise in the 8192 stage, where 18.5 of the 22.6 points short of 16384 are lost. Whole games keep the
  opening trained. Run methods-8x6-joint-<source arm>.

Stage 1 at another rate (stage1-rate)
- stage1-rate --rate R [--base FILE] [--total N] trains best-long's 8M copy on with its own recipe (TC, sums kept, late
  starts) but base rate R instead of 0.02, to --total games in all (default 2M more), evaluated every 500k like
  best-long: the same games as best-long's own continuation, so the two compare directly. Run methods-8x6-best-long-tc<R>.

A large-tile count term (stage1-scale)
- stage1-scale [--rate R] [--scale-rate S] [--base FILE] [--total N] adds Yeh's large-tile count term (a weight per
  count of 2048s, 4096s, 8192s, 16384s and 32768s on the real board) at zero to tc0.005's 10M copy, then trains on
  like stage1-rate (TC at 0.005, sums kept, late starts), evaluated every 500k. Its two fixes for the scale-sharing
  test's failure: boards without a 2048 read no term (that entry was one bias shared by every early board), and the
  term learns at S times the tuples' rate (default 1/64, one tuple read's share). Compared with tc0.005's own
  continuation from 10M. Run methods-8x6-stage1-scale. The first try learned a penalty of over 100k for every count
  holding a 16384 (downgrading overvalues those boards), so the network stopped making 16384. --scale-below 16384
  keeps boards with a 16384 or more off the table (run methods-8x6-stage1-scale-below16384). That one still learned
  +33k to +55k for boards with an 8192 short of 16384, which the move making 16384 drops. --scale-by-state reads the
  term from the board the move is made from, as the stage is chosen, so every move of a decision reads the same entry:
  the term can't favour one move or put a cliff at a merge, only take up the part of the value the large tiles already
  on the board explain, leaving the tuples the rest (run methods-8x6-stage1-scale-by-state, -below<tile> if both).

The count term from scratch (scratch-scale)
- scratch-scale [--scale-rate S] retraces the best network's whole path from zero with the large-tile count term in
  it from the first game: 8x6, TC at 0.02, downgrading from 16384, late starts from 8192 in a fifth of games (the long
  run's recipe); the TC sums reset at 1.5M; on at 0.02 to 8M; then 0.005 to 10M. The term counts every large tile
  (2048 to 32768), skips boards without a 2048 and learns at S of the tuples' rate (default 1/64). Patched onto the
  converged network it made cliffs at merges; from the start the tuples can learn alongside it, and it's the feature
  that can tell a first 16384 from a first 8192, which downgrading reads alike. 1,000-game evaluations every 500k and
  copies kept at 2M, 5M, 8M and 10M, for comparison with the best network's own path. Run methods-8x6-scratch-scale;
  --scale-by-state reads the term from the board each move is made from (methods-8x6-scratch-scale-by-state).
  scripts/stage1_dashboard.py --scratch shows them against the best network's own path on absolute games.

Moving on at stalls from scratch (scratch-stall)
- scratch-stall [--total N] [--with-scale] trains 8x6 from zero with the long run's recipe (TC, downgrading from 16384,
  late starts from 8192 in a fifth of games) and moves through four phases at stalls instead of at set games: TC at
  0.02; TC at 0.02 with the sums reset; 0.005; 0.00125. Each phase runs at least the long window (500k) before it's
  checked, every 100k, and a stall is only acted on once the phase has run its floor (1.5M for the first, 1M for the
  rest): earlier, the first point the criteria were met is logged, kept as would_stall_<n>.npz and recorded in the
  state as would_have; a stall is the long runs' rule (score up less than 3k and the 32768 rate up less than 0.5
  points over 500k). A stall in the last phase ends the run, as does --total (default 10M). A copy is kept at each
  stall. The control for whether the best network's fixed path (reset at 1.5M, cut at 8M) is the best route.
  --with-scale adds the count term as in scratch-scale. Run methods-8x6-scratch-stall.
  --network NAME trains another network on the same schedule (run methods-8x6-scratch-stall-NAME), its rates a weight
  scaled so a board's total step matches the 8x6's.
- second-reset [--base FILE] [--total N] carries scratch-stall's 4.6M network on with the same stall rule through two
  phases: TC at 0.02 with the sums reset, then 0.005, each running at least 500k. A stall at 0.005 ends it, as does
  --total games added (default 3M). Run methods-8x6-second-reset.

A tailored stage 1 (stage1-goal)
- stage1-goal --goal-value B [--base FILE] trains a single-stage network (default best-long's 5M copy) on whole games
  from an empty board, each ending on the move that makes 16384, worth B after its reward: it learns to build 16384,
  scoring well on the way, and nothing past it, which is stage 2's job. Moves to 16384 are chosen with B exactly, and
  its afterstates learn B. No downgrading (the goal's boards get patterns of their own), no late starts, TC at 0.02
  with the base's sums kept, --total more games (default 2M), a whole-game evaluation every 250k. Its scores past
  16384 mean nothing: the 16384 rate is the measure, then stage2 --base on it. Run methods-8x6-stage1-goal<B/1000>k.

Housekeeping
- Training runs in sessions of at most STEP games (the TC arms, EVAL_EVERY), each a foreground scripts/train_ntuple.py.
  scripts/methods_dashboard.py shows the whole batch live in a browser.
  Rerunning a command resumes its arm. To stop one: pkill -f -- "--run methods-8x6-<arm> --" (the trainer saves a
  checkpoint and this script stops with it).
- A session only starts with MIN_FREE_GB of disk free, since a save that runs out of space kills the run.
- Greedy evaluations on the usual seeds are made of latest.npz when a session ends on a multiple of EVAL_EVERY, so
  no snapshots are kept for them. The only extra files are the copy at the first stall and each TD run's 2.5M
  network, which the branches start from.
- td-yeh and the TC finishes wait, checking once a minute, for the network they start from.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_2048.agents.ntuple.network import NETWORKS  # noqa: E402
from deep_2048.bench import file_sha256  # noqa: E402
from deep_2048.agents.ntuple.train import HISTORY_NAME  # noqa: E402
from deep_2048.experiments import EXPERIMENTS_DIR  # noqa: E402

ROOT = Path(__file__).parent.parent
CHECKPOINTS = Path("checkpoints/ntuple")
SYMMETRIES = 8

TC_ARMS = {"tc-0.005": 0.005, "tc-0.01": 0.01, "tc-0.04": 0.04}  # base rate a weight
OPENINGS = {"td-0.1": 0.1, "td-0.5": 0.5}  # step a board
ARMS = (*TC_ARMS, *OPENINGS, "td-yeh", "td-guei-tc", "td-yeh-tc", "td-opt", "best-long", "stage2", "stage2-finish", "stage1-focus", "joint", "stage1-goal",
        "stage1-rate", "stage1-scale", "scratch-scale", "scratch-stall", "second-reset")
BEST = CHECKPOINTS / "decision-8x6-tc-reset" / "games_2000000.npz"  # the best network so far
FINISH_RATE = 0.02  # the TC finishes' base rate a weight, the long run's
LONG_START = 0.1  # td-opt's opening step a board


class Plan:
    """The batch's sizes: the real ones, or a few thousand games on a small network to check the machinery."""

    def __init__(self, smoke: bool) -> None:
        self.smoke = smoke
        self.prefix = "smoke-methods" if smoke else "methods-8x6"
        self.network = "2x4-2x6" if smoke else "8x6"
        self.total = 6_000 if smoke else 3_000_000
        self.step = 500 if smoke else 100_000
        self.eval_every = 2_000 if smoke else 500_000
        self.branch_at = 5_000 if smoke else 2_500_000
        self.stall_window = 500 if smoke else 250_000
        # The smoke test stalls as soon as the window allows, so every branch and cut is exercised
        self.stall_gain = float("inf") if smoke else 3_000.0
        self.smooth = 4 if smoke else 50  # history rows averaged for a score
        self.eval_games = 5 if smoke else 1_000
        self.min_free_gb = 0.2 if smoke else 6.0
        self.poll_seconds = 5 if smoke else 60
        self.reads = len(NETWORKS[self.network]) * SYMMETRIES
        self.cuts = (self.total // 2, self.total * 3 // 4)  # Guei's schedule: a tenth from halfway, a hundredth from three quarters
        # td-opt's long stall: both the score and the 32768 rate flat over a longer window
        self.long_total = 12_000 if smoke else 8_000_000
        self.tc_games = 3_000 if smoke else 2_000_000
        self.long_window = 1_000 if smoke else 500_000
        self.long_gain = float("inf") if smoke else 3_000.0
        self.long_gain_32768 = float("inf") if smoke else 0.5  # percentage points
        self.downgrade = True  # td-opt --no-downgrade turns it off for its run
        self.extra_env: dict[str, str] = {}  # stage2's settings for its trainer
        # stage2: the second stage's tile and downgrading tile, the pool's games, the budget and the hold
        self.stage_tile, self.stage_downgrade = (1024, 2048) if smoke else (16384, 32768)
        self.pool_games = 300 if smoke else 65_000
        self.stage2_total, self.stage2_hold = (12_000, 6_000) if smoke else (10_000_000, 5_000_000)
        self.stage_base: Path | None = None  # stage2 --base: stage 1 from another network than the best, with its own pool
        self.base_tag = ""
        # stage1-focus: the tile its games start from, the pool's games, the games it adds and its evaluation spacing
        self.focus_tile = 256 if smoke else 4096
        self.focus_pool_games = 300 if smoke else 40_000
        self.focus_games = 3_000 if smoke else 2_000_000
        self.focus_eval = 1_000 if smoke else 250_000

    def environment(self, rate: float, coherent: bool) -> dict[str, str]:
        settings = {"NTUPLE_NETWORK": self.network, "NTUPLE_LEARNING_RATE": repr(rate), "NTUPLE_OPTIMIZER": "tc" if coherent else "td",
                    "NTUPLE_TEMPORAL_COHERENCE": "true" if coherent else "false", "NTUPLE_DOWNGRADE_TILE": "16384", "NTUPLE_LATE_START_FRACTION": "0.2",
                    "NTUPLE_LATE_START_TILE": "8192", "NTUPLE_THREADS": os.environ.get("THREADS", "2" if self.smoke else "8")}
        if self.smoke:
            settings |= {"NTUPLE_LOG_EVERY": "50", "NTUPLE_CHECKPOINT_EVERY": "250", "NTUPLE_THREADS_AFTER": "200", "NTUPLE_LATE_START_TILE": "512", "NTUPLE_DOWNGRADE_TILE": "2048"}
        environment = os.environ | settings | self.extra_env
        if not self.downgrade:
            environment.pop("NTUPLE_DOWNGRADE_TILE", None)
        return environment


class Run:
    def __init__(self, plan: Plan, arm: str) -> None:
        self.plan, self.arm = plan, arm
        self.name = f"{plan.prefix}-{arm}"
        self.checkpoints = CHECKPOINTS / self.name
        self.experiment = EXPERIMENTS_DIR / self.name
        self.latest = self.checkpoints / "latest.npz"
        self.state_path = self.checkpoints / "methods.json"
        self.state: dict = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}

    def save_state(self) -> None:
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2) + "\n")

    def history(self) -> np.ndarray | None:
        path = self.experiment / HISTORY_NAME
        if not path.exists():
            return None
        history = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
        return history if history.size else None

    def games(self) -> int:
        """Games played so far: where the history ends, or where the run starts if it hasn't trained yet."""
        history = self.history() if self.latest.exists() else None
        return int(history["games"][-1]) if history is not None else int(self.state.get("start", 0))

    def score(self, games: int, column: str = "avg_score") -> float | None:
        """The training score (or another history column) at a game count: the mean of the last `smooth` batches up to
        it, or None if the run's own history doesn't reach back that far."""
        history = self.history()
        if history is None or column not in history.dtype.names:
            return None
        upto = history[column][history["games"] <= games]
        return float(upto[-self.plan.smooth:].mean()) if upto.size >= self.plan.smooth else None

    def stalled(self, games: int, since: int) -> bool:
        """Whether the score has gained less than the plan's threshold over the last window, all of it at the current rate."""
        window = self.plan.stall_window
        if games - since < window:
            return False
        now, before = self.score(games), self.score(games - window)
        if now is None or before is None:
            return False
        print(f"  stall check at {games:,} games: {now - before:+,.0f} over the last {window:,} games (a stall is under {self.plan.stall_gain:+,.0f})", flush=True)
        return now - before < self.plan.stall_gain

    def stalled_long(self, games: int, since: int) -> bool:
        """td-opt's rule: under the threshold in both the score and the 32768 rate over the last long window, all at the current rate."""
        window = self.plan.long_window
        if games - since < window:
            return False
        gains = []
        for column in ("avg_score", "rate_32768"):
            now, before = self.score(games, column), self.score(games - window, column)
            if now is None or before is None:
                return False
            gains.append(now - before)
        print(f"  long stall check at {games:,} games, over the last {window:,}: score {gains[0]:+,.0f} (a stall is under {self.plan.long_gain:+,.0f}), "
              f"32768 rate {gains[1]:+.2f} points (under {self.plan.long_gain_32768:+.2f})", flush=True)
        return gains[0] < self.plan.long_gain and gains[1] < self.plan.long_gain_32768

    def require_space(self) -> None:
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.checkpoints).free / 1e9
        if free < self.plan.min_free_gb:
            raise SystemExit(f"Only {free:.1f} GB of disk is free and {self.name} needs {self.plan.min_free_gb:g} GB to save safely. Free some space and rerun the same command: it resumes.")

    def train(self, target: int, rate: float, coherent: bool, start_from: Path | None = None, optimism: float | None = None, reset_coherence: bool = False,
              reset_own: bool = False) -> None:
        """One training session, up to `target` games in total. Stops this script if the trainer stopped short.
        `reset_coherence` zeroes the TC sums of `start_from` in the first session (never a resumed run's own);
        `reset_own` zeroes the run's own sums as this session resumes it (a scheduled reset)."""
        self.require_space()
        command = [sys.executable, "-u", "scripts/train_ntuple.py", "--run", self.name, "--games", str(target)]
        if self.latest.exists():
            command += ["--resume", str(self.latest)] + (["--reset-coherence"] if reset_own else [])
        elif start_from is not None:
            command += ["--resume", str(start_from)] + (["--reset-coherence"] if reset_coherence else [])
        elif optimism is not None:  # only the very first session: a resumed run's weights are already its own
            command += ["--optimistic-unvisited", repr(optimism)]
        # What the session is doing, for scripts/methods_dashboard.py: its rate isn't in the history
        self.state |= {"current": [self.games(), target, rate, "tc" if coherent else "td"], "reads": self.plan.reads, "total": self.state.get("total", self.plan.total)}
        self.save_state()
        kind = "temporal coherence, base rate" if coherent else "plain TD, rate"
        print(f"\n{self.name}: training to {target:,} games ({kind} {rate:g} a weight, {rate * self.plan.reads:g} a board)", flush=True)
        self.experiment.mkdir(parents=True, exist_ok=True)
        with open(self.experiment / "train.log", "ab") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=self.plan.environment(rate, coherent), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            try:
                while chunk := process.stdout.read1(65536):
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                    log.write(chunk)
            except KeyboardInterrupt:
                process.wait()  # the trainer has the same interrupt and saves a checkpoint on its way out
                raise SystemExit(f"\n{self.name} interrupted. Rerun the same command to resume.") from None
            process.wait()
        if self.games() < target:
            raise SystemExit(f"\n{self.name} stopped at {self.games():,} games, short of {target:,}. Rerun the same command to resume.")
        self.state.setdefault("rates", []).append([target, rate, "tc" if coherent else "td"])
        self.state.pop("current", None)
        self.save_state()

    def evaluate(self, games: int) -> None:
        """Greedy on the usual seeds, from latest.npz, recorded in the run's evaluations.json."""
        if games in self.state.get("evaluated", []):
            return
        command = [sys.executable, "scripts/evaluate.py", str(self.latest), "--agent", "ntuple", "-n", str(self.plan.eval_games), "--experiment", str(self.experiment)]
        output = subprocess.run(command, cwd=ROOT, capture_output=True, text=True).stdout.replace("\r", "\n")
        lines = [line for line in output.splitlines() if line.startswith("N-tuple")]
        print("\n".join(lines) if lines else f"The evaluation at {games:,} games printed no result:\n{output[-2000:]}", flush=True)
        if lines:
            self.state.setdefault("evaluated", []).append(games)
            self.save_state()

    def keep_copy(self, name: str) -> Path:
        """Copy latest.npz to a file a branch can start from."""
        self.require_space()
        path = self.checkpoints / name
        shutil.copyfile(self.latest, path.with_suffix(".tmp"))
        os.replace(path.with_suffix(".tmp"), path)
        print(f"  kept {path}", flush=True)
        return path

    def after_session(self, games: int) -> None:
        if games == self.plan.branch_at and self.arm not in TC_ARMS and not self.arm.endswith("-tc") and not self.arm.startswith(("td-opt", "stage2")) and self.arm != "best-long":
            self.keep_copy(f"games_{games}.npz")
        if games % self.plan.eval_every == 0:
            self.evaluate(games)


def next_stop(games: int, stops: list[int]) -> int:
    return min(stop for stop in stops if stop > games)


def session_stops(plan: Plan, limit: int, extra: tuple[int, ...] = ()) -> list[int]:
    return sorted({*range(plan.step, limit + 1, plan.step), *(stop for stop in extra if stop <= limit), limit})


def run_tc(plan: Plan, arm: str, limit: int) -> None:
    run = Run(plan, arm)
    stops = session_stops(plan, limit)
    stops = [stop for stop in stops if stop % plan.eval_every == 0 or stop == limit]
    while (games := run.games()) < limit:
        target = next_stop(games, stops)
        run.train(target, TC_ARMS[arm], coherent=True)
        run.after_session(target)


def run_opening(plan: Plan, arm: str, limit: int) -> None:
    """Plain TD from scratch on Guei's schedule, keeping a copy of the network at its first stall for td-yeh."""
    run = Run(plan, arm)
    start = OPENINGS[arm] / plan.reads
    stops = session_stops(plan, limit, plan.cuts)
    while (games := run.games()) < limit:
        target = next_stop(games, stops)
        rate = start / 10 ** sum(games >= cut for cut in plan.cuts)
        run.train(target, rate, coherent=False)
        run.after_session(target)
        if "first_stall" not in run.state and target < plan.cuts[0] and run.stalled(target, since=0):
            run.keep_copy("first_stall.npz")
            run.state["first_stall"] = target
            run.save_state()
            print(f"{run.name} first stalled at {target:,} games: td-yeh can branch from here. This run carries on at the same rate to {plan.cuts[0]:,}.", flush=True)
        elif "first_stall" not in run.state and target >= plan.cuts[0]:
            run.state["first_stall"] = None  # the schedule's own cut came first: Yeh's rule and Guei's schedule are the same run
            run.save_state()
            print(f"{run.name} reached its first scheduled cut without stalling, so Yeh's rule has nothing different to do.", flush=True)


def wait_for(plan: Plan, what: str, ready) -> None:
    announced = False
    while not ready():
        if not announced:
            print(f"Waiting for {what} (checking every {plan.poll_seconds} s)…", flush=True)
            announced = True
        time.sleep(plan.poll_seconds)


def run_yeh(plan: Plan, opening: str, limit: int, now: bool) -> None:
    """The opening's network at its first stall, carried on with the rate cut to a tenth at every stall (twice at most).

    With `now`, the opening is taken to have stalled where it stands: the rule looks back STALL_WINDOW games, so it
    is slow to notice a run that went flat early, and a plateau plain on the chart needn't wait for it.
    """
    run, source = Run(plan, "td-yeh"), Run(plan, opening)
    if now and "start" not in run.state and source.state.get("first_stall") is None:
        if not source.latest.exists():
            raise SystemExit(f"{source.name} hasn't trained yet")
        if time.time() - (source.experiment / HISTORY_NAME).stat().st_mtime < 120:
            raise SystemExit(f"{source.name} is still training: stop it first (pkill -f -- \"--run {source.name} --\"), then rerun this.")
        source.keep_copy("first_stall.npz")
        source.state["first_stall"] = source.games()
        source.save_state()
        print(f"{source.name} taken as stalled at {source.state['first_stall']:,} games", flush=True)
    if "start" not in run.state:
        def decided() -> bool:
            source.state = json.loads(source.state_path.read_text()) if source.state_path.exists() else {}
            return "first_stall" in source.state

        wait_for(plan, f"{source.name}'s first stall", decided)
        if source.state["first_stall"] is None:
            raise SystemExit(f"{source.name} reached its scheduled cut without stalling, so there is no separate Yeh-rule run to make.")
        run.state |= {"start": source.state["first_stall"], "from": opening, "cut_at": [source.state["first_stall"]], "rate": OPENINGS[opening] / plan.reads / 10}
        run.save_state()
    stops = session_stops(plan, limit, (plan.branch_at,))
    while (games := run.games()) < limit:
        target = next_stop(games, stops)
        run.train(target, run.state["rate"], coherent=False, start_from=source.checkpoints / "first_stall.npz")
        run.after_session(target)
        if len(run.state["cut_at"]) < 2 and run.stalled(target, since=run.state["cut_at"][-1]):
            run.state["rate"] /= 10
            run.state["cut_at"].append(target)
            run.save_state()
            print(f"{run.name} stalled at {target:,} games: rate cut to {run.state['rate']:g} a weight ({run.state['rate'] * plan.reads:g} a board)", flush=True)


def run_finish(plan: Plan, arm: str, source_arm: str, limit: int) -> None:
    """The source run's network at the branch point, trained on with temporal coherence from zeroed sums."""
    run, source = Run(plan, arm), Run(plan, source_arm)
    start = source.checkpoints / f"games_{plan.branch_at}.npz"
    if "start" not in run.state:
        wait_for(plan, str(start), start.exists)
        run.state |= {"start": plan.branch_at, "from": source_arm}
        run.save_state()
    stops = sorted({plan.branch_at + (plan.total - plan.branch_at) // 2, limit})
    while (games := run.games()) < limit:
        target = next_stop(games, stops)
        run.train(target, FINISH_RATE, coherent=True, start_from=start)
        run.evaluate(target)


def long_arm(optimism: float, tag: str | None = None) -> str:
    """td-opt's run for an optimism value (and a variant's tag): the first one, at 100, keeps the plain name."""
    name = "td-opt" if optimism == 100 and not tag else f"td-opt-{optimism:g}"
    return f"{name}-{tag}" if tag else name


def cut_deadline(state: dict, window: int) -> int:
    """The latest point for the run's next cut: Guei's schedule over its TD games, at least `window` after the last cut."""
    td_games = state["total"] - state["tc_games"]
    cuts = state["cut_at"]
    scheduled = (td_games // 2, td_games * 3 // 4)[len(cuts)]
    return max(scheduled, state.get("hold_until", 0), (cuts[-1] if cuts else 0) + window)


def run_long(plan: Plan, total: int, until: int, optimism: float, tc_games: int, tag: str | None = None, hold_until: int = 0, downgrade: bool = True,
             arm: str | None = None, start_from: Path | None = None) -> None:
    """Plain TD at 0.1 a board from optimistic weights, cut a tenth at each long stall (twice), then a TC finish.

    `total` is the run's budget (the TC finish starts by total - tc_games at the latest); `until` only pauses it early.
    With `start_from` (stage2's prepared network, already optimistic where it should be) the first session starts
    from it and the trainer applies no optimism of its own.
    """
    run = Run(plan, arm or long_arm(optimism, tag))
    if "start" not in run.state:
        run.state |= {"start": 0, "optimism": optimism, "rate": LONG_START / plan.reads, "cut_at": [], "tc_from": None, "total": total, "tc_games": tc_games,
                      "hold_until": hold_until, "downgrade": downgrade}
        run.save_state()
    elif run.state["optimism"] != optimism:
        raise SystemExit(f"{run.name} was started with optimism {run.state['optimism']:g}; rerun with --optimism {run.state['optimism']:g}")
    elif run.state.get("downgrade", True) != downgrade:
        raise SystemExit(f"{run.name} was started {'with' if run.state.get('downgrade', True) else 'without'} downgrading: rerun with the same setting")
    plan.downgrade = run.state.get("downgrade", True)
    hold_until = run.state.get("hold_until", 0)
    if "tc_games" not in run.state:  # runs started before the latest cut points were recorded
        run.state["tc_games"] = tc_games
        run.save_state()
    tc_games = run.state["tc_games"]
    while (games := run.games()) < min(until, run.state["total"]):
        target = min(until, run.state["total"], (games // plan.step + 1) * plan.step)
        if run.state["tc_from"] is not None:
            run.train(target, FINISH_RATE, coherent=True, start_from=start_from)
            run.after_session(target)
            continue
        run.train(target, run.state["rate"], coherent=False, start_from=start_from, optimism=None if start_from is not None else optimism)
        run.after_session(target)
        since = run.state["cut_at"][-1] if run.state["cut_at"] else 0
        two_cuts = len(run.state["cut_at"]) >= 2
        stalled = target > hold_until and run.stalled_long(target, since)
        if target >= run.state["total"] - tc_games or (two_cuts and stalled):
            run.keep_copy(f"td_end_{target}.npz")
            run.state["tc_from"] = target
            run.state["total"] = min(run.state["total"], target + tc_games)
            run.save_state()
            print(f"{run.name}: switching to temporal coherence at {FINISH_RATE:g} a weight from {target:,} games, to {run.state['total']:,}", flush=True)
        elif not two_cuts and (stalled or target >= cut_deadline(run.state, plan.long_window)):
            why = "stalled" if stalled else "reached its latest cut point"
            run.keep_copy(f"before_cut_{target}.npz")
            run.state["rate"] /= 10
            run.state["cut_at"].append(target)
            run.state.setdefault("cut_why", []).append("stall" if why == "stalled" else "deadline")
            run.save_state()
            print(f"{run.name} {why} at {target:,} games: rate cut to {run.state['rate']:g} a weight ({run.state['rate'] * plan.reads:g} a board)", flush=True)
    print(f"\n{run.name} is at {run.games():,} games (budget {run.state['total']:,}).")


def stage2_arm(optimism: float, tag: str | None) -> str:
    return f"stage2-opt{optimism:g}" + (f"-{tag}" if tag else "")


def prepare_stage2(plan: Plan, run: "Run", optimism: float, keep_sums: bool = False) -> Path:
    """Build (once) the shared pool of first-16384 boards and this run's starting network; set the trainer's settings.
    `keep_sums` gives the second stage a copy of stage 1's temporal coherence sums too, so each weight carries on at the rate it had reached."""
    from deep_2048.agents.ntuple import NTupleNetwork
    from deep_2048.agents.ntuple.models import CheckpointMetadata
    from deep_2048.agents.ntuple.stages import first_boards, stage_two

    if plan.stage_base is not None:
        base_path = plan.stage_base
        pool_path = CHECKPOINTS / f"{plan.prefix}-stage2-pool-{plan.base_tag}" / f"first-{plan.stage_tile}.npy"
    else:
        base_path = BEST if not plan.smoke else Run(plan, "best-long").latest
        pool_path = CHECKPOINTS / f"{plan.prefix}-stage2-pool" / f"first-{plan.stage_tile}.npy"
    if not base_path.exists():
        raise SystemExit(f"{base_path} is missing")
    start = run.checkpoints / "start.npz"
    base = None
    if not pool_path.exists() or not start.exists():
        base, _ = NTupleNetwork.load(base_path)
    if not pool_path.exists():
        pool_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Building the pool: {plan.pool_games:,} greedy games of {base_path}, each one's first board holding {plan.stage_tile}...", flush=True)
        boards, finals = first_boards(base, plan.stage_tile, plan.pool_games, 90_000_000, int(os.environ.get("THREADS", "8")))
        np.save(pool_path, boards)
        (pool_path.parent / "source.txt").write_text(f"{base_path}\nsha256 {file_sha256(base_path)}\n{plan.pool_games} games from seed 90,000,000: "
                                                       f"{boards.size} reached {plan.stage_tile}; mean score {finals.mean():,.0f}\n")
        print(f"  {boards.size:,} boards ({boards.size / plan.pool_games:.1%} of games), mean score {finals.mean():,.0f}: {pool_path}", flush=True)
    if not start.exists():
        staged, optimistic = stage_two(base, plan.stage_tile, plan.stage_downgrade, optimism)
        tables = sum(16 ** len(t) for t in staged.tuples)
        extra = None
        if keep_sums:
            with np.load(base_path) as data:
                if "coherence_errors" not in data:
                    raise SystemExit(f"{base_path} holds no temporal coherence sums to keep")
                # Laid out as stage_two lays out the weights: stage 1's tables, their copy for stage 2, then anything after
                extra = {name: np.concatenate([data[name][:tables], data[name][:tables], data[name][tables:]]) for name in ("coherence_errors", "coherence_abs_errors")}
        metadata = CheckpointMetadata(games_played=0, moves_played=0, learning_rate=(FINISH_RATE if keep_sums else LONG_START / plan.reads), temporal_coherence=keep_sums,
                                      optimizer="tc" if keep_sums else "td")
        staged.save(start, metadata, extra=extra)
        print(f"{start}: stage 1 from {base_path}; stage 2 from {plan.stage_tile} (downgrading from {plan.stage_downgrade}), "
              f"{optimistic:,} of its {tables:,} weights ({optimistic / tables:.1%}) untouched and set to {optimism:g}", flush=True)
        run.state |= {"base": str(base_path), "base_sha256": file_sha256(base_path), "pool": str(pool_path), "optimistic_weights": optimistic}
        run.save_state()
    plan.extra_env = {"NTUPLE_DOWNGRADE_TILE": str(int(json.loads(str(np.load(start)["features"]))["downgrade_tile"])), "NTUPLE_STAGE_TILE": str(plan.stage_tile),
                      "NTUPLE_STAGE_DOWNGRADE_TILE": str(plan.stage_downgrade), "NTUPLE_START_POOL": str(pool_path), "NTUPLE_LATE_START_FRACTION": "0"}
    return start


def base_tag(path: Path) -> str:
    """A short name for a stage 1 network in run names: its run, and its games if the file is a games_<n> copy."""
    name = path.parent.name.removeprefix("methods-8x6-").removeprefix("smoke-methods-")
    games = path.stem.removeprefix("games_")
    if games.isdigit():
        number = int(games)
        name += f"-{number // 1_000_000}m" if number % 1_000_000 == 0 else f"-{number // 1000}k"
    return name


def run_stage2(plan: Plan, total: int | None, until: int | None, optimism: float, tc_games: int, tag: str | None, hold_until: int, stage_downgrade: int | None = None,
               tc_only: bool = False, keep_sums: bool = False, base: Path | None = None) -> None:
    if base is not None:
        plan.stage_base, plan.base_tag = base, base_tag(base)
    if stage_downgrade is not None and stage_downgrade != plan.stage_downgrade:
        plan.stage_downgrade = stage_downgrade
        tag = tag or f"dg{stage_downgrade}"
    if keep_sums and not (tc_only and plan.stage_downgrade == (2048 if plan.smoke else 16384)):
        raise SystemExit("--keep-sums needs --tc-only and --stage-downgrade 16384: stage 1's sums only fit a second stage that reads boards as stage 1 does")
    if tc_only:
        tag = f"{tag}-tc" if tag else "tc"
    if keep_sums:
        tag = f"{tag}-sums"
    if base is not None:
        tag = f"on-{plan.base_tag}-{tag}" if tag else f"on-{plan.base_tag}"
    run = Run(plan, stage2_arm(optimism, tag))
    if base is None and run.state.get("base") and Path(run.state["base"]) != BEST and not plan.smoke:
        raise SystemExit(f"{run.name} was built on {run.state['base']}: rerun with --base {run.state['base']}")
    total = total or run.state.get("total") or plan.stage2_total
    until = until or total
    recorded = run.state.get("stage_downgrade")
    if recorded is not None and recorded != plan.stage_downgrade:
        raise SystemExit(f"{run.name} was started with stage 2 downgrading from {recorded}: rerun with --stage-downgrade {recorded}")
    if recorded is None:
        run.state["stage_downgrade"] = plan.stage_downgrade
        run.save_state()
    start = prepare_stage2(plan, run, optimism, keep_sums)
    if tc_only and "start" not in run.state:  # stage 1's own recipe from the first game: no TD phase, no cuts
        run.state |= {"start": 0, "optimism": optimism, "rate": FINISH_RATE, "cut_at": [], "tc_from": 0, "total": total, "tc_games": total, "hold_until": 0, "downgrade": True}
        run.save_state()
    run_long(plan, total, until, optimism, tc_games, tag, hold_until, True, arm=run.arm, start_from=start)


def run_stage2_finish(plan: Plan, source_arm: str, rate: float, until: int | None) -> None:
    """Another TC finish for a stage2 run: its network where it switched to TC (td_end_<games>.npz), trained on with
    temporal coherence at `rate` from zeroed sums for the source's TC games, beside the source's own finish at FINISH_RATE."""
    source = Run(plan, source_arm)
    if not source.state_path.exists():
        raise SystemExit(f"{source.name} doesn't exist: --from takes a stage2 run's arm, e.g. stage2-opt1250-dg16384")
    run = Run(plan, f"{source_arm}-tc{rate:g}")
    if "start" not in run.state:
        def switched() -> bool:
            source.state = json.loads(source.state_path.read_text())
            return source.state.get("tc_from") is not None and (source.checkpoints / f"td_end_{source.state['tc_from']}.npz").exists()
        wait_for(plan, f"{source.name} to switch to TC", switched)
        tc_from = source.state["tc_from"]
        run.state |= {"start": tc_from, "from": source_arm, "start_file": str(source.checkpoints / f"td_end_{tc_from}.npz"), "rate": rate,
                      "total": tc_from + source.state["tc_games"], "stage_downgrade": source.state.get("stage_downgrade"), "optimism": source.state.get("optimism"),
                      "tc_from": tc_from}
        run.save_state()
    start = Path(run.state["start_file"])
    # Its reading settings from any network of its own line that's still on disk: its latest, where it branched, the source's start
    reference = next((path for path in (run.latest, start, source.checkpoints / "start.npz", source.latest) if path.exists()), None)
    if reference is None:
        raise SystemExit(f"None of {run.name}'s networks are on disk")
    with np.load(reference) as data:
        downgrade = int(json.loads(str(data["features"]))["downgrade_tile"])
    pool = Path(source.state.get("pool") or CHECKPOINTS / f"{plan.prefix}-stage2-pool" / f"first-{plan.stage_tile}.npy")
    plan.extra_env = {"NTUPLE_DOWNGRADE_TILE": str(downgrade), "NTUPLE_STAGE_TILE": str(plan.stage_tile), "NTUPLE_STAGE_DOWNGRADE_TILE": str(run.state["stage_downgrade"]),
                      "NTUPLE_START_POOL": str(pool), "NTUPLE_LATE_START_FRACTION": "0"}
    goal = run.state["total"] if until is None else min(until, run.state["total"])
    while (games := run.games()) < goal:
        target = min(goal, (games // plan.step + 1) * plan.step)
        run.train(target, rate, coherent=True, start_from=start)
        run.after_session(target)
    print(f"\n{run.name} is at {run.games():,} games (budget {run.state['total']:,}).")


def run_joint(plan: Plan, source_arm: str, added: int, until: int | None) -> None:
    """A finished stage2 run's network, both stages trained on together over whole games with the best recipe (TC at
    0.02, late starts), each weight carrying on from its own TC sums: stage 1's from the best network, stage 2's from the
    run's finish. Stage 2 now plays every board from 16384 up, so stage 1's tables no longer serve those boards too."""
    from deep_2048.agents.ntuple import NTupleNetwork
    from deep_2048.agents.ntuple.models import CheckpointMetadata

    source = Run(plan, source_arm)
    base = source.state.get("base")
    if base is None and source.state.get("from"):  # a stage2-finish branch: its stage 1 is the branched-from run's
        base = Run(plan, source.state["from"]).state.get("base")
    if not source.latest.exists() or base is None:
        raise SystemExit(f"{source.name} isn't a stage2 run with a network: --source-run takes one's arm, e.g. stage2-opt1250-dg16384-tc0.005")
    run = Run(plan, f"joint-{source_arm}")
    start = run.checkpoints / "start.npz"
    if "start" not in run.state:
        if source.games() < source.state.get("total", 0):
            raise SystemExit(f"{source.name} is at {source.games():,} of its {source.state['total']:,} games: let it finish first")
        base_path = Path(base)
        network, metadata = NTupleNetwork.load(source.latest)
        tables = sum(16 ** len(t) for t in network.tuples)
        with np.load(base_path) as base, np.load(source.latest) as staged:
            if not np.array_equal(base["weights"][:tables], network.weights[:tables]):
                raise SystemExit(f"{source.name}'s stage 1 isn't {base_path}'s network: it has been trained, so its sums can't be paired with it")
            extra = {}
            for name in ("coherence_errors", "coherence_abs_errors"):
                second = staged[name][tables:] if name in staged else np.zeros(network.weights.size - tables, dtype=np.float32)
                extra[name] = np.concatenate([base[name][:tables], second])
        run.require_space()
        start.parent.mkdir(parents=True, exist_ok=True)
        network.save(start, CheckpointMetadata(games_played=metadata.games_played, moves_played=metadata.moves_played, learning_rate=FINISH_RATE,
                                               temporal_coherence=True, optimizer="tc"), extra=extra)
        print(f"{start}: {source.latest} with stage 1's TC sums from {base_path} and stage 2's from its finish", flush=True)
        run.state |= {"start": metadata.games_played, "from": source_arm, "source_file": str(source.latest), "source_sha256": file_sha256(source.latest),
                      "base": str(base_path), "total": metadata.games_played + added, "rate": FINISH_RATE}
        run.save_state()
    with np.load(start) as data:
        features = json.loads(str(data["features"]))
    names = {"downgrade_tile": "NTUPLE_DOWNGRADE_TILE", "stage_tile": "NTUPLE_STAGE_TILE", "stage_downgrade_tile": "NTUPLE_STAGE_DOWNGRADE_TILE"}
    plan.extra_env = {variable: str(int(features[key])) for key, variable in names.items() if features.get(key) is not None}
    goal = run.state["total"] if until is None else min(until, run.state["total"])
    while (games := run.games()) < goal:
        target = min(goal, (games // plan.focus_eval + 1) * plan.focus_eval)
        run.train(target, FINISH_RATE, coherent=True, start_from=start)
        if target % plan.focus_eval == 0:
            run.evaluate(target)
    print(f"\n{run.name} is at {run.games():,} games (budget {run.state['total']:,}).")


def run_stage1_rate(plan: Plan, rate: float, source: Path | None, total: int | None, until: int | None) -> None:
    """best-long branched at a fixed network (default its 8M copy) and trained on to `total` games with TC at another base
    rate, its sums kept: the same recipe as best-long's continuation but for the rate, so the two compare at equal games."""
    from deep_2048.agents.ntuple import NTupleNetwork

    run = Run(plan, f"best-long-tc{rate:g}")
    if "start" not in run.state:
        source = source or Run(plan, "best-long").checkpoints / ("games_8000000.npz" if not plan.smoke else "latest.npz")
        if not source.exists():
            raise SystemExit(f"{source} is missing: pass --base, e.g. checkpoints/ntuple/methods-8x6-best-long/games_8000000.npz")
        _, metadata = NTupleNetwork.load(source)
        run.state |= {"start": metadata.games_played, "from": str(source), "base_sha256": file_sha256(source), "rate": rate,
                      "total": total or metadata.games_played + (2_000 if plan.smoke else 2_000_000)}
        run.save_state()
    elif total and total > run.state["total"]:
        run.state["total"] = total
        run.save_state()
    start = Path(run.state["from"])
    goal = run.state["total"] if until is None else min(until, run.state["total"])
    while (games := run.games()) < goal:
        target = min(goal, (games // plan.eval_every + 1) * plan.eval_every)
        run.train(target, run.state["rate"], coherent=True, start_from=start)
        run.after_session(target)
    print(f"\n{run.name} is at {run.games():,} games (budget {run.state['total']:,}).")


def run_stage1_scale(plan: Plan, rate: float, scale_rate: float, source: Path | None, total: int | None, until: int | None, below: int | None = None,
                     by_state: bool = False) -> None:
    """A stage 1 network with a large-tile count term added at zero (Yeh's feature: a weight per count of 2048s to
    32768s), with its own rate and no entry for boards without a 2048, trained on like stage1-rate. With `below`,
    boards holding that tile or more read no term; with `by_state`, the term is read from the board each move is made from."""
    from deep_2048.agents.ntuple import NTupleNetwork
    from deep_2048.agents.ntuple.models import CheckpointMetadata

    run = Run(plan, "stage1-scale" + ("-by-state" if by_state else "") + (f"-below{below}" if below else ""))
    start = run.checkpoints / "start.npz"
    if "start" not in run.state:
        source = source or Run(plan, "best-long-tc0.005").checkpoints / ("games_10000000.npz" if not plan.smoke else "latest.npz")
        if not source.exists():
            raise SystemExit(f"{source} is missing: pass --base")
        network, metadata = NTupleNetwork.load(source)
        if network.scale_feature or network.stage_tile is not None:
            raise SystemExit(f"{source} already has a scale term or a second stage")
        scaled = NTupleNetwork(network.tuples, downgrade_tile=network.downgrade_tile, scale_feature=True, scale_skip_empty=True, scale_rate=scale_rate,
                               scale_below_tile=below, scale_by_state=by_state)
        added = scaled.weights.size - network.weights.size
        scaled.weights[: network.weights.size] = network.weights
        with np.load(source) as data:
            extra = {name: np.concatenate([data[name], np.zeros(added, dtype=np.float32)]) for name in ("coherence_errors", "coherence_abs_errors") if name in data}
        run.require_space()
        start.parent.mkdir(parents=True, exist_ok=True)
        scaled.save(start, CheckpointMetadata(games_played=metadata.games_played, moves_played=metadata.moves_played, learning_rate=rate,
                                              temporal_coherence=True, optimizer="tc"), extra=extra or None)
        print(f"{start}: {source} with a {added:,}-weight large-tile count term at zero (its rate {scale_rate:g} of the tuples', none for boards "
              "without a 2048" + (f" or with a {below} or more" if below else "") + ")" + (", read from the board each move is made from" if by_state else "")
              + (", the tuples' TC sums kept" if extra else ""), flush=True)
        run.state |= {"start": metadata.games_played, "from": str(source), "base_sha256": file_sha256(source), "rate": rate, "scale_rate": scale_rate,
                      "scale_below": below, "scale_by_state": by_state,
                      "total": total or metadata.games_played + (2_000 if plan.smoke else 2_000_000)}
        run.save_state()
    elif total and total > run.state["total"]:
        run.state["total"] = total
        run.save_state()
    plan.extra_env = {"NTUPLE_SCALE_FEATURE": "true", "NTUPLE_SCALE_SKIP_EMPTY": "true", "NTUPLE_SCALE_RATE": repr(float(run.state["scale_rate"]))} \
        | ({"NTUPLE_SCALE_BELOW_TILE": str(run.state["scale_below"])} if run.state.get("scale_below") else {}) \
        | ({"NTUPLE_SCALE_BY_STATE": "true"} if run.state.get("scale_by_state") else {})
    goal = run.state["total"] if until is None else min(until, run.state["total"])
    while (games := run.games()) < goal:
        target = min(goal, (games // plan.eval_every + 1) * plan.eval_every)
        run.train(target, run.state["rate"], coherent=True, start_from=start)
        run.after_session(target)
    print(f"\n{run.name} is at {run.games():,} games (budget {run.state['total']:,}).")


SCRATCH_SCHEDULE = ((1_500_000, 0.02, False), (8_000_000, 0.02, True), (10_000_000, 0.005, False))  # (to games, rate, reset the sums at its start)
SCRATCH_KEEP = (2_000_000, 5_000_000, 8_000_000, 10_000_000)  # copies kept for 5,000-game comparisons with the best network's own path


def run_scratch_scale(plan: Plan, scale_rate: float, until: int | None, by_state: bool = False) -> None:
    """The best network's whole path from zero (8x6, TC at 0.02 with downgrading and late starts; the TC sums reset at
    1.5M; on at 0.02 to 8M; then 0.005 to 10M), with the large-tile count term trained from the first game: a weight per
    count of 2048s to 32768s, none for boards without a 2048, at `scale_rate` of the tuples' rate."""
    smoke = plan.smoke
    schedule = tuple((games // 1000, rate, reset) for games, rate, reset in SCRATCH_SCHEDULE) if smoke else SCRATCH_SCHEDULE
    keep = tuple(games // 1000 for games in SCRATCH_KEEP) if smoke else SCRATCH_KEEP
    step = plan.step if not smoke else 500
    run = Run(plan, "scratch-scale" + ("-by-state" if by_state else "") + ("" if scale_rate == 1 / 64 else f"-r{scale_rate:g}"))
    if "start" not in run.state:
        run.state |= {"start": 0, "scale_rate": scale_rate, "scale_by_state": by_state, "schedule": [list(phase) for phase in schedule], "total": schedule[-1][0],
                      "resets": []}
        run.save_state()
    plan.extra_env = {"NTUPLE_SCALE_FEATURE": "true", "NTUPLE_SCALE_SKIP_EMPTY": "true", "NTUPLE_SCALE_RATE": repr(float(run.state["scale_rate"]))} \
        | ({"NTUPLE_SCALE_BY_STATE": "true"} if run.state.get("scale_by_state") else {})
    goal = run.state["total"] if until is None else min(until, run.state["total"])
    while (games := run.games()) < goal:
        phase_start = 0
        for end, rate, reset in run.state["schedule"]:
            if games < end:
                break
            phase_start = end
        reset_now = reset and games == phase_start and phase_start not in run.state["resets"]
        target = min(goal, end, (games // step + 1) * step)
        if reset_now:
            print(f"{run.name}: resetting the temporal coherence sums at {games:,} games, as the best network's path did", flush=True)
        base_path = Path(run.state["base"]) if run.state.get("base") else None  # only read before the run has a latest.npz
        run.train(target, rate, coherent=True, start_from=base_path, reset_coherence=reset_now, reset_own=reset_now)
        if reset_now:
            run.state["resets"].append(phase_start)
            run.save_state()
        if target in keep and not (run.checkpoints / f"games_{target}.npz").exists():
            run.keep_copy(f"games_{target}.npz")
        if target % plan.eval_every == 0:
            run.evaluate(target)
    print(f"\n{run.name} is at {run.games():,} games (budget {run.state['total']:,}).")


STALL_PHASES = ((0.02, False), (0.02, True), (0.005, False), (0.00125, False))  # (TC base rate, reset the sums as it starts)
STALL_FLOORS = (1_500_000, 1_000_000, 1_000_000, 1_000_000)  # games each phase runs before a stall is acted on (the best path reset at 1.5M)
SECOND_RESET_PHASES = ((0.02, True), (0.005, False))  # scratch-stall's 4.6M network reset again, then cut (its 0.00125 phase added nothing)
SECOND_RESET_FLOORS = (500_000, 500_000)
SECOND_RESET_BASE = CHECKPOINTS / "methods-8x6-scratch-stall" / "stall_4600000.npz"


def run_scratch_stall(plan: Plan, total: int, until: int | None, scale_rate: float | None = None, base: Path | None = None, network: str | None = None, tag: str | None = None) -> None:
    """8x6 from zero with the long run's recipe (TC, downgrading from 16384, late starts), moving through STALL_PHASES at
    stalls instead of at fixed games: at least a long window into each phase, checked every step, a stall being under
    the long-run thresholds in both the score and the 32768 rate over the window. A stall in the last phase ends the
    run. With `scale_rate`, the large-tile count term trains from the first game as in scratch-scale. With `base`
    (second-reset), it carries that network on through SECOND_RESET_PHASES instead, its sums reset as it starts. With
    `base` and `network` (a network grown by scripts/add_tuples.py from scratch-stall's first stall), it carries on
    scratch-stall's own schedule from its second phase. `tag` is added to the run's name."""
    if network is not None:
        plan.network, plan.reads = network, len(NETWORKS[network]) * SYMMETRIES
    grown = base is not None and network is not None
    suffix = "" if tag is None else f"-{tag}"
    if base is not None:
        run = Run(plan, (f"grown-{network}" if grown else "second-reset") + suffix)
    else:
        run = Run(plan, "scratch-stall" + ("" if network is None else f"-{network}") + ("" if scale_rate is None else "-scale" + ("" if scale_rate == 1 / 64 else f"-r{scale_rate:g}")) + suffix)
    if "start" not in run.state:
        start = json.loads(str(np.load(base)["metadata"]))["games_played"] if base is not None else 0
        phases = STALL_PHASES[1:] if grown else SECOND_RESET_PHASES if base is not None else STALL_PHASES
        if network is not None:  # the same step a board as the 8x6
            phases = tuple((rate * len(NETWORKS["8x6"]) * SYMMETRIES / plan.reads, reset) for rate, reset in phases)
        run.state |= {"start": start, "total": start + total if base is not None else total, "phase": 0, "phase_starts": [start],
                      "phases": [list(phase) for phase in phases], "scale_rate": scale_rate, "ended_by": None, "base": str(base) if base is not None else None}
        if base is not None:
            run.state |= {"floors": [int(floor * plan.long_window / 500_000) for floor in (STALL_FLOORS[1:] if grown else SECOND_RESET_FLOORS)], "would_have": {}}
        run.save_state()
    if "floors" not in run.state:  # a state saved before the floors existed gets them too
        scale = plan.long_window / 500_000  # the smoke test's floors shrink with its window
        run.state |= {"floors": [int(floor * scale) for floor in STALL_FLOORS], "would_have": {}}
        run.save_state()
    if run.state["scale_rate"] is not None:
        plan.extra_env = {"NTUPLE_SCALE_FEATURE": "true", "NTUPLE_SCALE_SKIP_EMPTY": "true", "NTUPLE_SCALE_RATE": repr(float(run.state["scale_rate"]))}
    goal = run.state["total"] if until is None else min(until, run.state["total"])
    while (games := run.games()) < goal and run.state["ended_by"] is None:
        phase = run.state["phase"]
        rate, reset = run.state["phases"][phase]
        since = run.state["phase_starts"][phase]
        reset_now = reset and games == since and since not in run.state.get("resets", [])
        target = min(goal, (games // plan.step + 1) * plan.step)
        if reset_now:
            print(f"{run.name}: resetting the temporal coherence sums at {games:,} games", flush=True)
        base_path = Path(run.state["base"]) if run.state.get("base") else None  # only read before the run has a latest.npz
        run.train(target, rate, coherent=True, start_from=base_path, reset_coherence=reset_now, reset_own=reset_now)
        if reset_now:
            run.state.setdefault("resets", []).append(since)
            run.save_state()
        if target % plan.eval_every == 0:
            run.evaluate(target)
        stalled = target < goal and run.stalled_long(target, since)
        if stalled and target - since < run.state["floors"][phase]:
            # The criteria are met but the phase hasn't run its floor: note the first such point, keep a copy, carry on
            if str(phase) not in run.state["would_have"]:
                run.state["would_have"][str(phase)] = target
                run.keep_copy(f"would_stall_{target}.npz")
                run.save_state()
            print(f"{run.name}: the stall criteria are met at {target:,} games, but phase {phase + 1} runs at least {run.state['floors'][phase]:,} "
                  f"games (to {since + run.state['floors'][phase]:,}): carrying on. It would have moved on at {run.state['would_have'][str(phase)]:,}.", flush=True)
            stalled = False
        if stalled:
            run.keep_copy(f"stall_{target}.npz")
            if phase + 1 == len(run.state["phases"]):
                run.state["ended_by"] = f"stall at {target:,} in the last phase"
                print(f"{run.name}: stalled at {target:,} games in its last phase: done", flush=True)
            else:
                run.state["phase"] = phase + 1
                run.state["phase_starts"].append(target)
                next_rate, next_reset = run.state["phases"][phase + 1]
                print(f"{run.name}: stalled at {target:,} games: phase {phase + 2} of {len(run.state['phases'])}, TC at {next_rate:g}"
                      + (" with the sums reset" if next_reset else ""), flush=True)
            run.save_state()
    if run.games() >= run.state["total"] and not (run.checkpoints / f"games_{run.games()}.npz").exists():
        run.keep_copy(f"games_{run.games()}.npz")
    print(f"\n{run.name} is at {run.games():,} games (budget {run.state['total']:,}; phase {run.state['phase'] + 1} of {len(run.state['phases'])}"
          + (f"; ended by a {run.state['ended_by']}" if run.state["ended_by"] else "") + ").")
    print("  phase starts: " + ", ".join(f"{start:,}" for start in run.state["phase_starts"]))
    if run.state["would_have"]:
        print("  without the floors it would have moved on at: "
              + ", ".join(f"phase {int(phase) + 1} at {games:,}" for phase, games in sorted(run.state["would_have"].items())))


def goal_arm(value: float) -> str:
    return f"stage1-goal{value / 1000:g}k" if value % 1000 == 0 else f"stage1-goal{value:g}"


def run_stage1_goal(plan: Plan, value: float, base: Path | None, added: int, until: int | None) -> None:
    """A tailored stage 1: a network trained on with TC (its sums kept) on whole games from an empty board, each ending on
    the move that makes 16384, worth `value` after its reward. It learns to build 16384, scoring on the way, and nothing past it."""
    from deep_2048.agents.ntuple import NTupleNetwork
    from deep_2048.agents.ntuple.models import CheckpointMetadata

    run = Run(plan, goal_arm(value))
    start = run.checkpoints / "start.npz"
    goal_tile = 1024 if plan.smoke else 16384
    if "start" not in run.state:
        base = base or (Run(plan, "best-long").checkpoints / "games_5000000.npz" if not plan.smoke else Run(plan, "best-long").latest)
        if not base.exists():
            raise SystemExit(f"{base} is missing: pass --base, e.g. checkpoints/ntuple/methods-8x6-best-long/latest.npz")
        network, metadata = NTupleNetwork.load(base)
        if network.stage_tile is not None:
            raise SystemExit(f"{base} has a second stage: the tailored stage 1 starts from a single-stage network")
        # The same weights read without downgrading: boards holding the goal tile get patterns of their own, which learn the bonus
        tailored = NTupleNetwork(network.tuples, network.weights, downgrade_tile=None, scale_feature=network.scale_feature,
                                 scale_when_downgraded=network.scale_when_downgraded)
        with np.load(base) as data:
            extra = {name: data[name] for name in ("coherence_errors", "coherence_abs_errors") if name in data}
        run.require_space()
        tailored.save(start, CheckpointMetadata(games_played=metadata.games_played, moves_played=metadata.moves_played, learning_rate=FINISH_RATE,
                                                temporal_coherence=True, optimizer="tc"), extra=extra or None)
        print(f"{start}: {base} without downgrading" + (", its TC sums kept" if extra else ", fresh TC sums"), flush=True)
        run.state |= {"start": metadata.games_played, "from": str(base), "base_sha256": file_sha256(base), "goal_tile": goal_tile, "goal_value": value,
                      "total": metadata.games_played + added, "rate": FINISH_RATE}
        run.save_state()
    elif run.state["start"] + added > run.state["total"]:  # a larger --total extends a finished run
        print(f"{run.name}: budget raised from {run.state['total']:,} to {run.state['start'] + added:,} games", flush=True)
        run.state["total"] = run.state["start"] + added
        run.save_state()
    plan.downgrade = False
    plan.extra_env = {"NTUPLE_GOAL_TILE": str(run.state["goal_tile"]), "NTUPLE_GOAL_VALUE": repr(float(run.state["goal_value"])), "NTUPLE_LATE_START_FRACTION": "0"}
    goal = run.state["total"] if until is None else min(until, run.state["total"])
    while (games := run.games()) < goal:
        target = min(goal, (games // plan.focus_eval + 1) * plan.focus_eval)
        run.train(target, FINISH_RATE, coherent=True, start_from=start)
        if target % plan.focus_eval == 0:
            run.evaluate(target)
    print(f"\n{run.name} is at {run.games():,} games (budget {run.state['total']:,}).")


def run_stage1_focus(plan: Plan, added: int, until: int | None) -> None:
    """The best network trained on with its own recipe, every game starting from one of its own first-4096 boards."""
    from deep_2048.agents.ntuple import NTupleNetwork
    from deep_2048.agents.ntuple.stages import first_boards

    run = Run(plan, "stage1-focus")
    start = BEST if not plan.smoke else Run(plan, "best-long").latest
    pool_path = CHECKPOINTS / f"{plan.prefix}-stage1-focus-pool" / f"first-{plan.focus_tile}.npy"
    if not pool_path.exists() or "start" not in run.state:
        if not start.exists():
            raise SystemExit(f"{start} is missing")
        base, metadata = NTupleNetwork.load(start)
        if not pool_path.exists():
            pool_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Building the pool: {plan.focus_pool_games:,} greedy games of {start}, each one's first board holding {plan.focus_tile}...", flush=True)
            boards, finals = first_boards(base, plan.focus_tile, plan.focus_pool_games, 91_000_000, int(os.environ.get("THREADS", "8")))
            np.save(pool_path, boards)
            (pool_path.parent / "source.txt").write_text(f"{start}\nsha256 {file_sha256(start)}\n{plan.focus_pool_games} games from seed 91,000,000: "
                                                           f"{boards.size} reached {plan.focus_tile}; mean score {finals.mean():,.0f}\n")
            print(f"  {boards.size:,} boards ({boards.size / plan.focus_pool_games:.1%} of games), mean score {finals.mean():,.0f}: {pool_path}", flush=True)
        if "start" not in run.state:
            run.state |= {"start": metadata.games_played, "from": str(start), "base_sha256": file_sha256(start), "pool": str(pool_path),
                          "total": metadata.games_played + added, "rate": FINISH_RATE}
            run.save_state()
    plan.extra_env = {"NTUPLE_START_POOL": str(pool_path), "NTUPLE_LATE_START_FRACTION": "0"}
    goal = run.state["total"] if until is None else min(until, run.state["total"])
    while (games := run.games()) < goal:
        target = min(goal, (games // plan.focus_eval + 1) * plan.focus_eval)
        run.train(target, FINISH_RATE, coherent=True, start_from=start, reset_coherence=True)
        if target % plan.focus_eval == 0:
            run.evaluate(target)
    print(f"\n{run.name} is at {run.games():,} games (budget {run.state['total']:,}).")


def run_best_long(plan: Plan, total: int, until: int) -> None:
    """The best network so far, trained on with its own recipe (TC at 0.02, sums kept) to `total` games."""
    run = Run(plan, "best-long")
    start = BEST if not plan.smoke else Run(plan, "td-guei-tc").latest
    if "start" not in run.state:
        if not start.exists():
            raise SystemExit(f"{start} is missing")
        run.state |= {"start": 2_000_000 if not plan.smoke else Run(plan, "td-guei-tc").games(), "from": "decision-8x6-tc-reset", "total": total}
        run.save_state()
    elif total > run.state["total"]:
        # Extending a finished budget: keep the network where it stood, so anything built on it (stage2 --base) stays fixed
        games = run.games()
        if run.latest.exists() and not (run.checkpoints / f"games_{games}.npz").exists():
            run.keep_copy(f"games_{games}.npz")
        print(f"{run.name}: budget raised from {run.state['total']:,} to {total:,} games", flush=True)
        run.state["total"] = total
        run.save_state()
    while (games := run.games()) < min(until, run.state["total"]):
        target = min(until, run.state["total"], (games // plan.eval_every + 1) * plan.eval_every)
        run.train(target, FINISH_RATE, coherent=True, start_from=start)
        run.after_session(target)
    print(f"\n{run.name} is at {run.games():,} games (budget {run.state['total']:,}).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one arm of the methods batch")
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("--from", dest="source", choices=tuple(OPENINGS), help="td-yeh and td-guei-tc: the opening to branch from")
    parser.add_argument("--now", action="store_true", help="td-yeh: branch from the opening as it stands, without waiting for the stall rule (the opening must be stopped)")
    parser.add_argument("--until", type=int, help="Stop at this many games in total (default: the whole run)")
    parser.add_argument("--smoke", action="store_true", help="A few thousand games on a small network, to check the machinery")
    parser.add_argument("--optimism", type=float, default=100.0, help="td-opt: the value every weight starts at (default 100)")
    parser.add_argument("--total", type=int, help="td-opt: the most games in total (default 8M); best-long: default 5M; stage1-focus: the games to add (default 2M)")
    parser.add_argument("--hold-until", type=int, help="td-opt: keep the opening rate to this many games, with no stall checks before it")
    parser.add_argument("--no-downgrade", action="store_true", help="td-opt: train without downgrading boards from 16384 up")
    parser.add_argument("--tag", help="td-opt and stage2: a variant's name, added to its run's name")
    parser.add_argument("--tc-only", action="store_true", help="stage2: train with temporal coherence at 0.02 from the first game (stage 1's own recipe), to --total")
    parser.add_argument("--base", type=Path, help="stage2 and stage1-goal: the stage 1 network to build on (default: the best, decision-8x6-tc-reset at 2M), e.g. "
                        "checkpoints/ntuple/methods-8x6-best-long/games_5000000.npz; it gets its own pool of first-16384 boards, and the run's name says on-<network>")
    parser.add_argument("--scale-rate", type=float, default=1 / 64, help="stage1-scale: the large-tile count term's rate as a share of the tuples' (default 1/64)")
    parser.add_argument("--scale-below", type=int, help="stage1-scale: boards holding this tile or more read no count term, e.g. 16384 (run stage1-scale-below<tile>)")
    parser.add_argument("--scale-by-state", action="store_true", help="stage1-scale: read the count term from the board each move is made from (run stage1-scale-by-state)")
    parser.add_argument("--with-scale", action="store_true", help="scratch-stall: with the large-tile count term from the first game (run scratch-stall-scale)")
    parser.add_argument("--network", choices=tuple(NETWORKS), help="scratch-stall: another network on the same schedule, at the 8x6's step a board (run scratch-stall-<network>)")
    parser.add_argument("--goal-value", type=float, help="stage1-goal: what making 16384 is worth (a fixed bonus standing in for stage 2), e.g. 160000")
    parser.add_argument("--keep-sums", action="store_true", help="stage2 with --tc-only --stage-downgrade 16384: stage 2 starts with a copy of stage 1's TC sums as well as its weights")
    parser.add_argument("--stage-downgrade", type=int, help="stage2: the tile stage 2 downgrades from (default 32768; 16384 makes its start an exact copy of stage 1, tagged dg16384)")
    parser.add_argument("--source-run", help="stage2-finish and joint: the stage2 run to start from, by arm (e.g. stage2-opt1250-dg16384)")
    parser.add_argument("--rate", type=float, help="stage2-finish and stage1-rate: the TC base rate a weight (the runs' own is 0.02)")
    parser.add_argument("--tc-games", type=int, help="td-opt: games of the TC finish (default 2M)")
    args = parser.parse_args()
    plan = Plan(args.smoke)
    limit = min(args.until or plan.total, plan.total)
    os.chdir(ROOT)

    if args.arm == "td-opt":
        total = args.total or Run(plan, long_arm(args.optimism, args.tag)).state.get("total") or plan.long_total
        run_long(plan, total, args.until or total, args.optimism, args.tc_games or plan.tc_games, args.tag, args.hold_until or 0, not args.no_downgrade)
        return
    if args.arm == "stage2":
        hold = plan.stage2_hold if args.hold_until is None else args.hold_until
        # The budget is looked up in run_stage2, once the run's full name (with its dg/tc/sums tags) is known
        run_stage2(plan, args.total, args.until, args.optimism, args.tc_games or plan.tc_games, args.tag, hold, args.stage_downgrade, args.tc_only, args.keep_sums, args.base)
        return
    if args.arm == "stage2-finish":
        if not args.source_run or not args.rate:
            parser.error("stage2-finish needs --source-run and --rate")
        run_stage2_finish(plan, args.source_run, args.rate, args.until)
        return
    if args.arm == "scratch-stall":
        run_scratch_stall(plan, args.total or (6_000 if plan.smoke else 10_000_000), args.until, args.scale_rate if args.with_scale else None, network=args.network, tag=args.tag)
        return
    if args.arm == "second-reset":
        run_scratch_stall(plan, args.total or (6_000 if plan.smoke else 3_000_000), args.until, base=args.base or SECOND_RESET_BASE, network=args.network, tag=args.tag)
        return
    if args.arm == "scratch-scale":
        run_scratch_scale(plan, args.scale_rate, args.until, args.scale_by_state)
        return
    if args.arm == "stage1-scale":
        run_stage1_scale(plan, args.rate or 0.005, args.scale_rate, args.base, args.total, args.until, args.scale_below, args.scale_by_state)
        return
    if args.arm == "stage1-rate":
        if not args.rate:
            parser.error("stage1-rate needs --rate: the TC base rate a weight, e.g. 0.005 (best-long uses 0.02)")
        run_stage1_rate(plan, args.rate, args.base, args.total, args.until)
        return
    if args.arm == "stage1-goal":
        if args.goal_value is None:
            parser.error("stage1-goal needs --goal-value: what making 16384 is worth, e.g. 160000")
        run_stage1_goal(plan, args.goal_value, args.base, args.total or plan.focus_games, args.until)  # --total: games since its start
        return
    if args.arm == "joint":
        if not args.source_run:
            parser.error("joint needs --source-run: the finished stage2 run to train on, e.g. stage2-opt1250-dg16384-tc0.005")
        run_joint(plan, args.source_run, args.total or plan.focus_games, args.until)
        return
    if args.arm == "stage1-focus":
        run_stage1_focus(plan, args.total or plan.focus_games, args.until)
        return
    if args.arm == "best-long":
        total = args.total or Run(plan, args.arm).state.get("total") or (12_000 if plan.smoke else 5_000_000)  # a larger --total extends a finished run
        run_best_long(plan, total, args.until or total)
        return
    elif args.arm in TC_ARMS:
        run_tc(plan, args.arm, limit)
    elif args.arm in OPENINGS:
        run_opening(plan, args.arm, limit)
    elif args.arm == "td-yeh-tc":
        run_finish(plan, args.arm, "td-yeh", limit)
    else:
        recorded = Run(plan, args.arm).state.get("from")
        source = args.source or recorded
        if source is None:
            parser.error(f"{args.arm} needs --from td-0.1 or --from td-0.5: the opening it branches from")
        if recorded not in (None, source):
            parser.error(f"{args.arm} was started from {recorded}")
        if args.arm == "td-yeh":
            run_yeh(plan, source, limit, args.now)
        else:
            run_finish(plan, args.arm, source, limit)
    print(f"\n{plan.prefix}-{args.arm} is at {limit:,} games.")


if __name__ == "__main__":
    main()
