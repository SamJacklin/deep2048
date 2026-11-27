"""Chart training runs against the best system so far, with tick boxes to choose what is shown.

Usage examples:
    uv run python scripts/watch_against_best.py                              # every optimistic-8x6* run, live window
    uv run python scripts/watch_against_best.py --only optimistic-8x6-500 optimistic-8x6-1000
    uv run python scripts/watch_against_best.py --reference continued --tiles 32768 --zoom
    uv run python scripts/watch_against_best.py --once                       # save the chart and exit

Four panels: training score by games, training score by moves played, the share of games reaching each large tile,
and training speed. The boxes on the left switch each run on and off, choose which version of the best system is
drawn (the long run with round 1's reset at 1.5M, in black, or the same run continued with no reset, in grey),
choose which tiles the top-right panel shows, and zoom the score panels onto the top of the curves. Above the panels,
a line for each ticked run gives its score, how it compares with the ticked version of the best system at the same
number of games and at the same number of moves, how much it has gained over its last 100k and 250k games, and the
share of its recent training games reaching 8192, 16384 and 32768. The ticked versions of the best system get lines
too, at the point where their training ended. The options
above only set which boxes start ticked. The window redraws as new rows land in the histories and the choices stay
as they are; the chart is also saved as against_best.png in the first run's folder.

Scores are the trainer's own batch averages, smoothed over --smooth batches, not the seeded greedy evaluations.
Moves are the fairer measure of experience when one run's games are much shorter than another's. The best system's
histories were recorded before the trainer logged the 32768 rate, so it has no 32768 line.
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import CheckButtons

sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_2048.agents.ntuple.train import HISTORY_NAME  # noqa: E402
from deep_2048.experiments import EXPERIMENTS_DIR  # noqa: E402

TILES = {8192: ":", 16384: "-", 32768: "-."}
STYLE_NAMES = {":": "dotted", "-": "solid", "-.": "dash-dot"}
TILES_BY_LABEL = {f"Tile {tile}": tile for tile in TILES}
RUN_COLOURS = ("steelblue", "darkorange", "seagreen", "firebrick", "purple", "saddlebrown", "deeppink", "olive")
REFERENCES = {  # label: (the runs that make it up, colour)
    "Best: reset at 1.5M": (("long-8x6", "decision-8x6-tc-reset"), "black"),
    "Best run, no reset": (("long-8x6", "decision-8x6-control"), "gray"),
}
ZOOM = "Zoom scores to the top"
ROW_HEIGHT = 0.021


def load_history(path: Path) -> np.ndarray | None:
    """Return the training history, or None if it's missing, empty or the trainer is mid-write."""
    if not path.exists():
        return None
    try:
        history = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    except ValueError:
        return None
    return history if history.size else None


def columns_of(history: np.ndarray) -> dict[str, np.ndarray]:
    return {column: history[column] for column in history.dtype.names}


def load_joined(names: tuple[str, ...]) -> dict[str, np.ndarray] | None:
    """Several runs' histories joined end to end (later runs carry on the earlier one's counts), or None if any is missing."""
    parts = [load_history(EXPERIMENTS_DIR / name / HISTORY_NAME) for name in names]
    if any(part is None for part in parts):
        return None
    shared = set.intersection(*(set(part.dtype.names) for part in parts))
    return {column: np.concatenate([part[column] for part in parts]) for column in shared}


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """A trailing mean over up to `window` batches."""
    if window <= 1 or values.size < 2:
        return values
    sums = np.cumsum(np.insert(values.astype(np.float64), 0, 0.0))
    index = np.arange(1, values.size + 1)
    start = np.maximum(index - window, 0)
    return (sums[index] - sums[start]) / (index - start)


def at(x: np.ndarray, y: np.ndarray, where: float) -> float | None:
    """y where x first reaches `where`, or None if it never does."""
    reached = np.searchsorted(x, where)
    return float(y[reached]) if reached < x.size else None


def signed(value: float | None) -> str:
    return "     –" if value is None else f"{value / 1e3:+6.1f}k"


def start_value(name: str) -> int | None:
    """The starting value of an optimistic run, from its name (the first run, at 5,000, has no number)."""
    match = re.fullmatch(r"optimistic-8x6(?:-(\d+))?(?:-(.+))?", name)
    return (int(match.group(1)) if match.group(1) else 5000) if match else None


def display_name(name: str) -> str:
    """Start 500 for optimistic-8x6-500, and Start 500 reset for a branch of it named optimistic-8x6-500-reset."""
    value = start_value(name)
    if value is None:
        return name
    branch = re.fullmatch(r"optimistic-8x6(?:-\d+)?(?:-(.+))?", name).group(1)
    return f"Start {value:,}" + (f" {branch}" if branch else "")


def discover_runs() -> list[str]:
    """Every optimistic-8x6* run with a history, largest starting value first and each run's branches after it (so
    the first runs keep their colours as others are added)."""
    names = [path.parent.name for path in EXPERIMENTS_DIR.glob(f"optimistic-8x6*/{HISTORY_NAME}") if start_value(path.parent.name) is not None]
    plain = sorted((name for name in names if display_name(name) == f"Start {start_value(name):,}"), key=lambda name: -start_value(name))
    return plain + sorted((name for name in names if name not in plain), key=lambda name: (-start_value(name), name))


class Chart:
    def __init__(self, runs: list[str], window: int, visible: set[str], tiles: set[int], zoom: bool) -> None:
        self.runs, self.window = runs, window
        self.colours = dict(zip(runs, RUN_COLOURS))
        self.histories: dict[str, dict[str, np.ndarray]] = {}
        self.references = {label: joined for label, (names, _) in REFERENCES.items() if (joined := load_joined(names)) is not None}
        self.fig = plt.figure(figsize=(16.5, 9.5))
        # Room above the panels for the summary: two heading lines, a line for each version of the best system and a line a run
        grid = self.fig.add_gridspec(2, 3, width_ratios=(0.48, 1, 1), left=0.01, right=0.945, top=0.925 - ROW_HEIGHT * (len(runs) + 4), bottom=0.07, wspace=0.22, hspace=0.3)
        self.summary: list = []
        self.games_ax, self.tiles_ax = self.fig.add_subplot(grid[0, 1]), self.fig.add_subplot(grid[0, 2])
        self.moves_ax, self.speed_ax = self.fig.add_subplot(grid[1, 1]), self.fig.add_subplot(grid[1, 2])
        self.length_ax = self.speed_ax.twinx()
        panel = self.fig.add_subplot(grid[:, 0])
        panel.set_axis_off()
        panel.set_title("Show", loc="left", fontweight="bold")
        self.items = runs + list(self.references) + [f"Tile {tile}" for tile in TILES] + [ZOOM]
        active = [name in visible for name in runs] + [label in visible for label in self.references] + [tile in tiles for tile in TILES] + [zoom]
        self.boxes = CheckButtons(panel, [self.label(item) for item in self.items], active)
        for item, text in zip(self.items, self.boxes.labels):
            text.set_fontsize(9)
            text.set_color(self.colours.get(item) or (REFERENCES[item][1] if item in REFERENCES else "black"))
        self.boxes.on_clicked(lambda _: self.draw())

    def label(self, item: str) -> str:
        """A box's text: runs carry their latest games and smoothed score."""
        if item in self.histories:
            history = self.histories[item]
            return f"{display_name(item)}: {history['games'][-1] / 1e6:.2f}M games, {smooth(history['avg_score'], self.window)[-1] / 1e3:,.0f}k"
        if item in self.runs:
            return f"{display_name(item)}: no rows yet"
        if item in TILES_BY_LABEL:
            return f"{item} ({STYLE_NAMES[TILES[TILES_BY_LABEL[item]]]})"
        return item

    def summarise(self, shown: dict[str, bool]) -> None:
        """The lines above the panels: each ticked run's score, how it compares with the ticked version of the best
        system at the same games and at the same moves, and how much it has gained over its last 100k and 250k games."""
        for text in self.summary:
            text.remove()
        against = next((label for label in self.references if shown[label]), None)
        heading = f"Against {against}, at the same number of games and of moves; gains are over each run's own last 100k and 250k games" if against else "No version of the best system is ticked; gains are over each run's own last 100k and 250k games"
        rows = [("black", heading), ("black", f"{'':<15}{'games':>7}{'score':>9}{'same games':>13}{'same moves':>13}{'last 100k':>12}{'last 250k':>12}" + "".join(f"{'≥' + str(tile):>9}" for tile in TILES))]

        def row(colour: str, name: str, history: dict[str, np.ndarray], compare: bool) -> None:
            games, score = history["games"], smooth(history["avg_score"], self.window)
            same_games = same_moves = None
            if compare and against:
                reference = self.references[against]
                reference_score = smooth(reference["avg_score"], self.window)
                same_games, same_moves = at(reference["games"], reference_score, games[-1]), at(reference["moves"], reference_score, history["moves"][-1])
            gains = [score[-1] - at(games, score, games[-1] - span) if games[-1] - span >= games[0] else None for span in (100_000, 250_000)]
            rates = "".join(f"{smooth(history[f'rate_{tile}'], self.window)[-1]:>8.1f}%" if f"rate_{tile}" in history else f"{'–':>9}" for tile in TILES)
            rows.append((colour, f"{name:<15}{games[-1] / 1e6:>6.2f}M{score[-1] / 1e3:>8.1f}k"
                         f"{signed(None if same_games is None else score[-1] - same_games):>13}{signed(None if same_moves is None else score[-1] - same_moves):>13}{signed(gains[0]):>12}{signed(gains[1]):>12}{rates}"))

        for label in self.references:
            if shown[label]:
                row(REFERENCES[label][1], {"Best: reset at 1.5M": "Best, reset", "Best run, no reset": "Best, no reset"}.get(label, label), self.references[label], False)
        for name in self.runs:
            if shown[name] and name in self.histories:
                row(self.colours[name], display_name(name), self.histories[name], True)
        self.summary = [self.fig.text(0.24, 0.975 - ROW_HEIGHT * line, text, family="monospace", fontsize=9, color=colour, fontweight="bold" if line < 2 else "normal")
                        for line, (colour, text) in enumerate(rows)]

    def shown(self) -> dict[str, bool]:
        return dict(zip(self.items, self.boxes.get_status()))

    def draw(self) -> None:
        shown = self.shown()
        for item, text in zip(self.items, self.boxes.labels):
            text.set_text(self.label(item))
        # (label, columns, colour, line width), references first so the runs are drawn over them
        lines = [(label, self.references[label], REFERENCES[label][1], 1.5) for label in reversed(list(self.references)) if shown[label]]
        lines += [(display_name(name), self.histories[name], self.colours[name], 1.8) for name in self.runs if shown[name] and name in self.histories]
        for ax in (self.games_ax, self.tiles_ax, self.moves_ax, self.speed_ax, self.length_ax):
            ax.clear()
        self.summarise(shown)

        top = 0.0
        for ax, column, scale, label, title in ((self.games_ax, "games", 1e6, "Million games played", "Training score by games"),
                                                (self.moves_ax, "moves", 1e9, "Billion moves played (experience)", "Training score by experience")):
            for _, history, colour, width in lines:
                score = smooth(history["avg_score"], self.window)
                ax.plot(history[column] / scale, score, color=colour, linewidth=width)
                top = max(top, score.max())
            ax.set_title(title)
            ax.set_xlabel(label)
            ax.set_ylabel(f"Average score (mean of {self.window} batches)" if self.window > 1 else "Average score")
            ax.set_xlim(left=0)
            ax.grid(True, alpha=0.3)
        for ax in (self.games_ax, self.moves_ax):
            ax.set_ylim((0.8 * top, 1.02 * top) if shown[ZOOM] and top > 0 else (0, None))

        highest = 0.0
        tiles = [tile for tile in TILES if shown[f"Tile {tile}"]]
        for tile in tiles:
            for _, history, colour, width in lines:
                if f"rate_{tile}" in history:
                    rate = smooth(history[f"rate_{tile}"], self.window)
                    self.tiles_ax.plot(history["games"] / 1e6, rate, color=colour, linewidth=width, linestyle=TILES[tile])
                    highest = max(highest, rate.max())
        self.tiles_ax.set_title("Games reaching " + (", ".join(f"{tile} ({STYLE_NAMES[TILES[tile]]})" for tile in tiles) or "each tile"))
        self.tiles_ax.set_xlabel("Million games played")
        self.tiles_ax.set_ylabel("% of training games")
        self.tiles_ax.set_xlim(left=0)
        self.tiles_ax.set_ylim(0, 100 if highest > 25 or highest == 0 else 1.25 * highest)
        self.tiles_ax.grid(True, alpha=0.3)

        for _, history, colour, width in lines:
            self.speed_ax.plot(history["games"] / 1e6, smooth(history["avg_moves"] * history["games_per_second"] / 1e6, self.window), color=colour, linewidth=width * 0.8)
            self.length_ax.plot(history["games"] / 1e6, smooth(history["avg_moves"], self.window), color=colour, linewidth=1, linestyle=":", alpha=0.8)
        self.speed_ax.set_title("Training speed (solid) and game length (dotted)")
        self.speed_ax.set_xlabel("Million games played")
        self.speed_ax.set_ylabel("Million moves / second")
        self.speed_ax.set_xlim(left=0)
        self.speed_ax.set_ylim(bottom=0)
        self.speed_ax.grid(True, alpha=0.3)
        self.length_ax.yaxis.set_label_position("right")
        self.length_ax.yaxis.tick_right()
        self.length_ax.set_ylabel("Moves per game")
        self.length_ax.set_ylim(bottom=0)
        self.fig.canvas.draw_idle()


def main() -> None:
    parser = argparse.ArgumentParser(description="Chart training runs against the best system so far")
    parser.add_argument("--run", nargs="+", help="The runs to chart, recorded in experiments/NAME (default: every optimistic-8x6* run)")
    parser.add_argument("--only", nargs="+", metavar="NAME", help="Start with only these runs ticked (default: all of them)")
    parser.add_argument("--reference", choices=("best", "continued", "both", "none"), default="best",
                        help="Which version of the best system starts ticked: with its reset at 1.5M, continued with no reset, both or neither (default: best)")
    parser.add_argument("--tiles", type=int, nargs="*", choices=tuple(TILES), default=[16384, 32768], help="The tiles that start ticked (default: 16384 32768)")
    parser.add_argument("--zoom", action="store_true", help="Start zoomed onto the top of the score curves")
    parser.add_argument("--smooth", type=int, default=20, help="Batches each plotted point averages over (default: 20)")
    parser.add_argument("--interval", type=float, default=10.0, help="Seconds between refreshes (default: 10)")
    parser.add_argument("--once", action="store_true", help="Save the chart once and exit")
    args = parser.parse_args()

    runs = args.run or discover_runs()
    if not runs:
        raise SystemExit("No runs to chart: pass --run NAME")
    labels = list(REFERENCES)
    references = {"best": labels[:1], "continued": labels[1:], "both": labels, "none": []}[args.reference]
    chart = Chart(runs, args.smooth, set(args.only or runs) | set(references), set(args.tiles), args.zoom)
    paths = {name: EXPERIMENTS_DIR / name / HISTORY_NAME for name in runs}
    image = EXPERIMENTS_DIR / runs[0] / "against_best.png"
    last_modified: dict[str, float] = {}
    while True:
        modified = {name: path.stat().st_mtime for name, path in paths.items() if path.exists()}
        if modified != last_modified or not last_modified:
            # A run that hasn't started, or whose trainer is mid-write, keeps what it last showed until the next refresh
            for name in modified:
                if (history := load_history(paths[name])) is not None:
                    chart.histories[name] = columns_of(history)
            chart.draw()
            image.parent.mkdir(parents=True, exist_ok=True)
            chart.fig.savefig(image, dpi=110)
            last_modified = modified
        if args.once:
            print(f"Saved {image}")
            return
        if not plt.fignum_exists(chart.fig.number):
            return
        plt.pause(args.interval)
        # A box ticked since the last save redraws at once on screen; the saved image follows at the next change


if __name__ == "__main__":
    main()
