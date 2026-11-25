"""A live dashboard for the methods batch (or any group of runs), in the browser.

    uv run python scripts/methods_dashboard.py                    # the methods batch, at http://localhost:8765
    uv run python scripts/methods_dashboard.py --prefix optimistic-8x6
    uv run python scripts/methods_dashboard.py --once output/methods.html   # one self-contained snapshot, no server

The page redraws every 15 seconds from the runs' own files and keeps whatever is ticked. It shows

- a scoreboard: each run's state, games, rate, training score, how that compares with the best system at the same
  number of games, what it has gained over its last 100k and 250k games, its tile rates, its latest seeded
  evaluation (against the best system's on the same seeds where there is one), speed and time left;
- the training score over the whole run, and the same zoomed onto the top where the runs separate;
- how fast each run is still gaining, against the stall rule's threshold;
- each run's learning rate over time, as the step a board's value takes;
- the share of training games reaching 16384 and 32768;
- the seeded greedy evaluations with their 95% intervals, the firm numbers the training curves stand in for.

The best system is drawn in neutral ink in two versions: the long run continued with no reset (the like-for-like
reference for a run that isn't reset, and what the scoreboard compares against) and with round 1's reset at 1.5M.
Training scores are the trainer's batch averages, smoothed over SMOOTH batches. Every chart has one axis; pointing
at one lists every run's value there. The page needs no internet connection.
"""

import argparse
import json
import re
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_2048.agents.ntuple.train import HISTORY_NAME  # noqa: E402
from deep_2048.experiments import EXPERIMENTS_DIR  # noqa: E402

CHECKPOINTS = Path(__file__).parent.parent / "checkpoints" / "ntuple"
SMOOTH = 50  # batches a plotted score averages over
POINTS = 500  # points a line is thinned to
STALL_WINDOW, STALL_GAIN = 250_000, 3_000.0  # scripts/methods.py's stall rule
# scripts/methods.py's long stall rule for the td-opt runs: checked every LONG_STEP games, it needs LONG_WINDOW games at
# the current rate and both gains over that window under their thresholds; the TC finish starts by total - LONG_TC
LONG_STEP, LONG_WINDOW, LONG_GAIN, LONG_GAIN_32768, LONG_CUTS, LONG_TC = 100_000, 500_000, 3_000.0, 0.5, 2, 2_000_000
TRAINING_SECONDS = 240  # a history written to this recently belongs to a run that is training

# The methods batch's arms in their fixed order: a run keeps its colour whichever others are shown
# Slots of runs that finished and dropped off the page are reused (their results stay in experiments/): td-opt and
# td-opt-1250 took TC at 0.01's and 0.04's (never run); the 15M runs take the 2,500 run's and Guei's TC finish's
METHOD_ARMS = {"tc-0.005": "TC at 0.005", "td-opt": "TD 0.1 + optimism 100, long run", "td-opt-1250": "TD 0.1 + optimism 1,250, long run", "td-0.1": "TD from 0.1, Guei's schedule",
               "td-opt-1250-hold-nodg": "Optimism 1,250, 0.1 held to 8M, no downgrading", "td-yeh": "TD, Yeh's rule",
               "td-opt-1250-hold": "Optimism 1,250, 0.1 held to 8M, downgrading", "td-yeh-tc": "Yeh's rule, TC finish"}
REFERENCES = {"best": ("Best so far, no reset (TC at 0.02)", ("long-8x6", "decision-8x6-control")), "best_reset": ("Best so far, reset at 1.5M", ("long-8x6", "decision-8x6-tc-reset"))}
BEST_RATE = (0.02, 64)  # the long run's base rate a weight and the weights an update touches
METHODS_TOTAL = 3_000_000
# What each from-scratch arm opens with, as (step a board, rule), for a run whose own record doesn't say yet
OPENING_RATES = {"best-long": (0.02 * 64, "tc"), "tc-0.005": (0.005 * 64, "tc"), "tc-0.01": (0.01 * 64, "tc"), "tc-0.04": (0.04 * 64, "tc"), "td-0.1": (0.1, "td"), "td-0.5": (0.5, "td")}


def load_history(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        history = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    except ValueError:  # the trainer is mid-write
        return None
    return history if history.size else None


def smooth(values: np.ndarray, window: int = SMOOTH) -> np.ndarray:
    """A trailing mean over up to `window` batches."""
    sums = np.cumsum(np.insert(values.astype(np.float64), 0, 0.0))
    index = np.arange(1, values.size + 1)
    start = np.maximum(index - window, 0)
    return (sums[index] - sums[start]) / (index - start)


def at(x: np.ndarray, y: np.ndarray, where: float) -> float | None:
    """y where x first reaches `where`, or None outside x's range."""
    if where < x[0]:
        return None
    reached = int(np.searchsorted(x, where))
    return float(y[reached]) if reached < x.size else None


def long_checks(columns: dict[str, np.ndarray], state: dict) -> dict:
    """The td-opt runs' rate-cut rule as methods.py applies it: each check so far, and the rule applied to now.

    A check's means are of the last SMOOTH batches up to its game count, from the run's own history, as methods.py
    takes them, so the numbers match its stall checks.
    """
    games = columns["games"]

    def mean(column: str, upto: float) -> float | None:
        if column not in columns:
            return None
        values = columns[column][games <= upto]
        return float(values[-SMOOTH:].mean()) if values.size >= SMOOTH else None

    cuts = list(state.get("cut_at", []))
    tc_from = state.get("tc_from")
    total = int(state.get("total", 8_000_000))

    hold = int(state.get("hold_until", 0))

    def check(at_games: int) -> dict:
        since = max([cut for cut in cuts if cut < at_games] or [0])
        if tc_from is not None and at_games > tc_from:
            since = tc_from
        played = at_games - since
        entry = {"games": at_games, "played": played, "enough": played >= LONG_WINDOW, "score": None, "r32768": None}
        now_score, before_score = mean("avg_score", at_games), mean("avg_score", at_games - LONG_WINDOW)
        now_goal, before_goal = mean("rate_32768", at_games), mean("rate_32768", at_games - LONG_WINDOW)
        if None not in (now_score, before_score):
            entry["score"] = round(now_score - before_score)
        if None not in (now_goal, before_goal):
            entry["r32768"] = round(now_goal - before_goal, 2)
        entry["score_met"] = entry["score"] is not None and entry["score"] < LONG_GAIN
        entry["r32768_met"] = entry["r32768"] is not None and entry["r32768"] < LONG_GAIN_32768
        entry["held"] = at_games <= hold
        entry["stall"] = not entry["held"] and entry["enough"] and entry["score_met"] and entry["r32768_met"]
        return entry

    latest = int(games[-1])
    end = min(latest, tc_from) if tc_from is not None else latest
    tc_games = int(state.get("tc_games", LONG_TC))
    td_games = total - tc_games
    deadline = None
    if tc_from is None and len(cuts) < LONG_CUTS:  # scripts/methods.py's cut_deadline
        deadline = max((td_games // 2, td_games * 3 // 4)[len(cuts)], hold, (cuts[-1] if cuts else 0) + LONG_WINDOW)
    history = [check(at_games) for at_games in range(LONG_STEP, end + 1, LONG_STEP)]
    phase = "tc" if tc_from is not None else "td"
    return {"phase": phase, "cuts": cuts, "max_cuts": LONG_CUTS, "tc_from": tc_from, "total": total, "tc_latest": td_games, "deadline": deadline, "cut_why": state.get("cut_why", []),
            "rate": state.get("rate"), "reads": state.get("reads", 64), "optimism": state.get("optimism"), "hold_until": hold, "downgrade": state.get("downgrade", True),
            "window": LONG_WINDOW, "gain": LONG_GAIN, "gain_32768": LONG_GAIN_32768,
            "now": check(latest), "next": (latest // LONG_STEP + 1) * LONG_STEP, "history": history[-8:]}


def evaluations(names: tuple[str, ...]) -> dict[int, np.ndarray]:
    """Seeded greedy evaluations recorded for these runs: games → each game's score."""
    found: dict[int, np.ndarray] = {}
    for name in names:
        path = EXPERIMENTS_DIR / name / "evaluations.json"
        if not path.exists():
            continue
        try:
            records = json.loads(path.read_text())
        except ValueError:
            continue
        for record in records:
            label = record["result"]["label"]
            games = re.search(r"\(([\d,]+) games", label)
            if games and "depth" not in label.lower():
                found[int(games.group(1).replace(",", ""))] = np.array(record["result"]["scores"], dtype=np.float64)
    return found


def interval(values: np.ndarray) -> float:
    return float(1.96 * values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0


def series(columns: dict[str, np.ndarray]) -> dict:
    """A run's lines, thinned for the page."""
    games, moves = columns["games"], columns["moves"]
    score = smooth(columns["avg_score"])
    # Gain a 100k games over the trailing window, where the run's own history reaches back that far
    gain = np.full(games.size, np.nan)
    back = np.searchsorted(games, games - STALL_WINDOW)
    reaches = games - STALL_WINDOW >= games[0] - 1_000
    gain[reaches] = (score[reaches] - score[back[reaches]]) / (STALL_WINDOW / 100_000)
    keep = np.unique(np.concatenate([np.linspace(0, games.size - 1, min(POINTS, games.size)).round().astype(int), [games.size - 1]]))
    lines = {"games": games[keep], "moves": moves[keep], "score": score[keep], "gain": gain[keep]}
    for tile in (8192, 16384, 32768):
        if f"rate_{tile}" in columns:
            lines[f"r{tile}"] = smooth(columns[f"rate_{tile}"])[keep]
    return {key: [None if np.isnan(v) else round(float(v), 2) for v in values] for key, values in lines.items()}


def join(names: tuple[str, ...]) -> dict[str, np.ndarray] | None:
    parts = [load_history(EXPERIMENTS_DIR / name / HISTORY_NAME) for name in names]
    if any(part is None for part in parts):
        return None
    shared = set.intersection(*(set(part.dtype.names) for part in parts))
    return {column: np.concatenate([part[column] for part in parts]) for column in shared}


def discover(prefix: str) -> dict[str, str]:
    """The runs to show, in a fixed order: run name → display name."""
    if prefix.endswith(("methods-8x6", "smoke-methods")):  # every arm has a row, started or not
        return {f"{prefix}-{arm}": label for arm, label in METHOD_ARMS.items()}
    names = sorted(path.parent.name for path in EXPERIMENTS_DIR.glob(f"{prefix}*/{HISTORY_NAME}"))

    def order(name: str) -> tuple:
        rest = name[len(prefix):].strip("-")
        number = re.match(r"\d+", rest)
        return (rest[number.end():] != "" if number else rest != "", -(int(number.group()) if number else 10**9), rest)

    return {name: (name[len(prefix):].strip("-") or prefix) for name in sorted(names, key=order)[:8]}


def collect(prefix: str) -> dict:
    """Everything the page draws, read fresh from the runs' files."""
    now = time.time()
    references, reference_evaluations = {}, {}
    for key, (label, names) in REFERENCES.items():
        columns = join(names)
        if columns is None:
            continue
        reference_evaluations[key] = evaluations(names)
        references[key] = {"name": label, "lines": series(columns), "columns": columns,
                           "evaluations": [{"games": games, "mean": round(float(scores.mean())), "ci": round(interval(scores))} for games, scores in sorted(reference_evaluations[key].items())]}
    best = references.get("best")
    best_score = smooth(best["columns"]["avg_score"]) if best else None

    runs, total = [], 0
    for slot, (name, label) in enumerate(discover(prefix).items()):
        path = EXPERIMENTS_DIR / name / HISTORY_NAME
        state_path = CHECKPOINTS / name / "methods.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        history = load_history(path)
        run = {"id": name, "name": label, "slot": slot, "status": "not started", "from": state.get("from"), "goal": state.get("total") or (METHODS_TOTAL if "methods" in prefix else None)}
        reads = state.get("reads", 64)
        # The rate over time: each finished session, then the one in progress
        segments, previous = [], state.get("start", 0)
        for target, rate, kind in state.get("rates", []):
            segments.append([previous, target, rate * reads, kind])
            previous = target
        if "current" in state:
            segments.append([state["current"][0], state["current"][1], state["current"][2] * reads, state["current"][3]])
        arm = name[len(prefix):].strip("-")
        if history is not None:
            played = int(history["games"][-1])
            if not segments and arm in OPENING_RATES:
                segments.append([0, played, *OPENING_RATES[arm]])
            elif segments and segments[-1][1] < played:  # a session in progress that the record doesn't describe: assume the last rate
                segments.append([segments[-1][1], played, segments[-1][2], segments[-1][3]])
        if "methods" in prefix:
            total = max(total, METHODS_TOTAL)
        run["rates"] = segments
        run["cuts"] = [a[1] for a, b in zip(segments, segments[1:]) if b[2] < a[2] * 0.999 and b[3] == a[3] == "td"]
        run["first_stall"] = state.get("first_stall")
        if history is None:
            if state.get("from") and "start" not in state:
                run["status"] = "waiting"
            runs.append(run)
            continue
        columns = {column: history[column] for column in history.dtype.names}
        games, score = columns["games"], smooth(columns["avg_score"])
        run["lines"] = series(columns)
        if arm.startswith("td-opt") and "cut_at" in state:
            run["checks"] = long_checks(columns, state)
        latest = {"games": int(games[-1]), "score": round(float(score[-1])), "speed": round(float(columns["games_per_second"][-20:].mean()), 1)}
        total = max(total, latest["games"])
        for span, key in ((100_000, "gain100"), (250_000, "gain250")):
            before = at(games, score, games[-1] - span)
            latest[key] = None if before is None else round(score[-1] - before)
        for tile in (8192, 16384, 32768):
            if f"rate_{tile}" in columns:
                latest[f"r{tile}"] = round(float(smooth(columns[f"rate_{tile}"])[-1]), 1)
        if best:
            same = at(best["columns"]["games"], best_score, games[-1])
            latest["vs_best"] = None if same is None else round(score[-1] - same)
        latest["rate"] = segments[-1][2] if segments else None
        latest["rate_kind"] = segments[-1][3] if segments else None
        run["latest"] = latest
        own = evaluations((name,))
        run["evaluations"] = []
        for at_games, scores in sorted(own.items()):
            entry = {"games": at_games, "mean": round(float(scores.mean())), "ci": round(interval(scores))}
            theirs = reference_evaluations.get("best", {}).get(at_games)
            if theirs is not None and theirs.size == scores.size:
                entry |= {"vs_best": round(float((scores - theirs).mean())), "vs_best_ci": round(interval(scores - theirs))}
            run["evaluations"].append(entry)
        goal = state.get("total") or (METHODS_TOTAL if "methods" in prefix else 0)
        if now - path.stat().st_mtime < TRAINING_SECONDS:
            run["status"] = "training"
            if goal > games[-1] and latest["speed"] > 0:
                run["eta_hours"] = round((goal - games[-1]) / latest["speed"] / 3600, 1)
        else:
            run["status"] = "finished" if goal and games[-1] >= goal else "stopped"
        runs.append(run)

    for reference in references.values():
        reference.pop("columns")
        total = max(total, reference["lines"]["games"][-1])
    for reference in references.values():
        reference["rate"] = BEST_RATE[0] * BEST_RATE[1]
    return {"prefix": prefix, "updated": time.strftime("%H:%M:%S"), "total": total, "runs": runs, "references": references,
            "stall_gain_per_100k": STALL_GAIN / (STALL_WINDOW / 100_000), "smooth": SMOOTH}


PAGE = Path(__file__).with_name("methods_dashboard.html")


def page(data: dict | None) -> bytes:
    """The page, with the data built in for a snapshot or fetched live otherwise."""
    html = PAGE.read_text()
    return html.replace("/*__DATA__*/null", json.dumps(data) if data is not None else "null").encode()


def serve(prefix: str, port: int, browser: bool) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path.startswith("/api/data"):
                body, kind = json.dumps(collect(prefix)).encode(), "application/json"
            elif self.path in ("/", "/index.html"):
                body, kind = page(None), "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_: object) -> None:  # keep the terminal quiet
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    address = f"http://localhost:{port}"
    print(f"Dashboard for {prefix}* at {address} (Ctrl-C to stop; the runs carry on)")
    if browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="A live dashboard for a group of training runs")
    parser.add_argument("--prefix", default="methods-8x6", help="Show the runs whose names start with this (default: methods-8x6)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="Don't open the page in a browser")
    parser.add_argument("--once", type=Path, metavar="FILE", help="Write one self-contained snapshot to FILE and exit")
    args = parser.parse_args()
    if args.once:
        args.once.parent.mkdir(parents=True, exist_ok=True)
        args.once.write_bytes(page(collect(args.prefix)))
        print(f"Wrote {args.once}")
    else:
        serve(args.prefix, args.port, not args.no_browser)


if __name__ == "__main__":
    main()
