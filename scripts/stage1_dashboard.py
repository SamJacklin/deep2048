"""A live dashboard for the runs that train stage 1 (scripts/methods.py best-long, joint, stage1-focus), in the browser.

    uv run python scripts/stage1_dashboard.py                    # at http://localhost:8767
    uv run python scripts/stage1_dashboard.py --once output/stage1.html   # one self-contained snapshot, no server

The stage 2 dashboard's page, pointed at the runs that train stage 1: best-long (the best recipe carried on, the
yardstick), joint-* (both stages trained together on whole games), stage1-goal* (tailored to building 16384: their
whole-game scores past 16384 mean nothing, their 16384 rate is the measure) and the finished stage1-focus. Each run's card
gives its games, the share of its training games reaching 16384, its training score and its latest whole-game
evaluation against the best network on the same seeds; the charts put every run on one axis, games added since its
start. The best network alone (decision-8x6-tc-reset at 2M) is drawn in neutral ink.
The page redraws every 15 seconds from the run's own files.
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
sys.path.insert(0, str(Path(__file__).parent))

from methods_dashboard import CHECKPOINTS, POINTS, SMOOTH, TRAINING_SECONDS, at, interval, load_history, smooth  # noqa: E402
from stage2_dashboard import PAGE, STAGE1, evaluation_records  # noqa: E402

from deep_2048.agents.ntuple.train import HISTORY_NAME  # noqa: E402
from deep_2048.experiments import EXPERIMENTS_DIR  # noqa: E402

# The stage 2 page's words, swapped for the focus run's
WORDS = [
    ("<title>Stage 2</title>", "<title>Stage 1</title>"),
    ("Stage 2: the late game from 16384", "Stage 1: getting more games to 16384"),
    ("Converts 16384 → 32768", "Reaches 16384"),
    ("Training score from 16384", "Training score"),
    ("Score gained after the starting board, same smoothing.", "Mean score of the training games (the focus run's from its 4096 board), same smoothing."),
    ("Score from 16384", "Training score"),
    ("Share of training games (each starts from a first-16384 board) that reach 32768", "Share of training games that reach 16384 (whole games; the focus run's start from a first-4096 board)"),
    ("Diamonds mark rate cuts and the TC switch.", "The flat line is the best network's whole-game rate."),
    ("Stage 1 then stage 2 on the usual 1,000 seeds, every 500k games", "Whole games on the usual 1,000 seeds, the same as the best network's evaluation"),
    ("The flat line is stage 1 alone.", "The flat line is the best network."),
    ("first at 500k", "first at 250k–500k"),
    ("· 32768 ${percent(evaluation.r32768)}", "· 16384 ${percent(evaluation.r16384)}"),
    ("Games of stage 2 training", "Games of training added"),
    ("Stage 1 alone (the bar to beat)", "The best network (the bar to beat)"),
    ("Stage 1 alone (seeded games)", "Best network, whole games"),
    ("TC finish since", "TC at 0.02 a weight, from"),
    ("TC finish: no more checks. Ends at", "No rate cuts or checks. Ends after"),
    ("No stage 2 runs yet. ", "No stage 1 runs yet. "),
    ('flat("Stage 1 alone", DATA.stage1.mean', 'flat("Best network", DATA.stage1.mean'),
    ("fromZero: true, marks", "fromZero: false, marks"),
    ("The first evaluation comes at 500k games", "The first evaluation comes 250k games in"),
]


def summary(scores: np.ndarray, tops: np.ndarray) -> dict:
    return {"mean": round(float(scores.mean())), "ci": round(interval(scores)), "r16384": round(100 * float((tops >= 16384).mean()), 1),
            "r32768": round(100 * float((tops >= 32768).mean()), 1), "conversion": round(100 * float((tops >= 16384).mean()), 1)}


def run_names(prefix: str) -> list[tuple[str, str]]:
    """The stage 1 runs to show, with their labels: the best recipe carried on, both stages trained together, the focus run."""
    found = [(f"{prefix}-best-long", "Best network, trained on")]
    for path in sorted(CHECKPOINTS.glob(f"{prefix}-best-long-tc*/methods.json")):
        found.append((path.parent.name, f"Best network, trained on at TC {path.parent.name.rsplit('-tc', 1)[1]} from 8M"))
    for path in sorted(CHECKPOINTS.glob(f"{prefix}-joint-*/methods.json")):
        finish = re.search(r"-tc(\d[\d.e-]*)$", path.parent.name)
        found.append((path.parent.name, "Both stages together" + (f" (stage 2's TC {finish.group(1)} finish)" if finish else "")))
    for path in sorted(CHECKPOINTS.glob(f"{prefix}-stage1-scale*/methods.json")):
        below = path.parent.name.partition("-below")[2]
        found.append((path.parent.name, "Large-tile count term" + (f", none from {below}" if below else "") + ", from tc0.005 at 10M"))
    for path in sorted(CHECKPOINTS.glob(f"{prefix}-stage1-goal*/methods.json")):
        value = json.loads(path.read_text()).get("goal_value")
        found.append((path.parent.name, "Tailored stage 1" + (f" (16384 worth {value / 1000:g}k)" if value is not None else "")))
    found.append((f"{prefix}-stage1-focus", "Stage 1 focus (games from 4096)"))
    return [(name, label) for name, label in found if (CHECKPOINTS / name / "methods.json").exists()]


# The best network's own path from zero, stitched from its runs: (run, the games it covers up to)
BEST_PATH = (("long-8x6", 1_500_000), ("decision-8x6-tc-reset", 2_000_000), ("methods-8x6-best-long", 8_000_000), ("methods-8x6-best-long-tc0.005", 10_000_000))


def best_path() -> tuple[np.ndarray | None, dict]:
    """The best network's training history and evaluations from zero to 10M, stitched from the runs it went through."""
    pieces, records, start = [], {}, 0
    for name, end in BEST_PATH:
        history = load_history(EXPERIMENTS_DIR / name / HISTORY_NAME)
        if history is not None:
            pieces.append(history[(history["games"] > start) & (history["games"] <= end)])
        records |= {games: record for games, record in evaluation_records(name).items() if start < games <= end}
        start = end
    if not pieces:
        return None, records
    columns = [name for name in pieces[0].dtype.names if all(name in piece.dtype.names for piece in pieces)]
    return np.concatenate([piece[columns] for piece in pieces]), records


def collect(prefix: str, scratch: bool = False) -> dict:
    """Every run on one axis: games added since its start (the best network's 2M, or the stage 2 run's end). With
    `scratch`, the runs from zero (scratch-scale, scratch-stall) and the best network's own path, on absolute games."""
    now = time.time()
    stage1_records = evaluation_records(STAGE1[0]).get(STAGE1[1])
    stage1 = summary(*stage1_records) if stage1_records else None
    if scratch:
        names = [("@best-path", "The best network's own path")] + [
            (path.parent.name, "From scratch with the count term" + (", read by state" if "-by-state" in path.parent.name else "")
             + (f" (rate {path.parent.name.rsplit('-r', 1)[1]})" if "-r0" in path.parent.name or "-r1" in path.parent.name else ""))
            for path in sorted(CHECKPOINTS.glob(f"{prefix}-scratch-scale*/methods.json"))] + [
            (path.parent.name, "From scratch, moving on at stalls" + (", with the count term" if "-scale" in path.parent.name else ""))
            for path in sorted(CHECKPOINTS.glob(f"{prefix}-scratch-stall*/methods.json"))]
    else:
        names = run_names(prefix)
    runs = []
    for slot, (name, label) in enumerate(names):
        if name == "@best-path":
            state = {"start": 0, "total": 10_000_000, "rate": 0.02}
            history, records = best_path()
            history_path = None
        else:
            state = json.loads((CHECKPOINTS / name / "methods.json").read_text())
            history_path = EXPERIMENTS_DIR / name / HISTORY_NAME
            history, records = load_history(history_path), evaluation_records(name)
        origin = int(state.get("start", 0))
        run = {"id": name, "name": label, "slot": slot, "optimism": None, "status": "waiting", "goal": (state.get("total") or origin) - origin, "hold": 0,
               "cuts": [], "cut_why": [], "tc_from": 0, "rate": (state.get("rate") or 0.02) * 64, "checks": {"phase": "tc", "total": (state.get("total") or origin) - origin}}
        if history is not None and history.size:
            columns = {column: history[column] for column in history.dtype.names}
            games = columns["games"] - origin
            score, reach = smooth(columns["avg_score"]), smooth(columns["rate_16384"])
            keep = np.unique(np.concatenate([np.linspace(0, games.size - 1, min(POINTS, games.size)).round().astype(int), [games.size - 1]]))
            run["lines"] = {"games": [int(g) for g in games[keep]], "score": [round(float(v)) for v in score[keep]], "conversion": [round(float(v), 2) for v in reach[keep]]}
            before_score, before_reach = at(games, score, games[-1] - 500_000), at(games, reach, games[-1] - 500_000)
            speed = float(columns["games_per_second"][-20:].mean())
            run["latest"] = {"games": int(games[-1]), "score": round(float(score[-1])), "conversion": round(float(reach[-1]), 2),
                             "score_change": None if before_score is None else round(float(score[-1] - before_score)),
                             "conversion_change": None if before_reach is None else round(float(reach[-1] - before_reach), 2), "speed": round(speed)}
            training = history_path is not None and now - history_path.stat().st_mtime < TRAINING_SECONDS
            run["status"] = "training" if training else ("finished" if run["goal"] and games[-1] >= run["goal"] else "stopped")
            if training and run["goal"] > games[-1] and speed > 0:
                run["eta_hours"] = round((run["goal"] - games[-1]) / speed / 3600, 1)
        run["evaluations"] = []
        for at_games, (scores, tops) in sorted(records.items()):
            entry = {"games": at_games - origin} | summary(scores, tops)
            if stage1_records is not None and stage1_records[0].size == scores.size:
                difference = scores - stage1_records[0]
                entry |= {"vs_stage1": round(float(difference.mean())), "vs_stage1_ci": round(interval(difference))}
            run["evaluations"].append(entry)
        runs.append(run)
    return {"updated": time.strftime("%H:%M:%S"), "runs": runs, "stage1": stage1, "smooth": SMOOTH, "pool": None}


def page(data: dict | None) -> bytes:
    html = PAGE.read_text()
    for old, new in WORDS:
        html = html.replace(old, new)
    return html.replace("/*__DATA__*/null", json.dumps(data) if data is not None else "null").encode()


def serve(prefix: str, port: int, browser: bool, scratch: bool = False) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path.startswith("/api/data"):
                body, kind = json.dumps(collect(prefix, scratch)).encode(), "application/json"
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

        def log_message(self, *_: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    address = f"http://localhost:{port}"
    print(f"Stage 1 dashboard at {address} (Ctrl-C to stop; the run carries on)")
    if browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="A live dashboard for the runs that train stage 1")
    parser.add_argument("--prefix", default="methods-8x6", help="The runs' prefix (default: methods-8x6; smoke-methods for the smoke runs)")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--scratch", action="store_true", help="The runs from zero (scratch-scale) against the best network's own path, on absolute games")
    parser.add_argument("--once", type=Path, metavar="FILE", help="Write one self-contained snapshot to FILE and exit")
    args = parser.parse_args()
    if args.once:
        args.once.parent.mkdir(parents=True, exist_ok=True)
        args.once.write_bytes(page(collect(args.prefix, args.scratch)))
        print(f"Wrote {args.once}")
    else:
        serve(args.prefix, args.port, not args.no_browser, args.scratch)


if __name__ == "__main__":
    main()
