"""A focused live dashboard for the stage 2 runs (scripts/methods.py stage2), in the browser.

    uv run python scripts/stage2_dashboard.py                    # at http://localhost:8766
    uv run python scripts/stage2_dashboard.py --once output/stage2.html   # one self-contained snapshot, no server

One card a run says where it is in its schedule (the hold, the cuts, the TC finish), what it will do next and why,
its conversion from 16384 to 32768 now and how that has moved, and its latest whole-game evaluation against stage 1
alone on the same seeds. Three charts follow: conversion to 32768 in the training games (each starts at 16384),
the training score from 16384, and the whole-game evaluations with their 95% intervals. Stage 1 alone is drawn in
neutral ink where it has a like-for-like number. The page redraws every 15 seconds from the runs' own files.
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

from methods_dashboard import CHECKPOINTS, POINTS, SMOOTH, TRAINING_SECONDS, at, interval, load_history, long_checks, smooth  # noqa: E402

from deep_2048.agents.ntuple.train import HISTORY_NAME  # noqa: E402
from deep_2048.experiments import EXPERIMENTS_DIR  # noqa: E402

STAGE1 = ("decision-8x6-tc-reset", 2_000_000)  # stage 1: its evaluation at 2M on the usual seeds
SLOTS = {625: 0, 1250: 1, 2500: 2}  # a run keeps its colour by its optimism value; others take the next free slots
PAGE = Path(__file__).with_name("stage2_dashboard.html")


def evaluation_records(name: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """A run's greedy evaluations on the usual seeds: games → (scores, max tiles). Search evaluations are left out."""
    path = EXPERIMENTS_DIR / name / "evaluations.json"
    found: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if not path.exists():
        return found
    try:
        records = json.loads(path.read_text())
    except ValueError:
        return found
    for record in records:
        label = record["result"]["label"]
        games = re.search(r"\(([\d,]+) games", label)
        if games and "depth" not in label.lower() and "search" not in label.lower():
            found[int(games.group(1).replace(",", ""))] = (np.array(record["result"]["scores"], dtype=np.float64), np.array(record["result"]["max_tiles"]))
    return found


def summary(scores: np.ndarray, tops: np.ndarray) -> dict:
    reached = (tops >= 16384).mean()
    return {"mean": round(float(scores.mean())), "ci": round(interval(scores)), "r16384": round(100 * float(reached), 1),
            "r32768": round(100 * float((tops >= 32768).mean()), 1), "conversion": round(100 * float((tops >= 32768).mean() / reached), 1) if reached else None}


def collect(prefix: str) -> dict:
    now = time.time()
    stage1_records = evaluation_records(STAGE1[0]).get(STAGE1[1])
    stage1 = summary(*stage1_records) if stage1_records else None
    names = sorted({path.parent.name for path in EXPERIMENTS_DIR.glob(f"{prefix}-stage2-opt*/{HISTORY_NAME}")}
                   | {path.parent.name for path in CHECKPOINTS.glob(f"{prefix}-stage2-opt*/methods.json")})
    taken, runs = set(), []
    for name in names:
        state_path = CHECKPOINTS / name / "methods.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        optimism = state.get("optimism")
        slot = SLOTS.get(int(optimism)) if optimism is not None and float(optimism).is_integer() else None
        if slot is None or slot in taken:
            slot = next((index for index in (3, 4, 5, 6, 7, 0, 2) if index not in taken), len(taken) % 8)
        taken.add(slot)
        label = f"Optimism {optimism:,.0f}" if optimism is not None else name
        tag = name.split(f"opt{optimism:g}", 1)[-1].strip("-") if optimism is not None else ""
        on = re.match(r"on-(.*?-\d+[mk])(?:-|$)", tag)  # stage2 --base: built on another stage 1
        if on:
            tag = tag[on.end():]
        finish = re.search(r"-?tc(\d[\d.e-]*)$", tag)  # a stage2-finish branch: <source arm>-tc<rate>
        if finish:
            tag = tag[: finish.start()]
        sums = tag.endswith("-sums")
        tag = tag.removesuffix("-sums")
        tc_only = tag.endswith("tc")
        tag = tag.removesuffix("tc").rstrip("-")
        if tag.startswith("dg"):
            tag = f"warm start, stage 2 downgrading from {tag[2:]}"
        elif state.get("stage_downgrade"):
            tag = ", ".join(filter(None, [tag, f"stage 2 downgrading from {state['stage_downgrade']}"]))
        if tc_only:
            tag = ", ".join(filter(None, [tag, "TC from the start" + (" with stage 1's sums" if sums else "")]))
        if finish:
            tag = ", ".join(filter(None, [tag, f"TC finish at {finish.group(1)}"]))
        if on:
            tag = ", ".join(filter(None, [f"on {on.group(1)}", tag]))
        run = {"id": name, "name": label + (f" ({tag})" if tag else ""), "slot": slot, "optimism": optimism, "status": "waiting",
               "goal": state.get("total"), "hold": state.get("hold_until", 0), "cuts": state.get("cut_at", []), "cut_why": state.get("cut_why", []),
               "tc_from": state.get("tc_from"), "rate": (state.get("rate") or 0) * state.get("reads", 64), "optimistic_weights": state.get("optimistic_weights")}
        history_path = EXPERIMENTS_DIR / name / HISTORY_NAME
        history = load_history(history_path)
        if history is not None:
            columns = {column: history[column] for column in history.dtype.names}
            games = columns["games"]
            score = smooth(columns["avg_score"])
            conversion = smooth(columns["rate_32768"]) if "rate_32768" in columns else np.full(games.size, np.nan)
            keep = np.unique(np.concatenate([np.linspace(0, games.size - 1, min(POINTS, games.size)).round().astype(int), [games.size - 1]]))
            run["lines"] = {"games": [int(g) for g in games[keep]], "score": [round(float(v)) for v in score[keep]],
                            "conversion": [None if np.isnan(v) else round(float(v), 2) for v in conversion[keep]]}
            before_score, before_conversion = at(games, score, games[-1] - 500_000), at(games, conversion, games[-1] - 500_000)
            speed = float(columns["games_per_second"][-20:].mean())
            run["latest"] = {"games": int(games[-1]), "score": round(float(score[-1])), "conversion": round(float(conversion[-1]), 2),
                             "score_change": None if before_score is None else round(float(score[-1] - before_score)),
                             "conversion_change": None if before_conversion is None else round(float(conversion[-1] - before_conversion), 2), "speed": round(speed)}
            if "cut_at" in state:
                run["checks"] = long_checks(columns, state)
            training = now - history_path.stat().st_mtime < TRAINING_SECONDS
            goal = state.get("total") or 0
            run["status"] = "training" if training else ("finished" if goal and games[-1] >= goal else "stopped")
            if training and goal > games[-1] and speed > 0:
                run["eta_hours"] = round((goal - games[-1]) / speed / 3600, 1)
        # Compared with its own stage 1 alone: the best network's evaluation, or its --base network's own run's at its games
        reference = stage1_records
        base_file = Path(state.get("base") or "")
        if base_file.name.startswith("games_") and base_file.stem.removeprefix("games_").isdigit() and base_file.parent.name != STAGE1[0]:
            reference = evaluation_records(base_file.parent.name).get(int(base_file.stem.removeprefix("games_")))
        if state.get("from") and not state.get("base"):  # a stage2-finish branch: its source's base
            source_state = CHECKPOINTS / f"{prefix}-{state['from']}" / "methods.json"
            if source_state.exists():
                base_file = Path(json.loads(source_state.read_text()).get("base") or "")
                if base_file.name.startswith("games_") and base_file.stem.removeprefix("games_").isdigit() and base_file.parent.name != STAGE1[0]:
                    reference = evaluation_records(base_file.parent.name).get(int(base_file.stem.removeprefix("games_")))
        run["evaluations"] = []
        for at_games, (scores, tops) in sorted(evaluation_records(name).items()):
            entry = {"games": at_games} | summary(scores, tops)
            if reference is not None and reference[0].size == scores.size:
                difference = scores - reference[0]
                entry |= {"vs_stage1": round(float(difference.mean())), "vs_stage1_ci": round(interval(difference))}
            run["evaluations"].append(entry)
        runs.append(run)
    runs.sort(key=lambda run: (run["slot"], run["id"]))
    pool = next(iter(CHECKPOINTS.glob(f"{prefix}-stage2-pool/source.txt")), None)
    return {"updated": time.strftime("%H:%M:%S"), "runs": runs, "stage1": stage1, "smooth": SMOOTH,
            "pool": pool.read_text().strip().splitlines()[-1] if pool else None}


def page(data: dict | None) -> bytes:
    return PAGE.read_text().replace("/*__DATA__*/null", json.dumps(data) if data is not None else "null").encode()


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

        def log_message(self, *_: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    address = f"http://localhost:{port}"
    print(f"Stage 2 dashboard at {address} (Ctrl-C to stop; the runs carry on)")
    if browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="A focused live dashboard for the stage 2 runs")
    parser.add_argument("--prefix", default="methods-8x6", help="The runs' prefix (default: methods-8x6; smoke-methods for the smoke runs)")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true")
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
