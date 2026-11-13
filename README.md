# Ryzen long runs: instructions

These are instructions for a Claude agent on the Ryzen machine (AMD Ryzen 7 9850X3D, 32 GB RAM, Windows). Follow them exactly and in order.
- **Don't** change code, open pull requests, or merge branches.
- **If a step fails,** record what happened in the results file (step 8), carry on where it's safe, and push what you have at the end.

## What this is

Two long training runs of the 2048 n-tuple agent, from scratch to 10,000,000 games each:

| Arm | Run name | What it does |
|---|---|---|
| `recipe` | `ryzen-recipe` | This project's current best method: temporal coherence (TC) at 0.02, downgrading from 16384, late-game starts. Its TC sums are reset once, at 1.5M games. |
| `optimism` | `ryzen-optimism` | Guei's recipe on top of the same setup. Every weight starts at 5,000 (320,000 per board), and training runs at a fixed rate of 0.0016 with TC off until 9M games. Then TC at 0.02 for the last 1M. |

- **While training:** each run evaluates itself greedily every 250k games, on 1,000 seeded games.
- **At the end:** exact depth-3 evaluations at the kept points: 1.5M, 5M and 10M for `recipe`, and 5M, 9M and 10M for `optimism`.
- **The scripts are on branch `feat/fast-tc`:** `scripts/run_ryzen_arms.sh` and `scripts/summarise_runs.py`.

The runs take a long time, probably 1–3 days with both running together. The scripts resume where they stopped if interrupted.

## 0. Before starting (the human operator does these)

- WSL2 with Ubuntu is installed on Windows (`wsl --install -d Ubuntu` in an administrator PowerShell, then reboot).
- The machine is plugged in, and Windows Update restarts are paused for a week (Settings → Windows Update → Pause updates).
- GitHub authentication works inside WSL for pushing to `SamJacklin/deep2048`. Either `gh auth login` with the GitHub CLI, or an SSH key added to GitHub.

**Everything below runs inside the WSL Ubuntu shell, not PowerShell.**

## 1. Stop Windows from sleeping

Run from WSL (these call Windows' `powercfg`):

```bash
powercfg.exe /change standby-timeout-ac 0
powercfg.exe /change hibernate-timeout-ac 0
powercfg.exe /change monitor-timeout-ac 30
```

## 2. Install tools

```bash
sudo apt-get update && sudo apt-get install -y git build-essential procps
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

## 3. Get the code

Clone into the Linux filesystem (your home directory), **not** under `/mnt/c`, which is much slower:

```bash
cd ~
git clone https://github.com/SamJacklin/deep2048.git
cd ~/deep2048
git checkout feat/fast-tc
git log --oneline -1    # expect: 2b03241 feat: add the Ryzen's long from-scratch runs and a results summary
uv sync
uv run pytest -q        # expect: 85 passed
```

If the commit at the top of `feat/fast-tc` is newer than `2b03241`, use it. Record the hash you used.

## 4. Record the machine

Save this output for the results file:

```bash
lscpu | grep -E "Model name|^CPU\(s\)|Thread"
free -g | head -2
nproc
```

## 5. Thread check (about 10–15 minutes)

Measure training speed on 8 against 16 threads. Each command trains a fresh network for 30,000 games:

```bash
cd ~/deep2048
for threads in 8 16; do
  NTUPLE_NETWORK=8x6 NTUPLE_LEARNING_RATE=0.02 NTUPLE_TEMPORAL_COHERENCE=true NTUPLE_DOWNGRADE_TILE=16384 \
  NTUPLE_THREADS=$threads NTUPLE_THREADS_AFTER=0 \
  uv run python scripts/train_ntuple.py --run smoke-threads-$threads --games 30000 > /tmp/threads-$threads.log 2>&1
  echo "== $threads threads"; grep -E "^\s+games\s+30,000" /tmp/threads-$threads.log
done
```

- **Record** each "games/s" figure at 30,000 games, and the 20,000 row too (`grep "games    20,000"`).
- **Then delete the test runs:** `rm -rf checkpoints/ntuple/smoke-threads-* experiments/smoke-threads-*`
- **The thread count for the runs:** both arms run together with **8 threads each**, using all 16 hardware threads, whatever the check shows. The check is only for the record.

## 6. Start both runs

Start them detached, so they keep running if this session ends:

```bash
cd ~/deep2048
THREADS=8 setsid nohup scripts/run_ryzen_arms.sh recipe   > ryzen-recipe.log 2>&1 < /dev/null &
THREADS=8 setsid nohup scripts/run_ryzen_arms.sh optimism > ryzen-optimism.log 2>&1 < /dev/null &
sleep 60; tail -3 experiments/ryzen-recipe/train.log experiments/ryzen-optimism/train.log
```

Both train logs should show progress lines (`games … avg score …`).

**What to expect along the way:**
- **`optimism` scores much lower than `recipe` for millions of games.** That's normal: optimistic weights take a long time to settle. Don't stop it.
- **`recipe` drops briefly just after 1.5M games,** when its TC sums are reset.
- **Disk:** each run keeps three 1.5 GB snapshots, plus one checkpoint while training.

## 7. Monitor until both finish

Check every hour or so:

```bash
cd ~/deep2048
tail -2 experiments/ryzen-recipe/train.log experiments/ryzen-optimism/train.log
grep -h "^N-tuple" ryzen-recipe.log ryzen-optimism.log | tail -6
pgrep -fl run_ryzen_arms || echo "no run is active"
```

- **Done:** a run is finished when its log (`ryzen-recipe.log` or `ryzen-optimism.log`) ends with `ryzen-recipe finished` or `ryzen-optimism finished`.
- **If the machine restarted, or a run died without that line:** rerun exactly the same command from step 6 for that arm. It resumes from its last checkpoint and repeats any missing evaluations. A run that already finished just reports so.
- **Record** any interruptions, with times, for the results file.

## 8. Write the results

When both runs have finished:

```bash
cd ~/deep2048
uv run python scripts/summarise_runs.py ryzen-recipe ryzen-optimism > /tmp/ryzen-tables.md
uv run python scripts/watch_ntuple.py --run ryzen-recipe ryzen-optimism --once --image experiments/figures/ryzen-arms.png
uv run python scripts/watch_ntuple.py --run ryzen-recipe --once
uv run python scripts/watch_ntuple.py --run ryzen-optimism --once
cp ryzen-recipe.log experiments/ryzen-recipe/run.log
cp ryzen-optimism.log experiments/ryzen-optimism/run.log
```

Create `experiments/ryzen-RESULTS.md` containing:
1. **The machine:** CPU, RAM, cores (step 4).
2. **The commit:** the hash used (step 3).
3. **The thread check:** games/s for 8 and 16 threads (step 5).
4. **Timings:** start and end times of each run, and any interruptions or problems.
5. **Results:** the tables from `/tmp/ryzen-tables.md`, pasted unchanged.
6. **Charts:** links to `figures/ryzen-arms.png`, `ryzen-recipe/training_progress.png` and `ryzen-optimism/training_progress.png`.

Don't interpret the results; the tables are what's needed.

## 9. Push the results

```bash
cd ~/deep2048
git checkout -b results/ryzen-long-runs
git add experiments/ryzen-recipe experiments/ryzen-optimism experiments/figures/ryzen-arms.png experiments/ryzen-RESULTS.md
git status --short     # check: only those paths, no checkpoints (they're gitignored)
git commit -m "docs: add the Ryzen long-run results"
git push -u origin results/ryzen-long-runs
```

Confirm the push succeeded, and report the branch name and the last commit hash. Leave the checkpoints under `checkpoints/ntuple/ryzen-*` on the machine; they're needed for follow-up work.
