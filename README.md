# Ryzen long runs: instructions

These are instructions for a Claude agent on the Ryzen machine (AMD Ryzen 7 9850X3D, 8 cores / 16 threads, 32 GB RAM, Windows). Follow them exactly and in order.
- **Don't** change code, open pull requests, or merge branches.
- **If a step fails,** record what happened in the results file (step 10), carry on where it's safe, and push what you have at the end.

## What this is

Two long training runs of the 2048 n-tuple agent, from scratch to 10,000,000 games each:

| Arm | Run name | What it does |
|---|---|---|
| `recipe` | `ryzen-recipe` | This project's current best method: temporal coherence (TC) at 0.02, downgrading from 16384, late-game starts. Its TC sums are reset once, at 1.5M games. |
| `optimism` | `ryzen-optimism` | Guei's recipe on top of the same setup. Every weight starts at 5,000 (320,000 per board), and training runs at a fixed rate of 0.0016 with TC off until 9M games. Then TC at 0.02 for the last 1M. |

- **While training:** each run evaluates itself greedily every 250k games, on 1,000 seeded games.
- **At the end:** exact depth-3 evaluations at the kept points: 1.5M, 5M and 10M for `recipe`, and 5M, 9M and 10M for `optimism`.
- **The scripts are on branch `feat/fast-tc`:** `scripts/run_ryzen_arms.sh`, `scripts/benchmark_speed.sh` and `scripts/summarise_runs.py`.

The runs take a long time, probably 1–3 days with both running together. The scripts resume where they stopped if interrupted.

## Part A: Windows setup

Run these in Windows, in **PowerShell as administrator**. A human may need to approve prompts and do the reboot.

### A1. Install WSL2 with Ubuntu

```powershell
wsl --install -d Ubuntu-24.04
```

- **Reboot** if asked.
- **Create the Linux user:** when Ubuntu first starts, create a username and password. Ubuntu can be opened from the Start menu, or by running `wsl` in PowerShell.
- **If WSL is already installed,** check it's version 2 with `wsl -l -v`. If it shows version 1, run `wsl --set-version Ubuntu-24.04 2`.

### A2. Give WSL enough memory and every CPU

WSL limits itself to half the machine's memory by default. Create or replace `%UserProfile%\.wslconfig`:

```powershell
@"
[wsl2]
memory=28GB
processors=16
swap=8GB
"@ | Set-Content -Path "$env:UserProfile\.wslconfig"
wsl --shutdown
```

Then reopen Ubuntu. Inside it, `nproc` should print 16, and `free -g` should show about 27 GB total.

### A3. Keep the machine awake

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change monitor-timeout-ac 30
```

- **Pause Windows Update for a week:** Settings → Windows Update → Pause updates.
- **Keep the machine plugged in.**

### A4. GitHub access from WSL (for pushing results at the end)

Run inside **Ubuntu**:

```bash
sudo apt-get update && sudo apt-get install -y gh
gh auth login      # choose GitHub.com, HTTPS, and log in with a browser or token
gh auth setup-git
```

A human may need to complete the login. Check it worked: `gh auth status` should show the account logged in.

**Everything below runs inside the Ubuntu (WSL) shell, not PowerShell.**

## Part B: Get the code and check it works

### B1. Install tools

```bash
sudo apt-get update && sudo apt-get install -y git build-essential procps
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

### B2. Get the code

Clone into the Linux filesystem (your home directory), **not** under `/mnt/c`, which is much slower:

```bash
cd ~
git clone https://github.com/SamJacklin/deep2048.git
cd ~/deep2048
git checkout feat/fast-tc
git log --oneline -1    # expect: aa522aa feat: add a speed benchmark for comparing machines
uv sync
uv run pytest -q        # expect: 85 passed
```

If the commit at the top of `feat/fast-tc` is newer than `aa522aa`, use it. Record the hash you used.

### B3. Record the machine

Save this output for the results file:

```bash
lscpu | grep -E "Model name|^CPU\(s\)|Thread"
free -g | head -2
nproc
```

### B4. Smoke test both arms (about 10 minutes)

These are short versions of the real runs, 40,000 games each, to check everything works end to end on this machine:

```bash
cd ~/deep2048
TOTAL=40000 EVAL_EVERY=10000 EVAL_GAMES=20 THREADS=8 NTUPLE_THREADS_AFTER=2000 scripts/run_ryzen_arms.sh recipe   2>&1 | tail -8
TOTAL=40000 EVAL_EVERY=10000 EVAL_GAMES=20 THREADS=8 NTUPLE_THREADS_AFTER=2000 scripts/run_ryzen_arms.sh optimism 2>&1 | tail -8
grep -h "sums reset" experiments/smoke-ryzen-recipe/train.log
grep -h "never-updated" experiments/smoke-ryzen-optimism/train.log
```

**Expected:**
- Each arm prints greedy evaluation lines (`N-tuple TD (10,000 games …`), three depth-3 lines, and finally `smoke-ryzen-recipe finished` or `smoke-ryzen-optimism finished`.
- The recipe log contains `Temporal coherence sums reset: every weight starts back at the full rate`.
- The optimism log contains `134,217,728 never-updated weights (100.0%) set to 5000`.
- Smoke-test scores are tiny and noisy; that's fine.

If anything is missing or errors, record it and stop: don't start the long runs.

Then delete the smoke runs:

```bash
rm -rf checkpoints/ntuple/smoke-ryzen-* experiments/smoke-ryzen-*
```

### B5. Speed benchmark (about 10 minutes)

**Nothing else may be running.** Run:

```bash
cd ~/deep2048
scripts/benchmark_speed.sh 2>&1 | grep -v VIRTUAL_ENV | tee /tmp/benchmark.txt
```

It trains from scratch on 1, 8 and 16 threads and times two evaluations. The same script on the development machine, an Apple M4 MacBook (10 cores: 4 performance and 6 efficiency, 16 GB), gave:

| Benchmark | Apple M4 (development machine) |
|---|---:|
| Training, 1 thread, 20,000 games | 0.53M moves/s (0.49M on a second run) |
| Training, 8 threads, 40,000 games | 2.33M moves/s (2.37M on a second run) |
| Training, 10 threads (M4 maximum), 40,000 games | 2.58M moves/s |
| Greedy evaluation, 1,000 games, all CPUs | 4 s, average score **71,753** |
| Depth-3 evaluation, 100 games, all CPUs | 57 s, average score **155,098** |

- **Speed:** record the Ryzen's numbers next to these, including 16 threads.
- **The two average scores are a check:** single-threaded training is fully seeded, so the Ryzen should reproduce 71,753 and 155,098 exactly. If they differ slightly, the CPUs round some arithmetic differently. Record it, but it isn't a problem.

Both long runs use 8 threads each, so the two of them together use all 16 hardware threads. Keep that setting whatever the benchmark shows.

## Part C: The long runs

### C1. Start both runs

Start them detached, so they keep running if this session ends:

```bash
cd ~/deep2048
THREADS=8 setsid nohup scripts/run_ryzen_arms.sh recipe   > ryzen-recipe.log 2>&1 < /dev/null &
THREADS=8 setsid nohup scripts/run_ryzen_arms.sh optimism > ryzen-optimism.log 2>&1 < /dev/null &
sleep 60; tail -3 experiments/ryzen-recipe/train.log experiments/ryzen-optimism/train.log
```

Both train logs should show progress lines (`games … avg score …`). Record the start time.

**What to expect along the way:**
- **`optimism` scores much lower than `recipe` for millions of games.** That's normal: optimistic weights take a long time to settle. Don't stop it.
- **`recipe` drops briefly just after 1.5M games,** when its TC sums are reset.
- **Disk:** each run keeps three 1.5 GB snapshots, plus one checkpoint while training.

### C2. Monitor until both finish

Check every hour or so:

```bash
cd ~/deep2048
tail -2 experiments/ryzen-recipe/train.log experiments/ryzen-optimism/train.log
grep -h "^N-tuple" ryzen-recipe.log ryzen-optimism.log | tail -6
pgrep -fl run_ryzen_arms || echo "no run is active"
```

- **Done:** a run is finished when its log (`ryzen-recipe.log` or `ryzen-optimism.log`) ends with `ryzen-recipe finished` or `ryzen-optimism finished`.
- **If the machine restarted, or a run died without that line:** rerun exactly the same command from C1 for that arm. It resumes from its last checkpoint and repeats any missing evaluations. A run that already finished just reports so.
- **Record** any interruptions, with times.

## Part D: Results

### D1. Write the results

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
1. **The machine:** CPU, RAM, cores (B3).
2. **The commit:** the hash used (B2).
3. **The smoke test:** what it printed and whether it passed (B4).
4. **The benchmark:** the Ryzen's numbers from `/tmp/benchmark.txt` beside the M4's table (B5), including whether the two scores matched.
5. **Timings:** start and end times of each run, and any interruptions or problems.
6. **Results:** the tables from `/tmp/ryzen-tables.md`, pasted unchanged.
7. **Charts:** links to `figures/ryzen-arms.png`, `ryzen-recipe/training_progress.png` and `ryzen-optimism/training_progress.png`.

Don't interpret the results; the tables are what's needed.

### D2. Push the results

```bash
cd ~/deep2048
git checkout -b results/ryzen-long-runs
git add experiments/ryzen-recipe experiments/ryzen-optimism experiments/figures/ryzen-arms.png experiments/ryzen-RESULTS.md
git status --short     # check: only those paths, no checkpoints (they're gitignored)
git commit -m "docs: add the Ryzen long-run results"
git push -u origin results/ryzen-long-runs
```

Confirm the push succeeded, and report the branch name and the last commit hash. Leave the checkpoints under `checkpoints/ntuple/ryzen-*` on the machine; they're needed for follow-up work.

**If the long runs can't be started or finished,** still push what you have: the results file with the smoke test and benchmark. Use the same branch name.
