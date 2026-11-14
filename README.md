# Stage 7 on the Ryzen: instructions

These are instructions for a Claude agent on the Ryzen machine (AMD Ryzen 7 9850X3D, 8 cores / 16 threads, 32 GB RAM, Windows with WSL2 Ubuntu, set up for the earlier long runs). Follow them exactly and in order.
- **Don't** change code, open pull requests, or merge branches.
- **Don't interpret the results.** The reports and timings are what's needed; the analysis happens elsewhere.
- **If a step fails,** record what happened in the results file (D2), carry on only where these instructions say it's safe, and push what you have at the end.

## What this is

Stage 7 is a diagnostic of the late game: why the project's best network converts so few first-16384 positions into a 32768. **No training happens.** It plays a large number of continuations with the frozen 2M-game network from genuine first-16384 positions and writes reports.

- **Part 1:** 64 greedy continuations from each of 5,000 entry positions, and 8 depth-3 continuations from 1,000 of them: does the entry position matter, and which board features predict it?
- **Part 2:** greedy games from entry positions, then four policies (greedy; depth 3; depth 3 for 200 moves; pruned depth 5 for 200 moves) replayed from points along them: when can failing games be rescued? A small pilot runs first, then an expansion chosen by a fixed rule written into `p2-plan.json`, capped at 10 hours.
- **Part 3 is not run here.** Don't start it.

The code is on branch **`feat/late-game-anatomy`**: `scripts/run_late_game_anatomy.sh` runs everything in order and resumes where it stopped if interrupted. The protocol is in `experiments/late-game-anatomy/README.md` on that branch.

Expected time: Part 1 about 1–2 hours, the Part 2 pilot about 1 hour, the expansion up to 10 hours (its estimate is in `p2-plan.json`).

## Part A: Check the machine

Run in **PowerShell**:

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
```

- **Pause Windows Update for a week:** Settings → Windows Update → Pause updates. **This needs a human.** Last time an update restarted the PC overnight. Record whether it was paused.
- **Keep the machine plugged in.**

Everything below runs inside the **Ubuntu (WSL) shell**. Check it still has all 16 threads and GitHub access:

```bash
nproc               # expect 16
gh auth status      # expect logged in
```

## Part B: Get the code and the network

### B1. Get the code

```bash
cd ~/deep2048
git status --short          # expect nothing that matters; record anything listed
git fetch origin
git checkout feat/late-game-anatomy
git pull --ff-only
git log --oneline -1        # expect 5fd1b15 style: wrap two long report lines in stage 7's script (or newer: record the hash)
uv sync
uv run pytest -q            # expect: 129 passed
```

### B2. Copy the network from the Mac

The network is on the development Mac. **A human must do two things:** turn on Remote Login on the Mac (System Settings → General → Sharing → Remote Login), and type the Mac's login password when rsync asks for it.

```bash
mkdir -p ~/deep2048/checkpoints/ntuple/decision-8x6-tc-reset
rsync -avP samjacklin@192.168.1.86:/Users/samjacklin/Documents/deep2048/checkpoints/ntuple/decision-8x6-tc-reset/games_2000000.npz \
      ~/deep2048/checkpoints/ntuple/decision-8x6-tc-reset/
sha256sum ~/deep2048/checkpoints/ntuple/decision-8x6-tc-reset/games_2000000.npz
```

- **The hash must be exactly** `8f117f2b449ddba0c48e8a1d71b1afe4720b7b8d5c168b88ac9fbfe8b0dde513`. If it differs, delete the file and copy it again. Never run with a file whose hash differs.
- **If rsync can't connect:** the Mac's address may have changed. A human can find it on the Mac with `ipconfig getifaddr en0`; use that instead of `192.168.1.86`. If it still can't connect, a human can copy the file across another way (USB stick, cloud drive) to the same path, and the hash check still applies.
- The file is 1.5 GB; over a home network it takes a minute or two.

### B3. Record the machine

Save this output for the results file:

```bash
lscpu | grep -E "Model name|^CPU\(s\)|Thread"
free -g | head -2
df -h ~ | tail -1
```

### B4. Smoke test (about 10 minutes)

A tiny version of the whole run, end to end:

```bash
cd ~/deep2048
SMOKE=1 scripts/run_late_game_anatomy.sh 2>&1 | tail -5
cat experiments/late-game-anatomy/smoke/stage6-replay-check.json
ls experiments/late-game-anatomy/smoke/
```

**Expected:**
- The last line is `… stage 7 finished`.
- `stage6-replay-check.json` says `"continuations 0 and 1 identical to stage 6's control": true`. This checks the Ryzen plays exactly as the Mac did.
- The smoke folder holds `part1.md`, `part2-pilot.md`, `p2-plan.json`, `part2-final.md`, `costs.jsonl` and `manifest.json`. The numbers in them are meaningless at this size.

**If the replay check is `false`, or anything errors:** record the full output and **stop: don't start the real run.** Push the results file (D2, D3) so the problem can be looked at.

Then delete the smoke outputs:

```bash
rm -rf experiments/late-game-anatomy/smoke output/late-game-anatomy-smoke
```

## Part C: The run

### C1. Start it

Start it detached, so it keeps running if this session ends:

```bash
cd ~/deep2048
setsid nohup scripts/run_late_game_anatomy.sh > anatomy.log 2>&1 < /dev/null &
sleep 120; tail -3 anatomy.log
```

The log should show `starting manifest`, `finished manifest`, `starting p1-greedy`, then a JSON line per chunk of 250 positions. Record the start time.

### C2. Monitor until it finishes

Check every 30–60 minutes:

```bash
cd ~/deep2048
grep -E "starting|finished" anatomy.log | tail -4
tail -1 experiments/late-game-anatomy/costs.jsonl
pgrep -f run_late_game_anatomy >/dev/null && echo running || echo "not running"
```

- **Done** when `anatomy.log` ends with `stage 7 finished`.
- **If the machine restarted, or the run died without that line:** check the end of `anatomy.log` for an error and record it. If it's not a code error (a restart, say), rerun exactly the C1 command. It resumes, skipping finished work. If it's a code error, don't rerun: record it, and push what exists (D2, D3).
- **If the last line is `Only N GB free: need 5`:** free space by deleting nothing but old smoke outputs; otherwise record it and push.
- **Record** any interruptions, with times.

### C3. Interim push after the pilot

When `anatomy.log` shows `finished p2-plan`, push the reports so far, so Part 1 and the pilot can be reviewed while the expansion runs:

```bash
cd ~/deep2048
git checkout -b results/late-game-anatomy-ryzen
git add experiments/late-game-anatomy
git status --short      # check: only files under experiments/late-game-anatomy/
git commit -m "docs: add stage 7's Part 1 and Part 2 pilot results from the Ryzen"
git push -u origin results/late-game-anatomy-ryzen
```

The run carries on meanwhile; that's fine. Later commits go on this same branch.

## Part D: Results

### D1. Collect

When `anatomy.log` ends with `stage 7 finished`:

```bash
cd ~/deep2048
cp anatomy.log experiments/late-game-anatomy/run.log
uv run python -c "
import json, collections
rows = [json.loads(l) for l in open('experiments/late-game-anatomy/costs.jsonl')]
total = collections.defaultdict(lambda: [0.0, 0.0, 0, 0])
for r in rows:
    parts = r['step'].split()
    key = f'p2-run {parts[1]} policy {parts[3]}' if parts[0] == 'p2-run' else r['step'].split(' positions')[0]
    t = total[key]; t[0] += r['wall_s']; t[1] += r['cpu_s']; t[2] += r.get('continuations', 0); t[3] += r.get('moves', 0)
print('| Step | Wall hours | CPU hours | Continuations | Moves |'); print('|---|---:|---:|---:|---:|')
for k, (w, c, n, m) in total.items(): print(f'| {k} | {w/3600:.2f} | {c/3600:.2f} | {n:,} | {m:,} |')
" > /tmp/costs.md
cat /tmp/costs.md
```

### D2. Write the results file

Create `experiments/late-game-anatomy/RYZEN-RESULTS.md` containing:
1. **The machine:** the B3 output.
2. **The commit:** the hash from B1, and the pytest result.
3. **The network copy:** how it was copied, and the sha256 it showed.
4. **The smoke test:** what it printed, including the replay check.
5. **Windows Update:** paused or not.
6. **Timings:** start and end times, and any interruptions or problems, with times.
7. **Compute:** the table from `/tmp/costs.md`.
8. **Results:** the full contents of `part1.md`, `part2-pilot.md`, `p2-plan.json` and `part2-final.md`, pasted unchanged, in that order.

Don't interpret the results.

### D3. Push

```bash
cd ~/deep2048
git checkout results/late-game-anatomy-ryzen 2>/dev/null || git checkout -b results/late-game-anatomy-ryzen
git add experiments/late-game-anatomy
git status --short      # check: only files under experiments/late-game-anatomy/ (outputs and checkpoints are gitignored)
git commit -m "docs: add stage 7's Ryzen results"
git push -u origin results/late-game-anatomy-ryzen
```

Confirm the push succeeded, and report the branch name and the last commit hash.
- **Leave** `output/late-game-anatomy/` and the checkpoint on the machine: Part 3 needs them.
- **If the run couldn't be started or finished,** still push what you have, with the results file explaining why, on the same branch.
