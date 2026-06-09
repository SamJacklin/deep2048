import { useState, useEffect, useCallback } from 'react';
import type { Replay, BenchmarkRow } from './types';
import { CHECKPOINTS } from './types';
import { useReplayPlayer } from './hooks/useReplayPlayer';
import Board from './components/Board';
import Controls from './components/Controls';
import QValueChart from './components/QValueChart';
import StatsPanel from './components/StatsPanel';
import CheckpointSelector from './components/CheckpointSelector';
import BenchmarkTable from './components/BenchmarkTable';
import ArchitectureSection from './components/ArchitectureSection';
import { ThemeProvider } from './toolkit/frontend/portfolio/components/ThemeContext';
import { ThemeTransition } from './toolkit/frontend/portfolio/components/ThemeTransition';
import Navbar from './toolkit/frontend/portfolio/components/Navbar/Navbar';

const navLogo = (
  <>
    sam<span className="text-blue-500">.</span>jacklin
    <span className="text-zinc-400 dark:text-zinc-700 mx-2">/</span>
    <span className="text-zinc-600 dark:text-zinc-300">deep-2048</span>
  </>
);

const navLinks = [
  { label: 'github ↗', href: 'https://github.com/samjacklin/deep2048', scroll: false },
];

function useBenchmark() {
  const [benchmark, setBenchmark] = useState<BenchmarkRow[] | null>(null);
  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}benchmark.json`)
      .then(r => r.ok ? r.json() : null)
      .then(d => d && setBenchmark(d))
      .catch(() => {});
  }, []);
  return benchmark;
}

function useReplays() {
  const [replays, setReplays] = useState<Record<string, Replay | null>>({});
  const [loading, setLoading] = useState<Record<string, boolean>>({});

  const load = useCallback(async (name: string) => {
    if (replays[name] !== undefined) return;
    setLoading(l => ({ ...l, [name]: true }));
    try {
      const res = await fetch(`${import.meta.env.BASE_URL}replays/${name}.json`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: Replay = await res.json();
      setReplays(r => ({ ...r, [name]: data }));
    } catch {
      setReplays(r => ({ ...r, [name]: null }));
    } finally {
      setLoading(l => ({ ...l, [name]: false }));
    }
  }, [replays]);

  return { replays, loading, load };
}

function AppContent() {
  const [selected, setSelected] = useState('best');
  const { replays, loading, load } = useReplays();
  const benchmark = useBenchmark();

  // Pre-load all checkpoints in background
  useEffect(() => {
    CHECKPOINTS.forEach(cp => load(cp.name));
  }, [load]);

  const replay = replays[selected] ?? null;
  const isLoading = loading[selected] ?? false;
  const player = useReplayPlayer(replay);
  const { currentFrame, frameIndex } = player;
  const prevFrame = replay && frameIndex > 0 ? replay.frames[frameIndex - 1] : null;

  return (
    <ThemeTransition>
      <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100">
        <Navbar logo={navLogo} links={navLinks} />

        <main className="max-w-5xl mx-auto px-6 py-10 flex flex-col gap-10 pt-24">
          {/* Intro */}
          <section className="flex flex-col gap-3 max-w-2xl">
            <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">
              An AI that taught itself to play 2048
            </h1>
            <p className="text-sm text-zinc-500 dark:text-zinc-400 leading-relaxed">
              No rules, no hints. Just a neural network playing millions of games until it figured
              out what works. Pick a checkpoint below to watch the strategy evolve, or see how it
              compares to a classical search algorithm.
            </p>
            <p className="text-sm text-zinc-400 dark:text-zinc-500 leading-relaxed">
              Under the hood: a Dueling Double DQN with prioritised experience replay, n-step
              returns, and a CNN that reads rows, columns, and local patterns in parallel.
              925K training steps. 512 tile in 61% of games.
            </p>
          </section>

          {/* Checkpoint selector */}
          <section>
            <CheckpointSelector
              checkpoints={CHECKPOINTS}
              selected={selected}
              replays={replays}
              loading={loading}
              benchmark={benchmark}
              onSelect={name => { setSelected(name); load(name); }}
            />
          </section>

          {/* Game viewer */}
          {isLoading ? (
            <div className="flex items-center justify-center py-32 text-zinc-400 dark:text-zinc-600 text-sm">
              Loading replay…
            </div>
          ) : replay && currentFrame ? (
            <section className="flex flex-col gap-6">
              {/* Checkpoint description */}
              <div className="flex items-center gap-3">
                <span className="text-sm font-bold text-zinc-800 dark:text-zinc-200">
                  {replay.checkpoint_label}
                </span>
                <span className="text-sm text-zinc-500">{replay.checkpoint_description}</span>
              </div>

              {/* Board + Controls */}
              <div className="flex flex-col items-center gap-4">
                <Board
                  board={currentFrame.board}
                  prevBoard={prevFrame?.board ?? null}
                />
                <div className="w-full max-w-[360px] rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white/60 dark:bg-zinc-900/60 p-4">
                  <Controls player={player} />
                </div>
              </div>

              {/* Stats + Q-Values */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white/60 dark:bg-zinc-900/60 p-5">
                  <StatsPanel
                    replay={replay}
                    frame={currentFrame}
                  />
                </div>
                <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white/60 dark:bg-zinc-900/60 p-5">
                  <QValueChart
                    qValues={currentFrame.q_values}
                    chosenAction={currentFrame.action}
                    validActions={currentFrame.valid_actions}
                    label={replay.score_label}
                  />
                </div>
              </div>

              {/* Game-over banner */}
              {currentFrame.valid_actions.length === 0 && (
                <div className="rounded-lg border border-zinc-300 dark:border-zinc-700 bg-white/80 dark:bg-zinc-900/80 px-5 py-4 flex items-center justify-between">
                  <div>
                    <span className="text-sm font-bold text-zinc-800 dark:text-zinc-200">Game over</span>
                    <span className="text-sm text-zinc-500 ml-3">
                      Final score: <span className="text-zinc-700 dark:text-zinc-300 tabular-nums">{replay.final_score.toLocaleString()}</span>
                      {' · '}
                      Best tile: <span className="text-zinc-700 dark:text-zinc-300 tabular-nums">{replay.max_tile_achieved}</span>
                    </span>
                  </div>
                  <button
                    onClick={player.reset}
                    className="text-xs text-blue-500 hover:text-blue-400 transition-colors"
                  >
                    Replay ↺
                  </button>
                </div>
              )}
            </section>
          ) : (
            <div className="flex items-center justify-center py-32 text-zinc-400 dark:text-zinc-600 text-sm">
              Replay not available — run{' '}
              <code className="mx-1 text-zinc-500">uv run python scripts/generate_replays.py</code>
              {' '}to generate data.
            </div>
          )}

          {/* Benchmark comparison */}
          {benchmark && (
            <section className="flex flex-col gap-3">
              <h2 className="text-xs text-zinc-400 dark:text-zinc-500 uppercase tracking-widest">How it stacks up</h2>
              <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white/60 dark:bg-zinc-900/60 p-5">
                <BenchmarkTable
                  rows={benchmark}
                  highlighted={CHECKPOINTS.find(cp => cp.name === selected)?.benchmarkLabel}
                />
              </div>
            </section>
          )}

          <ArchitectureSection />
        </main>

        {/* Footer */}
        <footer className="border-t border-zinc-200 dark:border-zinc-800 mt-10 py-8 px-6">
          <div className="max-w-5xl mx-auto flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <span className="font-mono text-xs text-zinc-400 dark:text-zinc-600">
              sam<span className="text-blue-500">.</span>jacklin
            </span>
            <span className="font-mono text-xs text-zinc-300 dark:text-zinc-700">
              DQN · Expectimax · PyTorch · Vite · React
            </span>
          </div>
        </footer>
      </div>
    </ThemeTransition>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <AppContent />
    </ThemeProvider>
  );
}
