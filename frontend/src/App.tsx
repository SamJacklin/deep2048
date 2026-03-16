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

function useBenchmark() {
  const [benchmark, setBenchmark] = useState<BenchmarkRow[] | null>(null);
  useEffect(() => {
    fetch('/benchmark.json')
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
      const res = await fetch(`/replays/${name}.json`);
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

export default function App() {
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
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      {/* Header */}
      <header className="border-b border-zinc-800 sticky top-0 z-10 bg-zinc-950/90 backdrop-blur">
        <div className="max-w-5xl mx-auto px-6 h-14 flex items-center justify-between">
          <span className="font-mono text-sm text-zinc-400">
            sam<span className="text-blue-500">.</span>jacklin
            <span className="text-zinc-700 mx-2">/</span>
            <span className="text-zinc-300">deep-2048</span>
          </span>
          <a
            href="https://github.com/samjacklin/deep2048"
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors font-mono"
          >
            github ↗
          </a>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-10 flex flex-col gap-10">
        {/* Intro */}
        <section className="flex flex-col gap-3 max-w-2xl">
          <h1 className="text-2xl font-bold text-zinc-100">
            Deep RL — 2048
          </h1>
          <p className="text-sm text-zinc-400 leading-relaxed">
            A Deep Q-Network trained for 925K steps with prioritised experience replay, n-step
            returns, and a dueling multi-path CNN — reaching 512 in 61% of games and 1024 in 9%.
            Watch the strategy evolve across training, then compare the trained agent against
            expectimax search with and without the learned evaluator.
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
          <div className="flex items-center justify-center py-32 text-zinc-600 text-sm">
            Loading replay…
          </div>
        ) : replay && currentFrame ? (
          <section className="flex flex-col gap-6">
            {/* Checkpoint description */}
            <div className="flex items-center gap-3">
              <span className="text-sm font-bold text-zinc-200">
                {replay.checkpoint_label}
              </span>
              <span className="text-sm text-zinc-500">{replay.checkpoint_description}</span>
            </div>

            {/* Main 3-column layout */}
            <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_1fr] gap-6 items-start">
              {/* Left: Stats */}
              <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-5">
                <StatsPanel
                  replay={replay}
                  frame={currentFrame}
                />
              </div>

              {/* Centre: Board + Controls */}
              <div className="flex flex-col items-center gap-4">
                <Board
                  board={currentFrame.board}
                  prevBoard={prevFrame?.board ?? null}
                />
                <div className="w-full max-w-[360px] rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
                  <Controls player={player} />
                </div>
              </div>

              {/* Right: Q-Values */}
              <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-5">
                <QValueChart
                  qValues={currentFrame.q_values}
                  chosenAction={currentFrame.action}
                  validActions={currentFrame.valid_actions}
                  label={replay.score_label}
                />

                {/* Architecture note */}
                <div className="mt-6 pt-4 border-t border-zinc-800 flex flex-col gap-2">
                  <p className="text-xs text-zinc-500 uppercase tracking-widest">Architecture</p>
                  <ul className="text-xs text-zinc-500 space-y-1 leading-relaxed">
                    <li className="text-zinc-400">Dueling Double DQN</li>
                    <li>3× Conv2d (3×3) + row/col paths</li>
                    <li>1024-dim hidden, 4 action outputs</li>
                    <li>PER + n-step returns (n=5)</li>
                    <li>Cosine LR · 925K steps</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Game-over banner */}
            {currentFrame.valid_actions.length === 0 && (
              <div className="rounded-lg border border-zinc-700 bg-zinc-900/80 px-5 py-4 flex items-center justify-between">
                <div>
                  <span className="text-sm font-bold text-zinc-200">Game over</span>
                  <span className="text-sm text-zinc-500 ml-3">
                    Final score: <span className="text-zinc-300 tabular-nums">{replay.final_score.toLocaleString()}</span>
                    {' · '}
                    Best tile: <span className="text-zinc-300 tabular-nums">{replay.max_tile_achieved}</span>
                  </span>
                </div>
                <button
                  onClick={player.reset}
                  className="text-xs text-blue-400 hover:text-blue-300 transition-colors"
                >
                  Replay ↺
                </button>
              </div>
            )}
          </section>
        ) : (
          <div className="flex items-center justify-center py-32 text-zinc-600 text-sm">
            Replay not available — run{' '}
            <code className="mx-1 text-zinc-500">uv run python scripts/generate_replays.py</code>
            {' '}to generate data.
          </div>
        )}

        {/* Benchmark comparison */}
        {benchmark && (
          <section className="flex flex-col gap-3">
            <h2 className="text-xs text-zinc-500 uppercase tracking-widest">Performance Comparison</h2>
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-5">
              <BenchmarkTable
                rows={benchmark}
                highlighted={CHECKPOINTS.find(cp => cp.name === selected)?.benchmarkLabel}
              />
            </div>
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-800 mt-10 py-8 px-6">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <span className="font-mono text-xs text-zinc-600">
            sam<span className="text-blue-500">.</span>jacklin
          </span>
          <span className="font-mono text-xs text-zinc-700">
            DQN · Expectimax · PyTorch · Vite · React
          </span>
        </div>
      </footer>
    </div>
  );
}
