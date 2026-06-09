import { useState } from 'react';
import type { BenchmarkRow, CheckpointMeta, Replay } from '../types';

interface CheckpointSelectorProps {
  checkpoints: CheckpointMeta[];
  selected: string;
  replays: Record<string, Replay | null>;
  loading: Record<string, boolean>;
  benchmark: BenchmarkRow[] | null;
  onSelect: (name: string) => void;
}

type Tab = 'progression' | 'comparison';

// --- Progression tab ---

function ProgressionStep({
  cp,
  isSelected,
  replay,
  isLoading,
  benchmarkRow,
  onSelect,
  stepNumber,
}: {
  cp: CheckpointMeta;
  isSelected: boolean;
  replay: Replay | null | undefined;
  isLoading: boolean;
  benchmarkRow: BenchmarkRow | null;
  onSelect: () => void;
  stepNumber: number;
}) {
  const avgScore = benchmarkRow?.avg_score ?? replay?.average_score ?? null;
  const pct512 = benchmarkRow?.tile_pcts['512'] ?? null;

  return (
    <button
      onClick={onSelect}
      className={`
        relative text-left p-4 rounded-lg border transition-all flex-1
        ${isSelected
          ? 'border-blue-500 bg-blue-50 dark:bg-blue-950/40 text-zinc-900 dark:text-zinc-100'
          : 'border-zinc-200 dark:border-zinc-800 bg-white/60 dark:bg-zinc-900/60 text-zinc-500 dark:text-zinc-400 hover:border-zinc-400 dark:hover:border-zinc-600 hover:text-zinc-700 dark:hover:text-zinc-200'}
      `}
    >
      <div className={`text-xs mb-2 font-mono tabular-nums ${isSelected ? 'text-blue-500' : 'text-zinc-400 dark:text-zinc-600'}`}>
        {String(stepNumber).padStart(2, '0')}
      </div>

      <div className={`text-sm font-bold mb-1 ${isSelected ? 'text-blue-500 dark:text-blue-400' : 'text-zinc-700 dark:text-zinc-300'}`}>
        {cp.label}
      </div>
      <div className="text-xs text-zinc-500 leading-snug mb-3">{cp.description}</div>

      {isLoading ? (
        <div className="text-xs text-zinc-400 dark:text-zinc-600">Loading…</div>
      ) : avgScore !== null ? (
        <div className="flex flex-col gap-1">
          <div className="text-xs text-zinc-400 dark:text-zinc-600">
            avg score{' '}
            <span className={`tabular-nums font-medium ${isSelected ? 'text-zinc-700 dark:text-zinc-200' : 'text-zinc-500 dark:text-zinc-400'}`}>
              {avgScore.toLocaleString()}
            </span>
          </div>
          {pct512 !== null && (
            <div className="text-xs text-zinc-400 dark:text-zinc-600">
              ≥512{' '}
              <span className={`tabular-nums font-medium ${isSelected ? 'text-zinc-700 dark:text-zinc-200' : 'text-zinc-500 dark:text-zinc-400'}`}>
                {pct512.toFixed(0)}%
              </span>
            </div>
          )}
        </div>
      ) : (
        <div className="text-xs text-zinc-300 dark:text-zinc-700">No data</div>
      )}
    </button>
  );
}

function ProgressionArrow({ isImproving }: { isImproving: boolean }) {
  const color = isImproving ? 'text-blue-500 dark:text-blue-600' : 'text-zinc-300 dark:text-zinc-700';
  return (
    <>
      <div className={`hidden md:flex items-center justify-center flex-shrink-0 w-8 ${color}`}>
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
          <path d="M5 12H19M19 12L13 6M19 12L13 18" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </div>
      <div className={`md:hidden flex justify-center py-1 ${color}`}>
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
          <path d="M12 5V19M12 19L6 13M12 19L18 13" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </div>
    </>
  );
}

function ProgressionTab({
  checkpoints,
  selected,
  replays,
  loading,
  benchmark,
  onSelect,
}: Omit<CheckpointSelectorProps, 'onSelect'> & { onSelect: (name: string) => void }) {
  const steps = checkpoints.filter(cp => cp.group === 'progression');

  const getAvgScore = (cp: CheckpointMeta): number | null => {
    const benchmarkRow = cp.benchmarkLabel
      ? (benchmark?.find(r => r.label === cp.benchmarkLabel) ?? null)
      : null;
    return benchmarkRow?.avg_score ?? replays[cp.name]?.average_score ?? null;
  };

  return (
    <div className="flex flex-col gap-4">
      <p className="text-xs text-zinc-500 leading-relaxed">
        Each checkpoint is a snapshot of the model at a different point in training — 100K to 925K steps.
        Watch the strategy evolve from chaotic tile merging to corner-anchored chains.
      </p>

      <div className="flex flex-col md:flex-row md:items-stretch gap-2 md:gap-0">
        {steps.map((cp, i) => {
          const benchmarkRow = cp.benchmarkLabel
            ? (benchmark?.find(r => r.label === cp.benchmarkLabel) ?? null)
            : null;
          const prevScore = i > 0 ? getAvgScore(steps[i - 1]) : null;
          const thisScore = getAvgScore(cp);
          const isImproving = prevScore !== null && thisScore !== null && thisScore > prevScore;

          return (
            <div key={cp.name} className="flex flex-col md:flex-row md:items-center flex-1 md:gap-0">
              {i > 0 && <ProgressionArrow isImproving={isImproving} />}
              <ProgressionStep
                cp={cp}
                isSelected={selected === cp.name}
                replay={replays[cp.name]}
                isLoading={loading[cp.name] ?? false}
                benchmarkRow={benchmarkRow}
                onSelect={() => onSelect(cp.name)}
                stepNumber={i + 1}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}

// --- Comparison tab ---

function StatCell({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="flex flex-col items-center gap-0.5">
      <span className="text-xs text-zinc-400 dark:text-zinc-600 uppercase tracking-wide">{label}</span>
      <span className={`text-2xl font-bold tabular-nums ${highlight ? 'text-blue-500 dark:text-blue-300' : 'text-zinc-800 dark:text-zinc-100'}`}>
        {value}
      </span>
    </div>
  );
}

function ComparisonCard({
  cp,
  isSelected,
  benchmarkRow,
  onSelect,
  side,
  winnerOn,
}: {
  cp: CheckpointMeta;
  isSelected: boolean;
  benchmarkRow: BenchmarkRow | null;
  onSelect: () => void;
  side: 'left' | 'right';
  winnerOn: Record<string, 'left' | 'right' | 'tie'>;
}) {
  const isLeft = side === 'left';

  return (
    <button
      onClick={onSelect}
      className={`
        relative text-left p-5 rounded-xl border transition-all w-full
        ${isSelected
          ? 'border-blue-500 bg-blue-50 dark:bg-blue-950/40'
          : 'border-zinc-200 dark:border-zinc-800 bg-white/60 dark:bg-zinc-900/60 hover:border-zinc-400 dark:hover:border-zinc-600'}
      `}
    >
      {isSelected && (
        <div className={`absolute top-3 ${isLeft ? 'right-3' : 'left-3'} text-xs text-blue-500 font-mono`}>
          ● watching
        </div>
      )}

      <div className={`text-xs font-mono mb-1 ${isLeft ? '' : 'md:text-right'} ${isSelected ? 'text-blue-500' : 'text-zinc-400 dark:text-zinc-600'}`}>
        {cp.group === 'comparison' ? 'classical' : 'learned'}
      </div>

      <div className={`text-lg font-bold mb-1 ${isSelected ? 'text-blue-500 dark:text-blue-400' : 'text-zinc-700 dark:text-zinc-200'}`}>
        {cp.label}
      </div>
      <div className="text-xs text-zinc-500 leading-snug mb-4">{cp.description}</div>

      {benchmarkRow ? (
        <div className={`flex gap-4 ${isLeft ? '' : 'md:justify-end'}`}>
          {(['avg_score', '512', '1024'] as const).map(key => {
            const winner = winnerOn[key];
            const iWin = (side === 'left' && winner === 'left') || (side === 'right' && winner === 'right');
            const value = key === 'avg_score'
              ? benchmarkRow.avg_score.toLocaleString()
              : `${benchmarkRow.tile_pcts[key as '512' | '1024'].toFixed(0)}%`;
            const label = key === 'avg_score' ? 'avg score' : `≥${key}`;
            return <StatCell key={key} label={label} value={value} highlight={iWin} />;
          })}
        </div>
      ) : (
        <div className="text-xs text-zinc-300 dark:text-zinc-700">Loading stats…</div>
      )}
    </button>
  );
}

function ComparisonTab({
  checkpoints,
  selected,
  benchmark,
  onSelect,
}: Omit<CheckpointSelectorProps, 'replays' | 'loading' | 'onSelect'> & { onSelect: (name: string) => void }) {
  const bestDqn = checkpoints.find(cp => cp.name === 'best');
  const heuristic = checkpoints.find(cp => cp.name === 'heuristic');

  if (!bestDqn || !heuristic) return null;

  const dqnRow = bestDqn.benchmarkLabel ? (benchmark?.find(r => r.label === bestDqn.benchmarkLabel) ?? null) : null;
  const heurRow = heuristic.benchmarkLabel ? (benchmark?.find(r => r.label === heuristic.benchmarkLabel) ?? null) : null;

  const winnerOn: Record<string, 'left' | 'right' | 'tie'> = {};
  if (dqnRow && heurRow) {
    winnerOn['avg_score'] = dqnRow.avg_score > heurRow.avg_score ? 'left' : dqnRow.avg_score < heurRow.avg_score ? 'right' : 'tie';
    winnerOn['512'] = dqnRow.tile_pcts['512'] > heurRow.tile_pcts['512'] ? 'left' : dqnRow.tile_pcts['512'] < heurRow.tile_pcts['512'] ? 'right' : 'tie';
    winnerOn['1024'] = dqnRow.tile_pcts['1024'] > heurRow.tile_pcts['1024'] ? 'left' : dqnRow.tile_pcts['1024'] < heurRow.tile_pcts['1024'] ? 'right' : 'tie';
  }

  return (
    <div className="flex flex-col gap-4">
      <p className="text-xs text-zinc-500 leading-relaxed">
        Select a card to load that agent's replay below. Does training on millions of games beat carefully hand-crafted rules?
      </p>

      <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_1fr] gap-3 md:gap-4 items-center">
        <ComparisonCard
          cp={bestDqn}
          isSelected={selected === bestDqn.name}
          benchmarkRow={dqnRow}
          onSelect={() => onSelect(bestDqn.name)}
          side="left"
          winnerOn={winnerOn}
        />

        <div className="hidden md:flex flex-col items-center gap-1">
          <span className="text-lg font-bold text-zinc-400 dark:text-zinc-600">vs</span>
        </div>
        <div className="md:hidden flex items-center justify-center gap-3 text-zinc-300 dark:text-zinc-700 text-xs">
          <div className="flex-1 h-px bg-zinc-200 dark:bg-zinc-800" />
          <span>vs</span>
          <div className="flex-1 h-px bg-zinc-200 dark:bg-zinc-800" />
        </div>

        <ComparisonCard
          cp={heuristic}
          isSelected={selected === heuristic.name}
          benchmarkRow={heurRow}
          onSelect={() => onSelect(heuristic.name)}
          side="right"
          winnerOn={winnerOn}
        />
      </div>
    </div>
  );
}

// --- Main export ---

export default function CheckpointSelector({
  checkpoints,
  selected,
  replays,
  loading,
  benchmark,
  onSelect,
}: CheckpointSelectorProps) {
  const selectedCp = checkpoints.find(cp => cp.name === selected);
  const defaultTab: Tab = selectedCp?.group === 'comparison' ? 'comparison' : 'progression';
  const [activeTab, setActiveTab] = useState<Tab>(defaultTab);

  const handleSelect = (name: string) => {
    const cp = checkpoints.find(c => c.name === name);
    if (cp?.group === 'comparison') setActiveTab('comparison');
    onSelect(name);
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="flex gap-1 border-b border-zinc-200 dark:border-zinc-800">
        {(['progression', 'comparison'] as Tab[]).map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`
              px-4 py-2 text-xs font-medium transition-colors border-b-2 -mb-px
              ${activeTab === tab
                ? 'border-blue-500 text-blue-500 dark:text-blue-400'
                : 'border-transparent text-zinc-400 dark:text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300'}
            `}
          >
            {tab === 'progression' ? 'Training Progression' : 'Learned vs Classical'}
          </button>
        ))}
      </div>

      {activeTab === 'progression' ? (
        <ProgressionTab
          checkpoints={checkpoints}
          selected={selected}
          replays={replays}
          loading={loading}
          benchmark={benchmark}
          onSelect={handleSelect}
        />
      ) : (
        <ComparisonTab
          checkpoints={checkpoints}
          selected={selected}
          benchmark={benchmark}
          onSelect={handleSelect}
        />
      )}
    </div>
  );
}
