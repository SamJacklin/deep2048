import type { BenchmarkRow, CheckpointMeta, Replay } from '../types';

interface CheckpointSelectorProps {
  checkpoints: CheckpointMeta[];
  selected: string;
  replays: Record<string, Replay | null>;
  loading: Record<string, boolean>;
  benchmark: BenchmarkRow[] | null;
  onSelect: (name: string) => void;
}

const GROUP_LABELS: Record<string, string> = {
  progression: 'Training Progression',
};

function CheckpointCard({
  cp,
  isSelected,
  replay,
  isLoading,
  benchmarkRow,
  onSelect,
}: {
  cp: CheckpointMeta;
  isSelected: boolean;
  replay: Replay | null | undefined;
  isLoading: boolean;
  benchmarkRow: BenchmarkRow | null;
  onSelect: () => void;
}) {
  return (
    <button
      onClick={onSelect}
      className={`
        text-left p-4 rounded-lg border transition-all
        ${isSelected
          ? 'border-blue-500 bg-blue-950/40 text-zinc-100'
          : 'border-zinc-800 bg-zinc-900/60 text-zinc-400 hover:border-zinc-600 hover:text-zinc-200'}
      `}
    >
      <div className={`text-sm font-bold mb-1 ${isSelected ? 'text-blue-400' : ''}`}>
        {cp.label}
      </div>
      <div className="text-xs text-zinc-500 mb-3 leading-snug">{cp.description}</div>

      {isLoading ? (
        <div className="text-xs text-zinc-600">Loading…</div>
      ) : benchmarkRow ? (
        <div className="flex flex-col gap-1.5">
          <div className="text-xs text-zinc-600">
            Avg score{' '}
            <span className={`tabular-nums font-medium ${isSelected ? 'text-zinc-200' : 'text-zinc-400'}`}>
              {benchmarkRow.avg_score.toLocaleString()}
            </span>
          </div>
          <div className="flex gap-3">
            {(['512', '1024', '2048'] as const).map(t => (
              <div key={t} className="flex flex-col gap-0.5">
                <span className="text-zinc-600" style={{ fontSize: '10px' }}>≥{t}</span>
                <span className={`tabular-nums font-medium ${isSelected ? 'text-zinc-200' : 'text-zinc-400'}`} style={{ fontSize: '11px' }}>
                  {benchmarkRow.tile_pcts[t].toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      ) : replay ? (
        <div className="flex flex-col gap-1">
          <div className="text-xs text-zinc-600">
            Avg score{' '}
            <span className={`tabular-nums ${isSelected ? 'text-zinc-300' : 'text-zinc-400'}`}>
              {replay.average_score.toLocaleString()}
            </span>
          </div>
          <div className="text-xs text-zinc-600">
            Step{' '}
            <span className={`tabular-nums ${isSelected ? 'text-zinc-300' : 'text-zinc-400'}`}>
              {replay.training_step.toLocaleString()}
            </span>
          </div>
          {replay.agent_type === 'drl' && (
            <div className="text-xs text-zinc-600">
              ε{' '}
              <span className={`tabular-nums ${isSelected ? 'text-zinc-300' : 'text-zinc-400'}`}>
                {replay.epsilon.toFixed(3)}
              </span>
            </div>
          )}
        </div>
      ) : (
        <div className="text-xs text-zinc-700">No data</div>
      )}
    </button>
  );
}

export default function CheckpointSelector({
  checkpoints,
  selected,
  replays,
  loading,
  benchmark,
  onSelect,
}: CheckpointSelectorProps) {
  const progression = checkpoints.filter(cp => cp.group === 'progression');

  const renderCard = (cp: CheckpointMeta) => {
    const benchmarkRow = cp.benchmarkLabel
      ? (benchmark?.find(r => r.label === cp.benchmarkLabel) ?? null)
      : null;
    return (
      <CheckpointCard
        key={cp.name}
        cp={cp}
        isSelected={selected === cp.name}
        replay={replays[cp.name]}
        isLoading={loading[cp.name] ?? false}
        benchmarkRow={benchmarkRow}
        onSelect={() => onSelect(cp.name)}
      />
    );
  };

  const bestDqn = checkpoints.find(cp => cp.name === 'best');
  const heuristic = checkpoints.find(cp => cp.name === 'heuristic');

  return (
    <div className="flex flex-col gap-8">
      {/* Training progression */}
      <div className="flex flex-col gap-3">
        <p className="text-xs text-zinc-500 uppercase tracking-widest">
          {GROUP_LABELS['progression']}
        </p>
        <div className="grid gap-3 grid-cols-2 md:grid-cols-4">
          {progression.map(renderCard)}
        </div>
      </div>

      {/* Head-to-head comparison */}
      {bestDqn && heuristic && (
        <div className="flex flex-col gap-3">
          <div className="flex items-baseline gap-3">
            <p className="text-xs text-zinc-500 uppercase tracking-widest">Learned vs Classical</p>
            <p className="text-xs text-zinc-600">Does training beat hand-crafted rules?</p>
          </div>
          <div className="grid gap-3 grid-cols-1 md:grid-cols-2">
            {[bestDqn, heuristic].map(cp => {
              const benchmarkRow = cp.benchmarkLabel
                ? (benchmark?.find(r => r.label === cp.benchmarkLabel) ?? null)
                : null;
              const isSelected = selected === cp.name;
              return (
                <button
                  key={cp.name}
                  onClick={() => onSelect(cp.name)}
                  className={`text-left p-5 rounded-lg border transition-all ${
                    isSelected
                      ? 'border-blue-500 bg-blue-950/40'
                      : 'border-zinc-800 bg-zinc-900/60 hover:border-zinc-600'
                  }`}
                >
                  <div className={`text-sm font-bold mb-1 ${isSelected ? 'text-blue-400' : 'text-zinc-300'}`}>
                    {cp.label}
                  </div>
                  <div className="text-xs text-zinc-500 mb-4 leading-snug">{cp.description}</div>

                  {benchmarkRow && (
                    <div className="flex gap-6">
                      <div>
                        <div className="text-xs text-zinc-600 mb-0.5">Avg score</div>
                        <div className={`text-2xl font-bold tabular-nums ${isSelected ? 'text-zinc-100' : 'text-zinc-300'}`}>
                          {benchmarkRow.avg_score.toLocaleString()}
                        </div>
                      </div>
                      {(['512', '1024'] as const).map(t => (
                        <div key={t}>
                          <div className="text-xs text-zinc-600 mb-0.5">≥{t}</div>
                          <div className={`text-2xl font-bold tabular-nums ${isSelected ? 'text-zinc-100' : 'text-zinc-300'}`}>
                            {benchmarkRow.tile_pcts[t].toFixed(0)}%
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
