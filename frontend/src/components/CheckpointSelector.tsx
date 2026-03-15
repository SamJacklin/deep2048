import type { CheckpointMeta, Replay } from '../types';

interface CheckpointSelectorProps {
  checkpoints: CheckpointMeta[];
  selected: string;
  replays: Record<string, Replay | null>;
  loading: Record<string, boolean>;
  onSelect: (name: string) => void;
}

const GROUP_LABELS: Record<string, string> = {
  progression: 'Training Progression',
  comparison: 'Approach Comparison',
};

function CheckpointCard({
  cp,
  isSelected,
  replay,
  isLoading,
  onSelect,
}: {
  cp: CheckpointMeta;
  isSelected: boolean;
  replay: Replay | null | undefined;
  isLoading: boolean;
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
      ) : replay ? (
        <div className="flex flex-col gap-1">
          {replay.agent_type === 'heuristic' ? (
            <div className="text-xs text-zinc-600">
              Search depth{' '}
              <span className={`tabular-nums ${isSelected ? 'text-zinc-300' : 'text-zinc-400'}`}>
                {replay.search_depth}
              </span>
            </div>
          ) : (
            <>
              <div className="text-xs text-zinc-600">
                Avg score{' '}
                <span className={`tabular-nums ${isSelected ? 'text-zinc-300' : 'text-zinc-400'}`}>
                  {replay.average_score.toLocaleString()}
                </span>
              </div>
              <div className="text-xs text-zinc-600">
                Avg tile{' '}
                <span className={`tabular-nums ${isSelected ? 'text-zinc-300' : 'text-zinc-400'}`}>
                  {Math.round(replay.average_max_tile)}
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
              {replay.agent_type === 'drl_search' && (
                <div className="text-xs text-zinc-600">
                  Search depth{' '}
                  <span className={`tabular-nums ${isSelected ? 'text-zinc-300' : 'text-zinc-400'}`}>
                    {replay.search_depth}
                  </span>
                </div>
              )}
            </>
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
  onSelect,
}: CheckpointSelectorProps) {
  const groups = ['progression', 'comparison'] as const;

  return (
    <div className="flex flex-col gap-6">
      {groups.map(group => {
        const groupCheckpoints = checkpoints.filter(cp => cp.group === group);
        if (groupCheckpoints.length === 0) return null;

        return (
          <div key={group} className="flex flex-col gap-3">
            <p className="text-xs text-zinc-500 uppercase tracking-widest">
              {GROUP_LABELS[group]}
            </p>
            <div className={`grid gap-3 ${
              group === 'progression'
                ? 'grid-cols-2 md:grid-cols-4'
                : 'grid-cols-2 md:grid-cols-3'
            }`}>
              {groupCheckpoints.map(cp => (
                <CheckpointCard
                  key={cp.name}
                  cp={cp}
                  isSelected={selected === cp.name}
                  replay={replays[cp.name]}
                  isLoading={loading[cp.name] ?? false}
                  onSelect={() => onSelect(cp.name)}
                />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
