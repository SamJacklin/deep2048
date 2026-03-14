import type { CheckpointMeta, Replay } from '../types';

interface CheckpointSelectorProps {
  checkpoints: CheckpointMeta[];
  selected: string;
  replays: Record<string, Replay | null>;
  loading: Record<string, boolean>;
  onSelect: (name: string) => void;
}

export default function CheckpointSelector({
  checkpoints,
  selected,
  replays,
  loading,
  onSelect,
}: CheckpointSelectorProps) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {checkpoints.map(cp => {
        const replay = replays[cp.name];
        const isLoading = loading[cp.name];
        const isSelected = selected === cp.name;

        return (
          <button
            key={cp.name}
            onClick={() => onSelect(cp.name)}
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
                <div className="text-xs text-zinc-600">
                  Avg score{' '}
                  <span className={`tabular-nums ${isSelected ? 'text-zinc-300' : 'text-zinc-400'}`}>
                    {replay.average_score.toLocaleString()}
                  </span>
                </div>
                <div className="text-xs text-zinc-600">
                  Avg tile{' '}
                  <span className={`tabular-nums ${isSelected ? 'text-zinc-300' : 'text-zinc-400'}`}>
                    {replay.average_max_tile}
                  </span>
                </div>
                <div className="text-xs text-zinc-600">
                  ε{' '}
                  <span className={`tabular-nums ${isSelected ? 'text-zinc-300' : 'text-zinc-400'}`}>
                    {replay.epsilon.toFixed(3)}
                  </span>
                </div>
              </div>
            ) : (
              <div className="text-xs text-zinc-700">No data</div>
            )}
          </button>
        );
      })}
    </div>
  );
}
