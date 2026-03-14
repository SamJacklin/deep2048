import type { Replay, ReplayFrame } from '../types';

interface StatsPanelProps {
  replay: Replay;
  frame: ReplayFrame;
}

function Stat({ label, value, highlight }: { label: string; value: string | number; highlight?: boolean }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-xs text-zinc-500 uppercase tracking-widest">{label}</span>
      <span className={`text-lg font-bold tabular-nums ${highlight ? 'text-blue-400' : 'text-zinc-100'}`}>
        {value}
      </span>
    </div>
  );
}

function Divider() {
  return <div className="border-t border-zinc-800 my-1" />;
}

const ARROWS: Record<string, string> = {
  UP: '↑ Up',
  DOWN: '↓ Down',
  LEFT: '← Left',
  RIGHT: '→ Right',
};

export default function StatsPanel({ replay, frame }: StatsPanelProps) {
  const maxTileOnBoard = Math.max(...frame.board.flat());
  const isDone = frame.valid_actions.length === 0;

  return (
    <div className="flex flex-col gap-4 text-zinc-200">
      {/* Live game stats */}
      <div className="grid grid-cols-2 gap-4">
        <Stat label="Score" value={frame.score.toLocaleString()} />
        <Stat label="Max Tile" value={maxTileOnBoard} highlight={maxTileOnBoard >= 512} />
        <Stat label="Move" value={`${frame.move_number} / ${replay.total_moves}`} />
        <Stat
          label="Action"
          value={frame.action ? (ARROWS[frame.action] ?? frame.action) : isDone ? 'Game over' : '—'}
          highlight={!!frame.action}
        />
      </div>

      {frame.reward > 0 && (
        <div className="text-xs text-zinc-400">
          Merged for <span className="text-blue-400 font-bold">+{frame.reward}</span>
        </div>
      )}

      <Divider />

      {/* Checkpoint info */}
      <div className="flex flex-col gap-3">
        <p className="text-xs text-zinc-500 uppercase tracking-widest">Checkpoint</p>
        <div className="grid grid-cols-2 gap-3">
          <div className="flex flex-col gap-0.5">
            <span className="text-xs text-zinc-600">Training steps</span>
            <span className="text-sm text-zinc-300 tabular-nums">{replay.training_step.toLocaleString()}</span>
          </div>
          <div className="flex flex-col gap-0.5">
            <span className="text-xs text-zinc-600">Epsilon (ε)</span>
            <span className="text-sm text-zinc-300 tabular-nums">{replay.epsilon.toFixed(3)}</span>
          </div>
          <div className="flex flex-col gap-0.5">
            <span className="text-xs text-zinc-600">Avg score</span>
            <span className="text-sm text-zinc-300 tabular-nums">{replay.average_score.toLocaleString()}</span>
          </div>
          <div className="flex flex-col gap-0.5">
            <span className="text-xs text-zinc-600">Avg max tile</span>
            <span className="text-sm text-zinc-300 tabular-nums">{replay.average_max_tile}</span>
          </div>
        </div>
      </div>

      <Divider />

      {/* Game result */}
      <div className="flex flex-col gap-3">
        <p className="text-xs text-zinc-500 uppercase tracking-widest">This game</p>
        <div className="grid grid-cols-2 gap-3">
          <div className="flex flex-col gap-0.5">
            <span className="text-xs text-zinc-600">Final score</span>
            <span className="text-sm text-zinc-300 tabular-nums">{replay.final_score.toLocaleString()}</span>
          </div>
          <div className="flex flex-col gap-0.5">
            <span className="text-xs text-zinc-600">Best tile</span>
            <span className="text-sm text-zinc-300 tabular-nums">{replay.max_tile_achieved}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
