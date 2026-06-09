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

function MetaRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-xs text-zinc-600">{label}</span>
      <span className="text-sm text-zinc-300 tabular-nums">{value}</span>
    </div>
  );
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
  const isDrl = replay.agent_type === 'drl';
  const isHeuristic = replay.agent_type === 'heuristic';

  return (
    <div className="flex flex-col gap-4 text-zinc-200">
      {/* Hero stats */}
      <div className="grid grid-cols-2 gap-4">
        <div className="flex flex-col gap-0.5">
          <span className="text-xs text-zinc-500 uppercase tracking-widest">Score</span>
          <span className="text-3xl font-bold tabular-nums text-zinc-100">{frame.score.toLocaleString()}</span>
        </div>
        <div className="flex flex-col gap-0.5">
          <span className="text-xs text-zinc-500 uppercase tracking-widest">Best tile</span>
          <span className={`text-3xl font-bold tabular-nums ${maxTileOnBoard >= 512 ? 'text-blue-400' : 'text-zinc-100'}`}>
            {maxTileOnBoard}
          </span>
        </div>
      </div>

      <Divider />

      {/* Move + action */}
      <div className="grid grid-cols-2 gap-3">
        <MetaRow label="Move" value={`${frame.move_number} / ${replay.total_moves}`} />
        <MetaRow
          label="Action"
          value={frame.action ? (ARROWS[frame.action] ?? frame.action) : isDone ? 'Game over' : '—'}
        />
      </div>

      <Divider />

      {/* Agent details */}
      <div className="flex flex-col gap-3">
        <p className="text-xs text-zinc-500 uppercase tracking-widest">Agent</p>
        <div className="grid grid-cols-2 gap-3">
          {!isHeuristic && (
            <MetaRow label="Training steps" value={replay.training_step.toLocaleString()} />
          )}
          {isDrl && (
            <MetaRow label="Exploration (ε)" value={replay.epsilon.toFixed(3)} />
          )}
          {!isHeuristic && (
            <>
              <MetaRow label="Avg score" value={replay.average_score.toLocaleString()} />
              <MetaRow label="Avg best tile" value={Math.round(replay.average_max_tile)} />
            </>
          )}
          {replay.search_depth !== null && (
            <MetaRow label="Search depth" value={replay.search_depth} />
          )}
          {isHeuristic && (
            <MetaRow label="Evaluator" value="Monotonicity + corner" />
          )}
        </div>
      </div>
    </div>
  );
}
