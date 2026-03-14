import type { ReplayPlayer } from '../hooks/useReplayPlayer';

interface ControlsProps {
  player: ReplayPlayer;
}

function IconButton({
  onClick,
  title,
  children,
  disabled,
}: {
  onClick: () => void;
  title: string;
  children: React.ReactNode;
  disabled?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      disabled={disabled}
      className="p-2 rounded text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
    >
      {children}
    </button>
  );
}

export default function Controls({ player }: ControlsProps) {
  const { frameIndex, playing, speedIndex, speeds, totalFrames, togglePlay, stepBack, stepForward, seek, cycleSpeed, reset } = player;
  const progress = totalFrames > 1 ? frameIndex / (totalFrames - 1) : 0;
  const currentSpeed = speeds[speedIndex];

  return (
    <div className="flex flex-col gap-3">
      {/* Scrubber */}
      <div className="flex items-center gap-3">
        <span className="text-xs text-zinc-600 tabular-nums w-8">{frameIndex}</span>
        <input
          type="range"
          min={0}
          max={Math.max(totalFrames - 1, 1)}
          value={frameIndex}
          onChange={e => seek(Number(e.target.value))}
          className="flex-1 accent-blue-500 cursor-pointer"
        />
        <span className="text-xs text-zinc-600 tabular-nums w-8 text-right">{totalFrames - 1}</span>
      </div>

      {/* Buttons */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1">
          {/* Reset */}
          <IconButton onClick={reset} title="Reset">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M2.5 8a5.5 5.5 0 1 1 1.1 3.3" strokeLinecap="round"/>
              <path d="M2.5 4.5v4h4" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </IconButton>

          {/* Step back */}
          <IconButton onClick={stepBack} title="Step back" disabled={frameIndex === 0}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M10 3L4 8l6 5V3z" fill="currentColor" stroke="none"/>
              <path d="M4 3v10" strokeLinecap="round"/>
            </svg>
          </IconButton>

          {/* Play / Pause */}
          <button
            onClick={togglePlay}
            className="p-2 rounded bg-blue-600 hover:bg-blue-500 text-white transition-colors"
            title={playing ? 'Pause' : 'Play'}
          >
            {playing ? (
              <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                <rect x="3" y="3" width="4" height="10" rx="1"/>
                <rect x="9" y="3" width="4" height="10" rx="1"/>
              </svg>
            ) : (
              <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                <path d="M4 2.5l10 5.5-10 5.5V2.5z"/>
              </svg>
            )}
          </button>

          {/* Step forward */}
          <IconButton onClick={stepForward} title="Step forward" disabled={frameIndex === totalFrames - 1}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M6 3l6 5-6 5V3z" fill="currentColor" stroke="none"/>
              <path d="M12 3v10" strokeLinecap="round"/>
            </svg>
          </IconButton>
        </div>

        {/* Speed */}
        <button
          onClick={cycleSpeed}
          className="text-xs text-zinc-400 hover:text-zinc-100 px-2 py-1 rounded hover:bg-zinc-800 transition-colors tabular-nums"
          title="Cycle playback speed"
        >
          {currentSpeed}×
        </button>
      </div>

      {/* Progress bar */}
      <div className="h-0.5 bg-zinc-800 rounded-full overflow-hidden">
        <div
          className="h-full bg-blue-600 transition-all duration-100"
          style={{ width: `${progress * 100}%` }}
        />
      </div>
    </div>
  );
}
