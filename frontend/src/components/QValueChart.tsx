import type { QValues } from '../types';

interface QValueChartProps {
  qValues: QValues;
  chosenAction: string | null;
  validActions: string[];
  label?: string;
}

const ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT'] as const;

const ARROWS: Record<string, string> = {
  UP: '↑',
  DOWN: '↓',
  LEFT: '←',
  RIGHT: '→',
};

export default function QValueChart({ qValues, chosenAction, validActions, label = 'Q-Values' }: QValueChartProps) {
  const validSet = new Set(validActions);
  const values = ACTIONS.map(a => qValues[a]);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = Math.max(max - min, 1e-6);

  return (
    <div className="flex flex-col gap-1.5">
      <p className="text-xs text-zinc-400 dark:text-zinc-500 uppercase tracking-widest">{label}</p>
      <p className="text-xs text-zinc-400 dark:text-zinc-600 mb-3">How strongly the agent prefers each move</p>
      {ACTIONS.map(action => {
        const val = qValues[action];
        const isChosen = action === chosenAction;
        const isValid = validSet.has(action);
        const pct = (val - min) / range;

        return (
          <div key={action} className="flex items-center gap-2">
            <span
              className={`w-6 text-center text-sm font-bold shrink-0 ${
                isChosen ? 'text-blue-500 dark:text-blue-400' : isValid ? 'text-zinc-600 dark:text-zinc-300' : 'text-zinc-300 dark:text-zinc-700'
              }`}
            >
              {ARROWS[action]}
            </span>

            <div className="flex-1 h-5 rounded bg-zinc-200 dark:bg-zinc-800 overflow-hidden relative">
              <div
                className={`h-full rounded transition-all duration-150 ${
                  isChosen
                    ? 'bg-blue-500'
                    : isValid
                    ? 'bg-zinc-400 dark:bg-zinc-500'
                    : 'bg-zinc-200 dark:bg-zinc-800'
                }`}
                style={{ width: `${pct * 100}%` }}
              />
            </div>

            <span
              className={`text-xs tabular-nums w-14 text-right shrink-0 ${
                isChosen ? 'text-blue-500 dark:text-blue-400' : isValid ? 'text-zinc-500 dark:text-zinc-400' : 'text-zinc-300 dark:text-zinc-700'
              }`}
            >
              {val.toFixed(3)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
