import type { BenchmarkRow } from '../types';

interface BenchmarkTableProps {
  rows: BenchmarkRow[];
  highlighted?: string;
}

const TILES = ['512', '1024', '2048', '4096', '8192', '16384'] as const;

function PctBar({ pct, active }: { pct: number; active: boolean }) {
  return (
    <div className="flex items-center gap-2">
      <div className="w-12 h-1.5 rounded-full bg-zinc-200 dark:bg-zinc-800 flex-shrink-0">
        <div
          className={`h-full rounded-full transition-all ${active ? 'bg-blue-400' : 'bg-zinc-400 dark:bg-zinc-600'}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className={`tabular-nums ${active ? 'text-blue-500 dark:text-blue-300' : 'text-zinc-500 dark:text-zinc-400'}`}>
        {pct.toFixed(0)}%
      </span>
    </div>
  );
}

export default function BenchmarkTable({ rows, highlighted }: BenchmarkTableProps) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs font-mono border-collapse">
        <thead>
          <tr className="border-b border-zinc-200 dark:border-zinc-800">
            <th className="text-left text-zinc-400 dark:text-zinc-500 uppercase tracking-widest py-2 pr-4 font-normal w-48">Agent</th>
            <th className="text-right text-zinc-400 dark:text-zinc-500 uppercase tracking-widest py-2 px-3 font-normal">Avg score</th>
            <th className="text-right text-zinc-400 dark:text-zinc-500 uppercase tracking-widest py-2 px-3 font-normal">Avg tile</th>
            {TILES.map(t => (
              <th key={t} className="text-left text-zinc-400 dark:text-zinc-500 uppercase tracking-widest py-2 px-3 font-normal">
                ≥{t}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => {
            const active = row.label === highlighted;
            return (
              <tr
                key={i}
                className={`border-b border-zinc-200/50 dark:border-zinc-800/50 last:border-0 transition-colors ${
                  active ? 'bg-blue-50 dark:bg-blue-950/20' : ''
                }`}
              >
                <td className={`py-3 pr-4 font-medium ${active ? 'text-blue-500 dark:text-blue-300' : 'text-zinc-600 dark:text-zinc-300'}`}>
                  {active && <span className="text-blue-500 mr-1.5">▶</span>}
                  {row.label}
                </td>
                <td className={`py-3 px-3 text-right tabular-nums ${active ? 'text-zinc-900 dark:text-zinc-100' : 'text-zinc-600 dark:text-zinc-300'}`}>
                  {row.avg_score.toLocaleString()}
                </td>
                <td className={`py-3 px-3 text-right tabular-nums ${active ? 'text-zinc-900 dark:text-zinc-100' : 'text-zinc-600 dark:text-zinc-300'}`}>
                  {row.avg_max_tile.toLocaleString()}
                </td>
                {TILES.map(t => (
                  <td key={t} className="py-3 px-3">
                    <PctBar pct={row.tile_pcts[t]} active={active} />
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="mt-3 text-xs text-zinc-400 dark:text-zinc-600">
        {rows[0]?.n_games ?? 0} games per agent · <code className="text-zinc-500">make benchmark</code> to regenerate
      </p>
    </div>
  );
}
