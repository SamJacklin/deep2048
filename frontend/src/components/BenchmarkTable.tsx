import type { BenchmarkRow } from '../types';

interface BenchmarkTableProps {
  rows: BenchmarkRow[];
}

const TILES = ['256', '512', '1024', '2048'] as const;

export default function BenchmarkTable({ rows }: BenchmarkTableProps) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs font-mono border-collapse">
        <thead>
          <tr className="border-b border-zinc-800">
            <th className="text-left text-zinc-500 uppercase tracking-widest py-2 pr-4 font-normal">Agent</th>
            <th className="text-right text-zinc-500 uppercase tracking-widest py-2 px-3 font-normal">Avg score</th>
            <th className="text-right text-zinc-500 uppercase tracking-widest py-2 px-3 font-normal">Avg tile</th>
            {TILES.map(t => (
              <th key={t} className="text-right text-zinc-500 uppercase tracking-widest py-2 px-3 font-normal">
                ≥{t}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-b border-zinc-800/50 last:border-0">
              <td className="py-2 pr-4 text-zinc-300">{row.label}</td>
              <td className="py-2 px-3 text-right tabular-nums text-zinc-200">
                {row.avg_score.toLocaleString()}
              </td>
              <td className="py-2 px-3 text-right tabular-nums text-zinc-200">
                {row.avg_max_tile.toLocaleString()}
              </td>
              {TILES.map(t => {
                const pct = row.tile_pcts[t];
                const highlight = t === '512' && pct >= 50;
                return (
                  <td
                    key={t}
                    className={`py-2 px-3 text-right tabular-nums ${
                      highlight ? 'text-blue-400 font-bold' : 'text-zinc-400'
                    }`}
                  >
                    {pct.toFixed(1)}%
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-xs text-zinc-600">
        Based on {rows[0]?.n_games ?? 0} games per agent. Generate with{' '}
        <code className="text-zinc-500">make benchmark</code>.
      </p>
    </div>
  );
}
