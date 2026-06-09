import { useRef } from 'react';
import { useTheme } from '../toolkit/frontend/portfolio/components/ThemeContext';

interface BoardProps {
  board: number[][];
  prevBoard?: number[][] | null;
}

// Tile value → [bg, text, font-size class]
function tileStyle(value: number, isDark: boolean): { bg: string; text: string; fs: string } {
  if (value === 0) return { bg: isDark ? '#27272a' : '#c4c4c8', text: isDark ? '#3f3f46' : '#a1a1aa', fs: '' };
  if (value === 2)    return { bg: isDark ? '#e4e4e7' : '#fafafa', text: '#18181b', fs: 'text-2xl' };
  if (value === 4)    return { bg: isDark ? '#d4d4d8' : '#f4f4f5', text: '#18181b', fs: 'text-2xl' };
  if (value === 8)    return { bg: '#fde68a', text: '#1c1917', fs: 'text-2xl' };
  if (value === 16)   return { bg: '#fcd34d', text: '#1c1917', fs: 'text-2xl' };
  if (value === 32)   return { bg: '#fdba74', text: '#1c1917', fs: 'text-2xl' };
  if (value === 64)   return { bg: '#fb923c', text: '#fff7ed', fs: 'text-2xl' };
  if (value === 128)  return { bg: '#fca5a5', text: '#1c1917', fs: 'text-xl' };
  if (value === 256)  return { bg: '#f87171', text: '#fff1f2', fs: 'text-xl' };
  if (value === 512)  return { bg: '#93c5fd', text: '#1e3a5f', fs: 'text-xl' };
  if (value === 1024) return { bg: '#60a5fa', text: '#fff',    fs: 'text-lg' };
  if (value === 2048) return { bg: '#3b82f6', text: '#fff',    fs: 'text-lg' };
  return               { bg: '#8b5cf6', text: '#fff',    fs: 'text-base' };
}

function Tile({
  value,
  popped,
  isDark,
}: {
  value: number;
  popped: boolean;
  isDark: boolean;
}) {
  const style = tileStyle(value, isDark);
  return (
    <div
      className={`flex items-center justify-center rounded font-bold select-none w-full h-full transition-colors duration-150 ${popped ? 'tile-pop' : ''}`}
      style={{ backgroundColor: style.bg, color: style.text }}
    >
      <span className={style.fs}>
        {value !== 0 ? value.toLocaleString() : ''}
      </span>
    </div>
  );
}

export default function Board({ board, prevBoard }: BoardProps) {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const popKey = useRef<Record<string, number>>({});

  const cells = board.flatMap((row, r) =>
    row.map((val, c) => {
      const prev = prevBoard?.[r]?.[c] ?? null;
      const popped = prev !== null && val !== prev && val !== 0;
      return { r, c, val, popped };
    }),
  );

  // Increment popKey when a cell pops so React re-mounts the tile
  cells.forEach(({ r, c, popped }) => {
    const key = `${r}-${c}`;
    if (popped) popKey.current[key] = (popKey.current[key] ?? 0) + 1;
  });

  return (
    <div
      className="grid gap-2 p-3 rounded-xl"
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
        gridTemplateRows: 'repeat(4, 1fr)',
        gap: '8px',
        backgroundColor: isDark ? '#18181b' : '#e4e4e7',
        aspectRatio: '1 / 1',
        width: '100%',
        maxWidth: '360px',
      }}
    >
      {cells.map(({ r, c, val, popped }) => (
        <Tile
          key={`${r}-${c}-${popKey.current[`${r}-${c}`] ?? 0}`}
          value={val}
          popped={popped}
          isDark={isDark}
        />
      ))}
    </div>
  );
}
