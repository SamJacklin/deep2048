const MODEL_CARDS = [
  {
    title: 'Multi-path CNN',
    subtitle: 'Built for how 2048 actually works',
    body: 'Three parallel convolution streams: local 3×3 filters for tile interactions, full-row kernels for LEFT/RIGHT merges, full-column kernels for UP/DOWN. The board has strong axis-aligned structure, so the network is designed to match it directly rather than hoping a generic CNN figures it out.',
  },
  {
    title: 'Dueling Double DQN',
    subtitle: 'Separates position value from move choice',
    body: 'Two output streams — one scores how good the current board is overall, one scores each individual move. Combining them reduces overconfident Q-value estimates, which vanilla DQN is prone to in long games where feedback is delayed.',
  },
  {
    title: 'PER + n-step returns',
    subtitle: 'Learns from what matters, not just what happened',
    body: 'A binary sum tree (O(log n) sampling) prioritises surprising transitions for replay — the moments where reality differed most from expectation. n-step returns (n=5) propagate reward five moves back, so the network learns to think ahead rather than optimise each tile merge in isolation.',
  },
];

const STACK_CARDS = [
  { title: 'Backend', detail: 'Python + PyTorch for model, training loop, and PER buffer. Typed with mypy, linted with ruff, managed with uv.' },
  { title: 'Frontend', detail: 'React + TypeScript with Vite and Tailwind. Static build deployed to S3 via GitHub Actions.' },
];

export default function ArchitectureSection() {
  return (
    <section className="flex flex-col gap-8">
      <h2 className="text-xs text-zinc-400 dark:text-zinc-500 uppercase tracking-widest">How it works</h2>

      {/* Model cards */}
      <div className="flex flex-col gap-3">
        <p className="text-xs text-zinc-400 dark:text-zinc-600 uppercase tracking-widest">The Model</p>
        <div className="grid gap-4 grid-cols-1 md:grid-cols-3">
          {MODEL_CARDS.map(card => (
            <div key={card.title} className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white/60 dark:bg-zinc-900/60 p-5 flex flex-col gap-2">
              <div className="text-sm font-bold text-zinc-800 dark:text-zinc-200">{card.title}</div>
              <div className="text-xs text-blue-500 dark:text-blue-400">{card.subtitle}</div>
              <p className="text-xs text-zinc-500 leading-relaxed mt-1">{card.body}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Stack cards */}
      <div className="flex flex-col gap-3">
        <p className="text-xs text-zinc-400 dark:text-zinc-600 uppercase tracking-widest">The Stack</p>
        <div className="grid gap-3 grid-cols-1 md:grid-cols-2">
          {STACK_CARDS.map(card => (
            <div key={card.title} className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white/60 dark:bg-zinc-900/60 p-4 flex flex-col gap-1.5">
              <div className="text-sm font-bold text-zinc-700 dark:text-zinc-300">{card.title}</div>
              <p className="text-xs text-zinc-500 leading-relaxed">{card.detail}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
