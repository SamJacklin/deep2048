export interface QValues {
  UP: number;
  DOWN: number;
  LEFT: number;
  RIGHT: number;
}

export interface ReplayFrame {
  board: number[][];
  score: number;
  action: string | null;
  q_values: QValues;
  valid_actions: string[];
  reward: number;
  move_number: number;
}

export interface Replay {
  checkpoint_name: string;
  checkpoint_label: string;
  checkpoint_description: string;
  agent_type: 'drl' | 'heuristic';
  search_depth: number | null;
  score_label: string;
  training_step: number;
  epsilon: number;
  average_score: number;
  average_max_tile: number;
  final_score: number;
  max_tile_achieved: number;
  total_moves: number;
  frames: ReplayFrame[];
}

export interface BenchmarkRow {
  label: string;
  n_games: number;
  avg_score: number;
  avg_max_tile: number;
  tile_pcts: { '256': number; '512': number; '1024': number; '2048': number };
}

export interface CheckpointMeta {
  name: string;
  label: string;
  description: string;
  group: 'progression' | 'comparison';
  benchmarkLabel?: string;
}

export const CHECKPOINTS: CheckpointMeta[] = [
  // Training progression — four DQN milestones
  { name: 'step_100083', label: 'Early Training',  description: 'Mostly random. Merging whatever it can reach with no real plan yet', group: 'progression' },
  { name: 'step_350068', label: 'Mid Training',    description: 'The 256 tile starts appearing consistently. Something is clicking',  group: 'progression' },
  { name: 'step_625066', label: 'Late Training',   description: 'Corners are being held and tiles are being chained',                 group: 'progression' },
  { name: 'best',        label: 'Best DQN',        description: 'The finished model. 512 in 6 out of 10 games, 1024 in nearly 1 in 10', group: 'progression', benchmarkLabel: 'DQN (best)' },
  // Approach comparison — same seed, different strategies
{ name: 'heuristic',   label: 'Heuristic Search', description: 'Hand-written placement rules plus exhaustive search. A useful baseline for what the learned model is actually worth', group: 'comparison', benchmarkLabel: 'Heuristic depth=2' },
];
