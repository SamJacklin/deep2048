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
  agent_type: 'drl' | 'drl_search' | 'heuristic';
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
}

export const CHECKPOINTS: CheckpointMeta[] = [
  // Training progression — four DQN milestones
  { name: 'step_100083', label: 'Early Training',  description: '100K steps — basic merges, heavy exploration', group: 'progression' },
  { name: 'step_350068', label: 'Mid Training',    description: '350K steps — 256 tile emerging',              group: 'progression' },
  { name: 'step_625066', label: 'Late Training',   description: '625K steps — 512 reached regularly',          group: 'progression' },
  { name: 'best',        label: 'Best DQN',        description: '925K steps — ≥512: 61%  ≥1024: 9%',          group: 'progression' },
  // Approach comparison — same seed, different strategies
  { name: 'drl_search',  label: 'DQN + Search',    description: 'DQN + expectimax — shows limits of learned Q-values as a planner', group: 'comparison' },
  { name: 'heuristic',   label: 'Heuristic Search', description: 'Hand-crafted evaluator, expectimax (depth 2)', group: 'comparison' },
];
