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
  training_step: number;
  epsilon: number;
  average_score: number;
  average_max_tile: number;
  final_score: number;
  max_tile_achieved: number;
  total_moves: number;
  frames: ReplayFrame[];
}

export interface CheckpointMeta {
  name: string;
  label: string;
  description: string;
}

export const CHECKPOINTS: CheckpointMeta[] = [
  { name: 'step_25143',  label: 'Early Training', description: '25K steps — mostly exploring' },
  { name: 'step_75041',  label: 'Mid Training',   description: '75K steps — patterns forming' },
  { name: 'step_125198', label: 'Late Training',  description: '125K steps — strong strategy' },
  { name: 'best',        label: 'Best Agent',     description: '150K steps — peak performance' },
];
