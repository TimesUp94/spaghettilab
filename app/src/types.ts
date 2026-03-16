export interface Replay {
  replay_id: string;
  video_path: string;
  duration_ms: number;
  frame_count: number;
}

export interface FrameDataPoint {
  timestamp_ms: number;
  p1_health_pct: number | null;
  p2_health_pct: number | null;
}

export interface DamageEvent {
  event_id: number;
  replay_id: string;
  timestamp_ms: number;
  frame_start: number;
  frame_end: number;
  target_side: number;
  damage_pct: number;
  pre_health_pct: number;
  post_health_pct: number;
}

export interface RoundResult {
  replay_id: string;
  round_index: number;
  round_start_ms: number;
  round_end_ms: number;
  winner: string;
  winner_final_hp: number;
  loser_final_hp: number;
  winner_min_hp: number;
  max_deficit: number;
  deficit_timestamp_ms: number;
  is_comeback: boolean;
}

export interface MatchStats {
  replay_id: string;
  total_rounds: number;
  p1_round_wins: number;
  p2_round_wins: number;
  total_damage_events: number;
  p1_damage_taken: number;
  p2_damage_taken: number;
  p1_biggest_hit: number;
  p2_biggest_hit: number;
  avg_round_duration_s: number;
  longest_round_s: number;
  shortest_round_s: number;
  comeback_count: number;
  close_rounds: number;
  avg_winner_final_hp: number;
  duration_s: number;
}

export interface Highlight {
  kind: string;
  label: string;
  timestamp_ms: number;
  end_ms: number;
  details: string;
  severity: number;
}

export interface Match {
  match_index: number;
  rounds: RoundResult[];
  winner: string;
  p1_rounds_won: number;
  p2_rounds_won: number;
  start_ms: number;
  end_ms: number;
}

export interface Note {
  note_id: number;
  replay_id: string;
  timestamp_ms: number;
  text: string;
  created_at: string;
}

export interface SpagSession {
  db_path: string;
  video_path: string;
  spag_path: string;
  replay_id: string;
}

export type ActiveTab = "matches" | "rounds" | "highlights" | "damage" | "notes";

// VOD Splitter types

export interface RoiRect {
  y1: number;
  y2: number;
  x1: number;
  x2: number;
}

export interface VodRoiConfig {
  p1_tension: RoiRect;
  p2_tension: RoiRect;
  timer: RoiRect;
  banner: RoiRect;
}

export interface DetectedSetInfo {
  index: number;
  start_secs: number;
  end_secs: number;
  gameplay_duration_secs: number;
  game_count: number;
}

export interface CutSetRequest {
  index: number;
  start_secs: number;
  end_secs: number;
}
