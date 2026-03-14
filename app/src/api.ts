import { invoke } from "@tauri-apps/api/core";
import type {
  Replay,
  FrameDataPoint,
  DamageEvent,
  RoundResult,
  MatchStats,
  Highlight,
} from "./types";

export async function getReplays(dbPath: string): Promise<Replay[]> {
  return invoke("get_replays", { dbPath });
}

export async function getFrameData(
  dbPath: string,
  replayId: string
): Promise<FrameDataPoint[]> {
  return invoke("get_frame_data", { dbPath, replayId });
}

export async function getDamageEvents(
  dbPath: string,
  replayId: string
): Promise<DamageEvent[]> {
  return invoke("get_damage_events", { dbPath, replayId });
}

export async function getRounds(
  dbPath: string,
  replayId: string
): Promise<RoundResult[]> {
  return invoke("get_rounds", { dbPath, replayId });
}

export async function getMatchStats(
  dbPath: string,
  replayId: string
): Promise<MatchStats> {
  return invoke("get_match_stats", { dbPath, replayId });
}

export async function getHighlights(
  dbPath: string,
  replayId: string
): Promise<Highlight[]> {
  return invoke("get_highlights", { dbPath, replayId });
}

export async function analyzeVideo(
  videoPath: string,
  outputDir: string,
  sampleEvery?: number
): Promise<string> {
  return invoke("analyze_video", { videoPath, outputDir, sampleEvery });
}

export async function exportClip(
  videoPath: string,
  startMs: number,
  endMs: number,
  outputPath: string
): Promise<void> {
  return invoke("export_clip", { videoPath, startMs, endMs, outputPath });
}

export async function resolveVideoPath(
  dbPath: string,
  replayId: string
): Promise<string> {
  return invoke("resolve_video_path", { dbPath, replayId });
}
