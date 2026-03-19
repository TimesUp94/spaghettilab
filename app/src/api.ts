import { invoke } from "@tauri-apps/api/core";
import type {
  Replay,
  FrameDataPoint,
  DamageEvent,
  RoundResult,
  MatchStats,
  Highlight,
  VodRoiConfig,
  DetectedSetInfo,
  CutSetRequest,
  Note,
  Drawing,
  SpagSession,
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
  sampleEvery?: number
): Promise<string> {
  return invoke("analyze_video", { videoPath, sampleEvery });
}

export async function reanalyzeReplay(dbPath: string, replayId: string): Promise<void> {
  return invoke("reanalyze_replay", { dbPath, replayId });
}

export async function reanalyzeAll(dbPath: string): Promise<void> {
  return invoke("reanalyze_all", { dbPath });
}

export async function getDefaultDbPath(): Promise<string> {
  return invoke("get_default_db_path");
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

// VOD Splitter

export async function extractPreviewFrame(
  videoPath: string,
  timestampSecs: number
): Promise<string> {
  return invoke("extract_preview_frame", { videoPath, timestampSecs });
}

export async function scanVod(
  videoPath: string,
  roiConfig: VodRoiConfig
): Promise<DetectedSetInfo[]> {
  return invoke("scan_vod", { videoPath, roiConfig });
}

export async function cutVodSets(
  videoPath: string,
  sets: CutSetRequest[],
  outputDir: string
): Promise<string[]> {
  return invoke("cut_vod_sets", { videoPath, sets, outputDir });
}

// Notes

export async function getNotes(dbPath: string, replayId: string): Promise<Note[]> {
  return invoke("get_notes", { dbPath, replayId });
}

export async function addNote(dbPath: string, replayId: string, timestampMs: number, text: string): Promise<Note> {
  return invoke("add_note", { dbPath, replayId, timestampMs, text });
}

export async function updateNote(dbPath: string, noteId: number, text: string, timestampMs?: number): Promise<void> {
  return invoke("update_note", { dbPath, noteId, text, timestampMs });
}

export async function deleteNote(dbPath: string, noteId: number): Promise<void> {
  return invoke("delete_note", { dbPath, noteId });
}

// Drawings

export async function getDrawings(dbPath: string, replayId: string): Promise<Drawing[]> {
  return invoke("get_drawings", { dbPath, replayId });
}

export async function saveDrawing(dbPath: string, replayId: string, timestampMs: number, strokesJson: string): Promise<Drawing | null> {
  return invoke("save_drawing", { dbPath, replayId, timestampMs, strokesJson });
}

export async function deleteDrawing(dbPath: string, drawingId: number): Promise<void> {
  return invoke("delete_drawing", { dbPath, drawingId });
}

// Winner overrides

export async function setRoundWinner(dbPath: string, replayId: string, roundIndex: number, winner: string): Promise<void> {
  return invoke("set_round_winner", { dbPath, replayId, roundIndex, winner });
}

// .spag file format

export async function exportSpag(dbPath: string, replayId: string, outputPath: string): Promise<void> {
  return invoke("export_spag", { dbPath, replayId, outputPath });
}

export async function openSpag(spagPath: string): Promise<SpagSession> {
  return invoke("open_spag", { spagPath });
}

export async function saveSpag(spagPath: string, dbPath: string): Promise<void> {
  return invoke("save_spag", { spagPath, dbPath });
}
