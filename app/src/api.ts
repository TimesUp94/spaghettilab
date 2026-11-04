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
  SpagzSession,
  StriveExportOptions,
  StriveExportSummary,
} from "./types";
import type { LabAccount, LabConfig, LabOptions, LabQuota, LabResult, LabServiceStatus, RemoteError, RemoteRunning } from "./lib/lab";

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

export async function downloadVod(url: string, outputDir: string): Promise<string> {
  return invoke("download_vod", { url, outputDir });
}

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

// Review questionnaire answers — free-form checklist stored in the same DB
// that rides inside .spag/.spagz files. Keys are template-id-prefixed
// (e.g. "ggst.roundstart.main_options"); meta keys start with an underscore
// (e.g. "_meta.selected_template").

export async function getQuestionnaireAnswers(
  dbPath: string,
  replayId: string
): Promise<Record<string, string>> {
  return invoke("get_questionnaire_answers", { dbPath, replayId });
}

export async function setQuestionnaireAnswer(
  dbPath: string,
  replayId: string,
  questionKey: string,
  answerText: string
): Promise<void> {
  return invoke("set_questionnaire_answer", { dbPath, replayId, questionKey, answerText });
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

// .spagz file format (lightweight, no video)

export async function exportSpagz(dbPath: string, replayId: string, outputPath: string): Promise<void> {
  return invoke("export_spagz", { dbPath, replayId, outputPath });
}

export async function openSpagz(spagzPath: string): Promise<SpagzSession> {
  return invoke("open_spagz", { spagzPath });
}

export async function saveSpagz(spagzPath: string, dbPath: string): Promise<void> {
  return invoke("save_spagz", { spagzPath, dbPath });
}

// Quick annotation sessions

export async function createQuickSession(
  videoPath: string,
  videoUrl?: string
): Promise<SpagzSession> {
  return invoke("create_quick_session", { videoPath, videoUrl: videoUrl ?? null });
}

// Strive replay export: drives the running game and records saved replays

export async function exportStriveReplays(options: StriveExportOptions): Promise<StriveExportSummary> {
  return invoke("export_strive_replays", { options });
}

/** Stop after the current replay (`now` false) or at once, finalising the current file. */
export async function cancelStriveExport(now: boolean): Promise<void> {
  return invoke("cancel_strive_export", { now });
}

// Lab analysis (research models, see tools/ggst_logger/model/LAB_RESULT.md)

export async function runLabAnalysis(dbPath: string, replayId: string, options: LabOptions): Promise<LabResult> {
  return invoke("run_lab_analysis", { dbPath, replayId, options });
}

export async function cancelLabAnalysis(replayId?: string): Promise<void> {
  return invoke("cancel_lab_analysis", { replayId: replayId ?? null });
}

/** The replay whose lab analysis is running, or null. */
export async function labAnalysisRunning(): Promise<string | null> {
  return invoke("lab_analysis_running");
}

export async function getLabResult(replayId: string): Promise<LabResult | null> {
  return invoke("get_lab_result", { replayId });
}

export async function deleteLabResult(replayId: string): Promise<void> {
  return invoke("delete_lab_result", { replayId });
}

// Lab sign-in and hosted analysis (docs/desktop-login.md). These fail with a RemoteError object, not a string.

export async function labConfig(): Promise<LabConfig> {
  return invoke("lab_config");
}

export async function labAccount(): Promise<LabAccount> {
  return invoke("lab_account");
}

/** Opens the system browser; resolves when the login ends. */
export async function labSignIn(): Promise<LabAccount> {
  return invoke("lab_sign_in");
}

export async function labSignInReopen(): Promise<void> {
  return invoke("lab_sign_in_reopen");
}

export async function labSignInCancel(): Promise<void> {
  return invoke("lab_sign_in_cancel");
}

export async function labSignOut(): Promise<LabAccount> {
  return invoke("lab_sign_out");
}

export async function labServiceStatus(): Promise<LabServiceStatus> {
  return invoke("lab_service_status");
}

/** null when the analysis isn't available right now. */
export async function labQuota(): Promise<LabQuota | null> {
  return invoke("lab_quota");
}

export async function labOpenWeb(page: "support" | "terms" | "privacy"): Promise<void> {
  return invoke("lab_open_web", { page });
}

export async function runLabAnalysisRemote(dbPath: string, replayId: string, options: { again?: boolean; video_path?: string }): Promise<LabResult> {
  return invoke("run_lab_analysis_remote", { dbPath, replayId, options });
}

export async function cancelLabAnalysisRemote(replayId: string): Promise<{ phase: string }> {
  return invoke("cancel_lab_analysis_remote", { replayId });
}

/** What a cancel means right now: asks the server, because the refund depends on the job's status at that moment. */
export async function labRemoteCancelPhase(replayId: string): Promise<{ phase: RemoteRunning["phase"] }> {
  return invoke("lab_remote_cancel_phase", { replayId });
}

export async function labRemoteRunning(): Promise<RemoteRunning | null> {
  return invoke("lab_remote_running");
}

export async function labRemoteTakeError(replayId: string): Promise<RemoteError | null> {
  return invoke("lab_remote_take_error", { replayId });
}

/** Leftovers of an earlier session: temp copies, failed deletes, saved jobs. */
export async function labRemoteResume(): Promise<void> {
  return invoke("lab_remote_resume");
}
