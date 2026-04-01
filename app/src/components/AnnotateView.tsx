import { useState, useCallback } from "react";
import { save } from "@tauri-apps/plugin-dialog";
import type { Replay, Note, Drawing, SpagzSession } from "../types";
import type { VideoSourceType } from "../lib/videoSourceDetect";
import { VideoPlayer } from "./VideoPlayer";
import { NotesPanel } from "./NotesPanel";
import {
  addNote,
  updateNote,
  deleteNote,
  getNotes,
  getDrawings,
  saveDrawing,
  deleteDrawing,
  exportSpagz,
  saveSpagz,
} from "../api";

interface Props {
  videoSrc: string | null;
  srcType: VideoSourceType;
  embedId?: string;
  replay: Replay;
  dbPath: string;
  spagzSession: SpagzSession | null;
  notes: Note[];
  drawings: Drawing[];
  onNotesChange: (notes: Note[]) => void;
  onDrawingsChange: (drawings: Drawing[]) => void;
  onBack: () => void;
  onLocateVideo?: () => void;
}

export function AnnotateView({
  videoSrc,
  srcType,
  embedId,
  replay,
  dbPath,
  spagzSession,
  notes,
  drawings,
  onNotesChange,
  onDrawingsChange,
  onBack,
  onLocateVideo,
}: Props) {
  const [seekToMs, setSeekToMs] = useState<number | null>(null);
  const [currentTimeMs, setCurrentTimeMs] = useState(0);

  // Note handlers
  const handleAddNote = useCallback(
    async (timestampMs: number, text: string) => {
      const note = await addNote(dbPath, replay.replay_id, timestampMs, text);
      onNotesChange([...notes, note]);
      // Auto-save to .spagz if we have a session
      if (spagzSession?.spagz_path) {
        saveSpagz(spagzSession.spagz_path, dbPath).catch(() => {});
      }
    },
    [dbPath, replay.replay_id, notes, onNotesChange, spagzSession]
  );

  const handleUpdateNote = useCallback(
    async (noteId: number, text: string, timestampMs?: number) => {
      await updateNote(dbPath, noteId, text, timestampMs);
      const updated = await getNotes(dbPath, replay.replay_id);
      onNotesChange(updated);
      if (spagzSession?.spagz_path) {
        saveSpagz(spagzSession.spagz_path, dbPath).catch(() => {});
      }
    },
    [dbPath, replay.replay_id, onNotesChange, spagzSession]
  );

  const handleDeleteNote = useCallback(
    async (noteId: number) => {
      await deleteNote(dbPath, noteId);
      onNotesChange(notes.filter((n) => n.note_id !== noteId));
      if (spagzSession?.spagz_path) {
        saveSpagz(spagzSession.spagz_path, dbPath).catch(() => {});
      }
    },
    [dbPath, notes, onNotesChange, spagzSession]
  );

  // Drawing handlers
  const handleSaveDrawing = useCallback(
    async (timestampMs: number, strokesJson: string) => {
      const result = await saveDrawing(dbPath, replay.replay_id, timestampMs, strokesJson);
      const updated = await getDrawings(dbPath, replay.replay_id);
      onDrawingsChange(updated);
      if (spagzSession?.spagz_path) {
        saveSpagz(spagzSession.spagz_path, dbPath).catch(() => {});
      }
      return result;
    },
    [dbPath, replay.replay_id, onDrawingsChange, spagzSession]
  );

  const handleDeleteDrawing = useCallback(
    async (drawingId: number) => {
      await deleteDrawing(dbPath, drawingId);
      onDrawingsChange(drawings.filter((d) => d.drawing_id !== drawingId));
      if (spagzSession?.spagz_path) {
        saveSpagz(spagzSession.spagz_path, dbPath).catch(() => {});
      }
    },
    [dbPath, drawings, onDrawingsChange, spagzSession]
  );

  const handleExportSpagz = useCallback(async () => {
    const path = await save({
      defaultPath: `${replay.replay_id}.spagz`,
      filters: [{ name: "Spaghetti Lab Analysis", extensions: ["spagz"] }],
    });
    if (!path) return;
    await exportSpagz(dbPath, replay.replay_id, path);
  }, [dbPath, replay.replay_id]);

  const handleSeek = useCallback((ms: number) => {
    setSeekToMs(ms);
  }, []);

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-2 bg-surface-1 border-b border-surface-4/30 shrink-0">
        <button
          onClick={onBack}
          className="btn-ghost !px-2 !py-1 text-xs"
        >
          &larr; Back
        </button>
        <div className="flex-1 min-w-0">
          <span className="text-sm font-medium text-text-primary truncate block">
            {replay.replay_id}
          </span>
        </div>
        <button
          onClick={handleExportSpagz}
          className="btn-primary !py-1.5 !px-3 !text-xs"
        >
          Save as .spagz
        </button>
      </div>

      {/* Main content */}
      <div className="flex-1 flex gap-4 p-4 overflow-hidden">
        {/* Video player */}
        <div className="flex-1 min-w-0">
          <VideoPlayer
            src={videoSrc}
            srcType={srcType}
            embedId={embedId}
            seekToMs={seekToMs}
            onSeeked={() => setSeekToMs(null)}
            durationMs={replay.duration_ms}
            rounds={[]}
            damageEvents={[]}
            selectedMatch={null}
            onClearSelection={() => {}}
            onLocateVideo={onLocateVideo}
            onTimeUpdate={setCurrentTimeMs}
            notes={notes}
            drawings={drawings}
            onSaveDrawing={handleSaveDrawing}
            onDeleteDrawing={handleDeleteDrawing}
          />
        </div>

        {/* Notes panel */}
        <div className="w-[320px] shrink-0 overflow-y-auto">
          <NotesPanel
            notes={notes}
            drawings={drawings}
            currentTimeMs={currentTimeMs}
            onAdd={handleAddNote}
            onUpdate={handleUpdateNote}
            onDelete={handleDeleteNote}
            onDeleteDrawing={handleDeleteDrawing}
            onSeek={handleSeek}
          />
        </div>
      </div>
    </div>
  );
}
