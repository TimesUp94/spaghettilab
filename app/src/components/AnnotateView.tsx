import { useState, useCallback } from "react";
import { ClipboardList, ArrowLeft, Save } from "lucide-react";
import { Button, IconButton } from "./ui";
import { useMediaQuery } from "../lib/useMediaQuery";
import { save } from "@tauri-apps/plugin-dialog";
import type { Replay, Note, Drawing, SpagzSession } from "../types";
import type { VideoSourceType } from "../lib/videoSourceDetect";
import { VideoPlayer } from "./VideoPlayer";
import { NotesPanel } from "./NotesPanel";
import { QuestionnairePanel } from "./QuestionnairePanel";
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
  const [questionnaireOpen, setQuestionnaireOpen] = useState(false);

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

  const wide = useMediaQuery("(min-width: 1024px)");

  const handleSeek = useCallback((ms: number) => {
    setSeekToMs(ms);
  }, []);

  const video = (
    <VideoPlayer
      fill={wide}
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
  );
  const notesPanel = (
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
  );

  return (
    <div className="h-full flex flex-col min-h-0">
      <header className="h-14 shrink-0 flex items-center gap-2 px-3 border-b border-surface-4/60 bg-surface-1">
        <IconButton icon={ArrowLeft} label="Back to the library" onClick={onBack} />
        <div className="flex-1 min-w-0">
          <div className="text-sm font-semibold truncate">{replay.replay_id}</div>
          <div className="text-[11px] text-text-muted">Notes and drawings</div>
        </div>
        <Button
          size="sm"
          variant={questionnaireOpen ? "primary" : "secondary"}
          icon={ClipboardList}
          onClick={() => setQuestionnaireOpen((o) => !o)}
          title="Open the review checklist"
        >
          Checklist
        </Button>
        <Button size="sm" variant="primary" icon={Save} onClick={handleExportSpagz}>
          <span className="hidden sm:inline">Save as .spagz</span>
        </Button>
      </header>

      {wide ? (
        <div className="flex-1 min-h-0 grid grid-cols-[minmax(0,1fr)_minmax(320px,28%)]">
          <section className="min-h-0 p-3">{video}</section>
          <aside className="min-h-0 overflow-y-auto border-l border-surface-4/60 bg-surface-1/50 p-3">{notesPanel}</aside>
        </div>
      ) : (
        <div className="flex-1 min-h-0 overflow-y-auto">
          <section className="p-3">{video}</section>
          <section className="border-t border-surface-4/60 p-3">{notesPanel}</section>
        </div>
      )}

      {/* Review questionnaire — slides in from the right */}
      <QuestionnairePanel
        open={questionnaireOpen}
        onClose={() => setQuestionnaireOpen(false)}
        dbPath={dbPath}
        replayId={replay.replay_id}
      />
    </div>
  );
}
