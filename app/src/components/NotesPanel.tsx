import { useState, useCallback } from "react";
import type { Note } from "../types";

interface Props {
  notes: Note[];
  onSeek: (ms: number) => void;
  onAdd: (timestampMs: number, text: string) => void;
  onUpdate: (noteId: number, text: string) => void;
  onDelete: (noteId: number) => void;
  currentTimeMs: number;
}

function formatTime(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const m = Math.floor(totalSec / 60);
  const s = totalSec % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function NotesPanel({
  notes,
  onSeek,
  onAdd,
  onUpdate,
  onDelete,
  currentTimeMs,
}: Props) {
  const [newText, setNewText] = useState("");
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editText, setEditText] = useState("");

  const handleAdd = useCallback(() => {
    const text = newText.trim();
    if (!text) return;
    onAdd(currentTimeMs, text);
    setNewText("");
  }, [newText, currentTimeMs, onAdd]);

  const startEdit = (note: Note) => {
    setEditingId(note.note_id);
    setEditText(note.text);
  };

  const saveEdit = () => {
    if (editingId === null) return;
    const text = editText.trim();
    if (text) {
      onUpdate(editingId, text);
    }
    setEditingId(null);
    setEditText("");
  };

  const cancelEdit = () => {
    setEditingId(null);
    setEditText("");
  };

  return (
    <div className="space-y-3">
      {/* Add note form */}
      <div className="flex gap-2 items-center">
        <span className="text-[10px] text-text-muted font-mono shrink-0">
          @ {formatTime(currentTimeMs)}
        </span>
        <input
          type="text"
          value={newText}
          onChange={(e) => setNewText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") handleAdd();
          }}
          placeholder="Add a note at current timestamp..."
          className="flex-1 bg-surface-3 border border-surface-4 rounded-lg px-3 py-1.5 text-xs text-text-primary placeholder:text-text-muted/50 outline-none focus:border-accent-green/50"
        />
        <button
          onClick={handleAdd}
          disabled={!newText.trim()}
          className="btn-primary !py-1.5 !px-3 !text-xs disabled:opacity-30 disabled:cursor-not-allowed"
        >
          Add
        </button>
      </div>

      {/* Notes list */}
      {notes.length === 0 ? (
        <div className="text-center py-8 text-text-muted text-xs">
          No notes yet. Pause the video and add one above.
        </div>
      ) : (
        <div className="space-y-1">
          {notes.map((note) => (
            <div
              key={note.note_id}
              className="flex items-start gap-3 p-2 rounded-lg bg-surface-2 border border-surface-4/30 hover:border-accent-green/20 transition-colors group"
            >
              <button
                onClick={() => onSeek(note.timestamp_ms)}
                className="text-[11px] font-mono text-accent-green shrink-0 pt-0.5 cursor-pointer hover:underline"
                title="Seek to this timestamp"
              >
                {formatTime(note.timestamp_ms)}
              </button>

              {editingId === note.note_id ? (
                <div className="flex-1 flex gap-1.5">
                  <input
                    type="text"
                    value={editText}
                    onChange={(e) => setEditText(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") saveEdit();
                      if (e.key === "Escape") cancelEdit();
                    }}
                    autoFocus
                    className="flex-1 bg-surface-3 border border-accent-green/30 rounded px-2 py-0.5 text-xs text-text-primary outline-none"
                  />
                  <button
                    onClick={saveEdit}
                    className="text-[10px] text-accent-green hover:underline cursor-pointer"
                  >
                    save
                  </button>
                  <button
                    onClick={cancelEdit}
                    className="text-[10px] text-text-muted hover:underline cursor-pointer"
                  >
                    cancel
                  </button>
                </div>
              ) : (
                <>
                  <span className="flex-1 text-xs text-text-secondary pt-0.5">
                    {note.text}
                  </span>
                  <div className="flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity shrink-0">
                    <button
                      onClick={() => startEdit(note)}
                      className="text-[10px] text-text-muted hover:text-text-primary cursor-pointer"
                    >
                      edit
                    </button>
                    <button
                      onClick={() => onDelete(note.note_id)}
                      className="text-[10px] text-text-muted hover:text-p1 cursor-pointer"
                    >
                      delete
                    </button>
                  </div>
                </>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
