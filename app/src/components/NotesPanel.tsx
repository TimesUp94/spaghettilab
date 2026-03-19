import { useState, useCallback, useMemo } from "react";
import type { Note, Drawing } from "../types";

interface Props {
  notes: Note[];
  drawings?: Drawing[];
  onSeek: (ms: number) => void;
  onAdd: (timestampMs: number, text: string) => void;
  onUpdate: (noteId: number, text: string) => void;
  onDelete: (noteId: number) => void;
  onDeleteDrawing?: (drawingId: number) => void;
  currentTimeMs: number;
}

function formatTime(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const m = Math.floor(totalSec / 60);
  const s = totalSec % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function DrawingPreview({ strokesJson }: { strokesJson: string }) {
  const canvasRef = useCallback((canvas: HTMLCanvasElement | null) => {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    let strokes: { points: [number, number][]; color: string; size: number; isEraser: boolean }[];
    try {
      strokes = JSON.parse(strokesJson);
    } catch {
      return;
    }

    for (const stroke of strokes) {
      if (stroke.points.length < 2) continue;
      ctx.lineWidth = Math.max(1, stroke.size * (w / 1920));
      ctx.lineCap = "round";
      ctx.lineJoin = "round";

      if (stroke.isEraser) {
        ctx.globalCompositeOperation = "destination-out";
        ctx.strokeStyle = "rgba(0,0,0,1)";
      } else {
        ctx.globalCompositeOperation = "source-over";
        ctx.strokeStyle = stroke.color;
      }

      ctx.beginPath();
      ctx.moveTo(stroke.points[0][0] * w, stroke.points[0][1] * h);
      for (let i = 1; i < stroke.points.length; i++) {
        ctx.lineTo(stroke.points[i][0] * w, stroke.points[i][1] * h);
      }
      ctx.stroke();
    }
  }, [strokesJson]);

  return (
    <canvas
      ref={canvasRef}
      width={120}
      height={68}
      className="rounded bg-black/40 border border-surface-4/30 shrink-0"
      style={{ width: 60, height: 34 }}
    />
  );
}

function DrawingListItem({
  drawing,
  onSeek,
  onDelete,
}: {
  drawing: Drawing;
  onSeek: (ms: number) => void;
  onDelete?: (drawingId: number) => void;
}) {
  return (
    <div
      onClick={() => onSeek(drawing.timestamp_ms)}
      className="flex items-center gap-3 p-2 rounded-lg bg-surface-2 border border-surface-4/30 hover:border-accent-gold/20 transition-colors group cursor-pointer"
    >
      <span
        className="text-[11px] font-mono text-accent-gold shrink-0 hover:underline"
        title="Seek to this timestamp"
      >
        {formatTime(drawing.timestamp_ms)}
      </span>
      <DrawingPreview strokesJson={drawing.strokes_json} />
      <span className="flex-1 text-[10px] text-text-muted">Drawing</span>
      {onDelete && (
        <button
          onClick={(e) => { e.stopPropagation(); onDelete(drawing.drawing_id); }}
          className="text-[10px] text-text-muted hover:text-p1 cursor-pointer opacity-0 group-hover:opacity-100 transition-opacity"
        >
          delete
        </button>
      )}
    </div>
  );
}

export function NotesPanel({
  notes,
  drawings = [],
  onSeek,
  onAdd,
  onUpdate,
  onDelete,
  onDeleteDrawing,
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

      {/* Combined notes + drawings list, sorted by timestamp */}
      {(() => {
        type TimelineItem =
          | { kind: "note"; note: Note; timestamp_ms: number }
          | { kind: "drawing"; drawing: Drawing; timestamp_ms: number };

        const items: TimelineItem[] = [
          ...notes.map((n) => ({ kind: "note" as const, note: n, timestamp_ms: n.timestamp_ms })),
          ...drawings.map((d) => ({ kind: "drawing" as const, drawing: d, timestamp_ms: d.timestamp_ms })),
        ].sort((a, b) => a.timestamp_ms - b.timestamp_ms);

        if (items.length === 0) {
          return (
            <div className="text-center py-8 text-text-muted text-xs">
              No notes yet. Pause the video and add one above, or press D to draw.
            </div>
          );
        }

        return (
          <div className="space-y-1">
            {items.map((item) => {
              if (item.kind === "note") {
                const note = item.note;
                return (
                  <div
                    key={`note-${note.note_id}`}
                    onClick={() => onSeek(note.timestamp_ms)}
                    className="flex items-start gap-3 p-2 rounded-lg bg-surface-2 border border-surface-4/30 hover:border-accent-green/20 transition-colors group cursor-pointer"
                  >
                    <span
                      className="text-[11px] font-mono text-accent-green shrink-0 pt-0.5 hover:underline"
                      title="Seek to this timestamp"
                    >
                      {formatTime(note.timestamp_ms)}
                    </span>

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
                            onClick={(e) => { e.stopPropagation(); startEdit(note); }}
                            className="text-[10px] text-text-muted hover:text-text-primary cursor-pointer"
                          >
                            edit
                          </button>
                          <button
                            onClick={(e) => { e.stopPropagation(); onDelete(note.note_id); }}
                            className="text-[10px] text-text-muted hover:text-p1 cursor-pointer"
                          >
                            delete
                          </button>
                        </div>
                      </>
                    )}
                  </div>
                );
              } else {
                const drawing = item.drawing;
                return (
                  <DrawingListItem
                    key={`draw-${drawing.drawing_id}`}
                    drawing={drawing}
                    onSeek={onSeek}
                    onDelete={onDeleteDrawing}
                  />
                );
              }
            })}
          </div>
        );
      })()}
    </div>
  );
}
