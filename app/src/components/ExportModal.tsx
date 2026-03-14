import { useState, useCallback } from "react";
import { save } from "@tauri-apps/plugin-dialog";
import { exportClip } from "../api";

interface Props {
  videoPath: string;
  startMs: number;
  endMs: number;
  onClose: () => void;
}

function formatTime(ms: number): string {
  const s = ms / 1000;
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(1);
  return `${m}:${sec.padStart(4, "0")}`;
}

export function ExportModal({ videoPath, startMs, endMs, onClose }: Props) {
  const [status, setStatus] = useState<"idle" | "exporting" | "done" | "error">(
    "idle"
  );
  const [error, setError] = useState<string | null>(null);
  const [padBefore, setPadBefore] = useState(2);
  const [padAfter, setPadAfter] = useState(2);

  const handleExport = useCallback(async () => {
    const outputPath = await save({
      filters: [{ name: "MP4 Video", extensions: ["mp4"] }],
      defaultPath: "clip.mp4",
    });
    if (!outputPath) return;

    setStatus("exporting");
    setError(null);

    try {
      const actualStart = Math.max(0, startMs - padBefore * 1000);
      const actualEnd = endMs + padAfter * 1000;
      await exportClip(videoPath, actualStart, actualEnd, outputPath);
      setStatus("done");
    } catch (err) {
      setError(String(err));
      setStatus("error");
    }
  }, [videoPath, startMs, endMs, padBefore, padAfter]);

  const duration = ((endMs - startMs) / 1000).toFixed(1);
  const totalDuration = (
    (endMs - startMs + (padBefore + padAfter) * 1000) /
    1000
  ).toFixed(1);

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div className="bg-surface-2 rounded-xl border border-surface-4/50 w-[420px] shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-surface-4/30">
          <h3 className="text-sm font-semibold text-text-primary">
            Export Clip
          </h3>
          <button
            onClick={onClose}
            className="text-text-muted hover:text-text-primary text-lg cursor-pointer"
          >
            x
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-4">
          <div className="grid grid-cols-2 gap-3">
            <div className="stat-card !p-3">
              <div className="text-[10px] text-text-muted mb-1">Start</div>
              <div className="text-sm font-mono text-text-primary">
                {formatTime(startMs)}
              </div>
            </div>
            <div className="stat-card !p-3">
              <div className="text-[10px] text-text-muted mb-1">End</div>
              <div className="text-sm font-mono text-text-primary">
                {formatTime(endMs)}
              </div>
            </div>
          </div>

          <div className="text-xs text-text-secondary text-center">
            Segment duration: {duration}s (with padding: {totalDuration}s)
          </div>

          {/* Padding controls */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-[10px] text-text-muted block mb-1">
                Pad before (s)
              </label>
              <input
                type="number"
                min="0"
                max="30"
                value={padBefore}
                onChange={(e) => setPadBefore(parseInt(e.target.value) || 0)}
                className="w-full bg-surface-3 border border-surface-4 rounded px-2 py-1 text-xs font-mono text-text-primary"
              />
            </div>
            <div>
              <label className="text-[10px] text-text-muted block mb-1">
                Pad after (s)
              </label>
              <input
                type="number"
                min="0"
                max="30"
                value={padAfter}
                onChange={(e) => setPadAfter(parseInt(e.target.value) || 0)}
                className="w-full bg-surface-3 border border-surface-4 rounded px-2 py-1 text-xs font-mono text-text-primary"
              />
            </div>
          </div>

          {/* Status */}
          {status === "exporting" && (
            <div className="text-xs text-accent-purple text-center animate-pulse">
              Exporting clip...
            </div>
          )}
          {status === "done" && (
            <div className="text-xs text-accent-green text-center">
              Clip exported successfully
            </div>
          )}
          {status === "error" && (
            <div className="text-xs text-p1 text-center">{error}</div>
          )}
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-2 p-4 border-t border-surface-4/30">
          <button onClick={onClose} className="btn-ghost">
            {status === "done" ? "Close" : "Cancel"}
          </button>
          {status !== "done" && (
            <button
              onClick={handleExport}
              disabled={status === "exporting"}
              className="btn-primary disabled:opacity-50"
            >
              {status === "exporting" ? "Exporting..." : "Export"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
