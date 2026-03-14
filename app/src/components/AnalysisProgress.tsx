import { useState, useCallback } from "react";
import { open, save } from "@tauri-apps/plugin-dialog";
import { analyzeVideo } from "../api";

interface Props {
  onComplete: (dbPath: string) => void;
  onCancel: () => void;
}

export function AnalysisProgress({ onComplete, onCancel }: Props) {
  const [status, setStatus] = useState<
    "select" | "running" | "done" | "error"
  >("select");
  const [videoPath, setVideoPath] = useState<string | null>(null);
  const [outputDir, setOutputDir] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSelectVideo = useCallback(async () => {
    const selected = await open({
      multiple: false,
      filters: [{ name: "Video Files", extensions: ["mp4", "mkv", "avi"] }],
    });
    if (selected) {
      setVideoPath(selected as string);
    }
  }, []);

  const handleSelectOutput = useCallback(async () => {
    const selected = await open({
      directory: true,
    });
    if (selected) {
      setOutputDir(selected as string);
    }
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (!videoPath || !outputDir) return;
    setStatus("running");
    setError(null);
    try {
      const dbPath = await analyzeVideo(videoPath, outputDir);
      setStatus("done");
      onComplete(dbPath);
    } catch (err) {
      setError(String(err));
      setStatus("error");
    }
  }, [videoPath, outputDir, onComplete]);

  return (
    <div className="h-screen flex items-center justify-center bg-surface-0">
      <div className="w-[500px] bg-surface-2 rounded-xl border border-surface-4/50 shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-surface-4/30">
          <h3 className="text-sm font-semibold text-text-primary">
            Analyze Video
          </h3>
          <button
            onClick={onCancel}
            className="text-text-muted hover:text-text-primary cursor-pointer"
          >
            x
          </button>
        </div>

        <div className="p-6 space-y-4">
          {status === "running" ? (
            <div className="text-center py-8">
              <div className="text-accent-purple text-sm animate-pulse mb-2">
                Analyzing video...
              </div>
              <div className="text-text-muted text-xs">
                This may take a few minutes depending on video length.
              </div>
              <div className="mt-6 w-full bg-surface-3 rounded-full h-1.5">
                <div className="bg-accent-purple h-1.5 rounded-full animate-pulse w-2/3" />
              </div>
            </div>
          ) : status === "error" ? (
            <div className="space-y-4">
              <div className="text-p1 text-sm text-center">
                Analysis failed
              </div>
              <div className="bg-surface-3 rounded-lg p-3 text-xs text-text-muted overflow-auto max-h-40 font-mono">
                {error}
              </div>
              <div className="flex justify-center gap-2">
                <button onClick={onCancel} className="btn-ghost">
                  Cancel
                </button>
                <button
                  onClick={() => setStatus("select")}
                  className="btn-primary"
                >
                  Try Again
                </button>
              </div>
            </div>
          ) : (
            <>
              {/* Video path */}
              <div>
                <label className="text-xs text-text-muted block mb-1.5">
                  Video File
                </label>
                <div className="flex gap-2">
                  <div className="flex-1 bg-surface-3 border border-surface-4 rounded-lg px-3 py-2 text-xs text-text-secondary truncate">
                    {videoPath || "No file selected"}
                  </div>
                  <button
                    onClick={handleSelectVideo}
                    className="btn-primary !py-2"
                  >
                    Browse
                  </button>
                </div>
              </div>

              {/* Output directory */}
              <div>
                <label className="text-xs text-text-muted block mb-1.5">
                  Output Directory
                </label>
                <div className="flex gap-2">
                  <div className="flex-1 bg-surface-3 border border-surface-4 rounded-lg px-3 py-2 text-xs text-text-secondary truncate">
                    {outputDir || "No directory selected"}
                  </div>
                  <button
                    onClick={handleSelectOutput}
                    className="btn-primary !py-2"
                  >
                    Browse
                  </button>
                </div>
              </div>

              <div className="text-[10px] text-text-muted">
                The analysis will extract health data, detect damage events,
                and identify round boundaries. Results are saved to a SQLite
                database in the output directory.
              </div>

              <div className="flex justify-end gap-2 pt-2">
                <button onClick={onCancel} className="btn-ghost">
                  Cancel
                </button>
                <button
                  onClick={handleAnalyze}
                  disabled={!videoPath || !outputDir}
                  className="btn-primary disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  Start Analysis
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
