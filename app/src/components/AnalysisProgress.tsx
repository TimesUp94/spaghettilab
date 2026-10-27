import { useState, useCallback, useEffect } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { listen } from "@tauri-apps/api/event";
import { Upload, Film, Play, AlertTriangle, CheckCircle2, ArrowLeft } from "lucide-react";
import { analyzeVideo } from "../api";
import { Button, Spinner } from "./ui";
import { fileName } from "../lib/matches";

interface Props {
  initialPath?: string | null;
  onComplete: (dbPath: string) => void;
  onCancel: () => void;
}

const STAGES = [
  { key: "read", label: "Reading the video" },
  { key: "frames", label: "Reading health, tension, timer and rounds" },
  { key: "events", label: "Finding damage, combos and rounds" },
  { key: "save", label: "Saving the results" },
] as const;

export function AnalysisProgress({ initialPath, onComplete, onCancel }: Props) {
  const [status, setStatus] = useState<"select" | "running" | "error">("select");
  const [videoPath, setVideoPath] = useState<string | null>(initialPath ?? null);
  const [error, setError] = useState<string | null>(null);
  const [pct, setPct] = useState<number | null>(null);
  const [stage, setStage] = useState(0);

  const handleSelectVideo = useCallback(async () => {
    const selected = await open({
      multiple: false,
      filters: [{ name: "Video Files", extensions: ["mp4", "mkv", "avi", "mov", "webm"] }],
    });
    if (selected) setVideoPath(selected as string);
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (!videoPath) return;
    setStatus("running");
    setError(null);
    setPct(null);
    setStage(0);
    try {
      const dbPath = await analyzeVideo(videoPath);
      onComplete(dbPath);
    } catch (err) {
      setError(String(err));
      setStatus("error");
    }
  }, [videoPath, onComplete]);

  useEffect(() => {
    if (status !== "running") return;
    const un = listen<string>("analyze-progress", (e) => {
      const line = e.payload;
      const m = line.match(/Progress (\d+)%/);
      if (m) {
        setPct(Number(m[1]));
        setStage(1);
      } else if (/Processed \d+ frames/.test(line)) {
        setPct(100);
        setStage(2);
      } else if (/Results ->/.test(line)) setStage(3);
    });
    return () => {
      un.then((fn) => fn());
    };
  }, [status]);

  // A dropped file starts at once
  useEffect(() => {
    if (initialPath) handleAnalyze();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="h-full overflow-y-auto flex items-start sm:items-center justify-center p-4 sm:p-8">
      <div className="w-full max-w-xl">
        <button onClick={onCancel} disabled={status === "running"} className="inline-flex items-center gap-1.5 text-sm text-text-secondary hover:text-text-primary mb-4 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed">
          <ArrowLeft size={15} /> Library
        </button>
        <h1 className="text-2xl font-semibold tracking-tight">Analyse a match</h1>
        <p className="text-sm text-text-secondary mt-1 mb-6">
          Spaghetti Lab watches the video and reads the game's HUD: both health bars, tension, the round timer and the
          round hearts. It works best on 1080p recordings of Guilty Gear Strive.
        </p>

        {status === "running" ? (
          <div className="rounded-xl bg-surface-2 border border-surface-4/70 p-5">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-lg bg-surface-3 flex items-center justify-center text-text-secondary shrink-0"><Film size={18} /></div>
              <div className="min-w-0">
                <div className="text-sm font-medium truncate">{fileName(videoPath)}</div>
                <div className="text-xs text-text-muted">Usually a few minutes for a full set. You can leave this screen open.</div>
              </div>
            </div>
            <div className="h-2 rounded-full bg-surface-3 overflow-hidden mb-1.5">
              {pct === null ? (
                <div className="h-full w-1/3 rounded-full bg-accent-purple/70 animate-pulse" />
              ) : (
                <div className="h-full rounded-full bg-accent-purple transition-[width] duration-300" style={{ width: `${pct}%` }} />
              )}
            </div>
            <div className="text-xs text-text-secondary tabular-nums text-right mb-4">{pct === null ? "Starting…" : `${pct}%`}</div>
            <ol className="space-y-2">
              {STAGES.map((s, i) => (
                <li key={s.key} className={`flex items-center gap-2 text-sm ${i < stage ? "text-text-secondary" : i === stage ? "text-text-primary" : "text-text-muted"}`}>
                  {i < stage ? <CheckCircle2 size={16} className="text-accent-green" /> : i === stage ? <Spinner size={14} className="text-accent-purple mx-[1px]" /> : <span className="w-4 h-4 rounded-full border border-surface-4" />}
                  {s.label}
                </li>
              ))}
            </ol>
          </div>
        ) : status === "error" ? (
          <div className="rounded-xl bg-surface-2 border border-p1/40 p-5 space-y-4">
            <div className="flex items-center gap-2 text-p1-light font-medium"><AlertTriangle size={18} /> The analysis failed</div>
            <pre className="bg-surface-1 rounded-lg p-3 text-[11px] text-text-secondary overflow-auto max-h-56 font-mono whitespace-pre-wrap select-text">{error}</pre>
            <div className="flex justify-end gap-2">
              <Button variant="ghost" onClick={onCancel}>Back to the library</Button>
              <Button variant="primary" onClick={() => setStatus("select")}>Try again</Button>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {videoPath ? (
              <div className="flex items-center gap-3 rounded-xl bg-surface-2 border border-surface-4/70 p-4">
                <div className="w-10 h-10 rounded-lg bg-accent-purple/15 flex items-center justify-center text-accent-purple shrink-0"><Film size={18} /></div>
                <div className="min-w-0 flex-1">
                  <div className="text-sm font-medium truncate">{fileName(videoPath)}</div>
                  <div className="text-xs text-text-muted truncate" title={videoPath}>{videoPath}</div>
                </div>
                <Button size="sm" variant="ghost" onClick={handleSelectVideo}>Change</Button>
              </div>
            ) : (
              <button
                onClick={handleSelectVideo}
                className="w-full rounded-xl border-2 border-dashed border-surface-4 hover:border-accent-purple/50 hover:bg-surface-2 transition-colors cursor-pointer py-10 flex flex-col items-center gap-2"
              >
                <div className="w-12 h-12 rounded-full bg-accent-purple/15 text-accent-purple flex items-center justify-center"><Upload size={22} /></div>
                <div className="text-sm font-semibold">Choose a match video</div>
                <div className="text-xs text-text-secondary">or drop it anywhere in the window · MP4, MKV or AVI</div>
              </button>
            )}
            <div className="flex justify-end gap-2">
              <Button variant="ghost" onClick={onCancel}>Cancel</Button>
              <Button variant="primary" icon={Play} onClick={handleAnalyze} disabled={!videoPath}>Start the analysis</Button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
