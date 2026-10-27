import { ArrowLeft } from "lucide-react";
import { useState, useCallback, useEffect } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { listen } from "@tauri-apps/api/event";
import { analyzeVideo, cancelStriveExport, exportStriveReplays } from "../api";
import type { StriveExportEvent, StriveExportOptions, StriveExportSummary } from "../types";

interface Props {
  /** Called with the analysis DB once the exported videos are analyzed. */
  onComplete: (dbPath: string) => void;
  onCancel: () => void;
}

type Phase = "setup" | "running" | "done" | "analyzing" | "error";
type Mode = "count" | "since" | "all";
type RowStatus = "listed" | "recording" | "recorded" | "partial" | "skipped";

interface Row {
  index: number;
  file: string;
  date?: string | null;
  time?: string | null;
  p1?: string | null;
  p2?: string | null;
  status: RowStatus;
  duration?: number;
}

const OUT_KEY = "strive-export-output-dir";

function todayIso(): string {
  const d = new Date();
  const p = (n: number) => n.toString().padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
}

function loadOutputDir(): string | null {
  try {
    return localStorage.getItem(OUT_KEY);
  } catch {
    return null;
  }
}

function saveOutputDir(dir: string) {
  try {
    localStorage.setItem(OUT_KEY, dir);
  } catch {
    /* per-viewer convenience only */
  }
}

const STATUS_LABEL: Record<RowStatus, string> = {
  listed: "listed",
  recording: "recording",
  recorded: "done",
  partial: "partial",
  skipped: "already exported",
};

const STATUS_CLASS: Record<RowStatus, string> = {
  listed: "text-text-muted",
  recording: "text-p1 animate-pulse",
  recorded: "text-accent-green",
  partial: "text-accent-gold",
  skipped: "text-text-muted",
};

export function StriveExportView({ onComplete, onCancel }: Props) {
  const [phase, setPhase] = useState<Phase>("setup");
  const [mode, setMode] = useState<Mode>("count");
  const [count, setCount] = useState(10);
  const [since, setSince] = useState(todayIso());
  const [start, setStart] = useState(0);
  const [dryRun, setDryRun] = useState(false);
  const [lang, setLang] = useState("auto");
  const [outputDir, setOutputDir] = useState<string | null>(loadOutputDir);
  const [acknowledged, setAcknowledged] = useState(false);

  const [rows, setRows] = useState<Row[]>([]);
  const [status, setStatus] = useState("");
  const [warnings, setWarnings] = useState<string[]>([]);
  const [captureInfo, setCaptureInfo] = useState("");
  const [total, setTotal] = useState<number | null>(null);
  const [stopping, setStopping] = useState<"none" | "soft" | "hard">("none");
  const [summary, setSummary] = useState<StriveExportSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [analyzeProgress, setAnalyzeProgress] = useState("");
  const [analyzeErrors, setAnalyzeErrors] = useState<string[]>([]);
  const [dbAfterErrors, setDbAfterErrors] = useState<string | null>(null);

  const upsertRow = useCallback((index: number, patch: Partial<Row>) => {
    setRows((prev) => {
      const i = prev.findIndex((r) => r.index === index);
      if (i === -1) {
        return [...prev, { index, file: "", status: "listed", ...patch } as Row].sort((a, b) => a.index - b.index);
      }
      const next = prev.slice();
      next[i] = { ...next[i], ...patch };
      return next;
    });
  }, []);

  // Progress events from the exporter (one JSON object per event)
  useEffect(() => {
    let cancelled = false;
    const unlistenPromise = listen<string>("strive-export-progress", (event) => {
      if (cancelled) return;
      let ev: StriveExportEvent;
      try {
        ev = JSON.parse(event.payload);
      } catch {
        return;
      }
      switch (ev.event) {
        case "ffmpeg":
          setCaptureInfo(`${ev.capture === "gfxcapture" ? "window capture" : "screen capture"}, ${ev.encoder ?? "no encoder"}`);
          break;
        case "warning":
          if (ev.message) setWarnings((w) => (w.includes(ev.message!) ? w : [...w, ev.message!]));
          break;
        case "status":
          if (ev.message) setStatus(ev.message);
          break;
        case "screen":
          setStatus(`Game screen: ${ev.detail ?? ev.screen}`);
          break;
        case "list":
          if (typeof ev.total === "number") setTotal(ev.total);
          break;
        case "entry":
          upsertRow(ev.index!, { file: ev.file ?? "", date: ev.date, time: ev.time, p1: ev.p1, p2: ev.p2, status: "listed" });
          break;
        case "recording":
          upsertRow(ev.index!, { status: "recording" });
          setStatus(`Recording replay ${ev.index! + 1}...`);
          break;
        case "recorded":
          upsertRow(ev.index!, { status: ev.partial ? "partial" : "recorded", duration: ev.duration_s });
          break;
        case "skipped":
          upsertRow(ev.index!, { status: "skipped" });
          break;
        case "error":
          if (ev.message) setError(ev.message);
          break;
      }
    });
    return () => {
      cancelled = true;
      unlistenPromise.then((fn) => fn());
    };
  }, [upsertRow]);

  const handleSelectOutput = useCallback(async () => {
    const dir = await open({ directory: true });
    if (dir) {
      setOutputDir(dir as string);
      saveOutputDir(dir as string);
    }
  }, []);

  const handleStart = useCallback(async () => {
    if (!outputDir) return;
    const options: StriveExportOptions = {
      output_dir: outputDir,
      mode,
      count: mode === "count" ? count : undefined,
      since: mode === "since" ? since : undefined,
      start: start > 0 ? start : undefined,
      dry_run: dryRun,
      lang: lang.trim() && lang !== "auto" ? lang.trim() : undefined,
    };
    setRows([]);
    setWarnings([]);
    setStatus("Starting...");
    setCaptureInfo("");
    setTotal(null);
    setError(null);
    setSummary(null);
    setDbAfterErrors(null);
    setAnalyzeProgress("");
    setAnalyzeErrors([]);
    setStopping("none");
    setPhase("running");
    try {
      const result = await exportStriveReplays(options);
      setSummary(result);
      setPhase("done");
    } catch (err) {
      setError(String(err));
      setPhase("error");
    }
  }, [outputDir, mode, count, since, start, dryRun, lang]);

  const handleStop = useCallback(async () => {
    const now = stopping !== "none";
    setStopping(now ? "hard" : "soft");
    try {
      await cancelStriveExport(now);
    } catch (err) {
      setWarnings((w) => [...w, `Could not stop: ${String(err)}`]);
    }
  }, [stopping]);

  // Files recorded so far: the summary's list, or (after an error) the rows
  // that finished recording.
  const exportedFiles = summary
    ? summary.exported
    : outputDir
      ? rows
          .filter((r) => (r.status === "recorded" || r.status === "partial") && r.file)
          .map((r) => `${outputDir.replace(/[\\/]+$/, "")}\\${r.file}`)
      : [];

  const handleAnalyze = useCallback(async () => {
    const files = exportedFiles;
    if (files.length === 0) return;
    setPhase("analyzing");
    setAnalyzeErrors([]);
    let dbPath: string | null = null;
    const errors: string[] = [];
    for (let i = 0; i < files.length; i++) {
      const name = files[i].split(/[\\/]/).pop();
      setAnalyzeProgress(`Analyzing ${i + 1} / ${files.length}: ${name}`);
      try {
        dbPath = await analyzeVideo(files[i]);
      } catch (err) {
        errors.push(`${name}: ${String(err).slice(0, 300)}`);
        setAnalyzeErrors([...errors]);
      }
    }
    if (dbPath && errors.length === 0) {
      onComplete(dbPath);
    } else {
      setAnalyzeProgress(
        dbPath
          ? `Analyzed ${files.length - errors.length} of ${files.length} videos.`
          : "No video could be analyzed."
      );
      setPhase("done");
      if (dbPath) setDbAfterErrors(dbPath);
    }
  }, [exportedFiles, onComplete]);

  const running = phase === "running" || phase === "analyzing";
  const recordedCount = rows.filter((r) => r.status === "recorded" || r.status === "partial").length;

  return (
    <div className="h-full overflow-y-auto p-4 sm:p-8">
      <div className="w-full max-w-2xl mx-auto">
        <button onClick={onCancel} disabled={running} className="inline-flex items-center gap-1.5 text-sm text-text-secondary hover:text-text-primary mb-4 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed">
          <ArrowLeft size={15} /> Library
        </button>
        <h1 className="text-2xl font-semibold tracking-tight">Export Strive replays</h1>
        <p className="text-sm text-text-secondary mt-1 mb-6">Plays your saved replays in Guilty Gear Strive and records each one, then analyses them for you.</p>
        <div className="bg-surface-2 rounded-xl border border-surface-4/70 flex flex-col">

        <div className="p-6 space-y-4 overflow-y-auto">
          {phase === "setup" && (
            <>
              <div className="bg-accent-gold/10 border border-accent-gold/30 rounded-lg p-3 text-xs text-text-secondary space-y-1.5">
                <div className="text-accent-gold font-medium">Before you start</div>
                <ul className="list-disc pl-4 space-y-1">
                  <li>
                    Guilty Gear -Strive- must be running and sitting at the <b>title screen</b> (or the main
                    menu), set to <b>keyboard controls</b> (U confirm, W A S D move, K / L switch tabs).
                  </li>
                  <li>
                    Spaghetti Lab <b>drives the game window</b>: it sends keys to it, opens Collection &gt; Replay
                    &gt; Saved replays and plays each replay while recording it. The game can stay behind other
                    windows.
                  </li>
                  <li>
                    You can keep using the PC, but <b>don't click into the game, type in it or minimise it</b>{" "}
                    until the export finishes. Replays record in real time (about 3 minutes each). Without FFmpeg
                    8+ (window capture) the game has to stay in front, so leave the PC alone then.
                  </li>
                  <li>
                    It only ever confirms "View replay" and "Back to the replay list"; anything it doesn't
                    recognise makes it wait or stop, never press.
                  </li>
                </ul>
              </div>

              {/* Which replays */}
              <div>
                <label className="text-xs text-text-muted block mb-1.5">Which replays</label>
                <div className="space-y-2">
                  <label className="flex items-center gap-2 text-xs text-text-primary cursor-pointer">
                    <input type="radio" checked={mode === "count"} onChange={() => setMode("count")} className="accent-accent-purple" />
                    The newest
                    <input
                      type="number"
                      min={1}
                      max={100}
                      value={count}
                      onChange={(e) => setCount(Math.max(1, parseInt(e.target.value || "1", 10)))}
                      onFocus={() => setMode("count")}
                      className="w-16 bg-surface-3 border border-surface-4 rounded px-2 py-1 text-xs text-text-primary outline-none focus:border-accent-purple/50"
                    />
                    replays
                  </label>
                  <label className="flex items-center gap-2 text-xs text-text-primary cursor-pointer">
                    <input type="radio" checked={mode === "since"} onChange={() => setMode("since")} className="accent-accent-purple" />
                    Played on or after
                    <input
                      type="date"
                      value={since}
                      onChange={(e) => setSince(e.target.value)}
                      onFocus={() => setMode("since")}
                      className="bg-surface-3 border border-surface-4 rounded px-2 py-1 text-xs text-text-primary outline-none focus:border-accent-purple/50"
                    />
                    <button
                      type="button"
                      onClick={() => {
                        setMode("since");
                        setSince(todayIso());
                      }}
                      className="text-[10px] text-accent-purple hover:underline cursor-pointer"
                    >
                      today
                    </button>
                  </label>
                  <label className="flex items-center gap-2 text-xs text-text-primary cursor-pointer">
                    <input type="radio" checked={mode === "all"} onChange={() => setMode("all")} className="accent-accent-purple" />
                    All saved replays
                  </label>
                </div>
              </div>

              {/* Output directory */}
              <div>
                <label className="text-xs text-text-muted block mb-1.5">Output folder</label>
                <div className="flex gap-2">
                  <div className="flex-1 bg-surface-3 border border-surface-4 rounded-lg px-3 py-2 text-xs text-text-secondary truncate">
                    {outputDir || "No folder selected"}
                  </div>
                  <button onClick={handleSelectOutput} className="btn-primary !py-2">
                    Browse
                  </button>
                </div>
                <div className="text-[10px] text-text-muted mt-1">
                  One mp4 per replay, named by match date and players, plus manifest.json. Replays already in
                  the folder are skipped.
                </div>
              </div>

              <details className="text-xs text-text-secondary">
                <summary className="cursor-pointer text-text-muted">Advanced</summary>
                <div className="mt-2 space-y-2 pl-1">
                  <label className="flex items-center gap-2">
                    Skip the newest
                    <input
                      type="number"
                      min={0}
                      value={start}
                      onChange={(e) => setStart(Math.max(0, parseInt(e.target.value || "0", 10)))}
                      className="w-16 bg-surface-3 border border-surface-4 rounded px-2 py-1 text-xs text-text-primary outline-none"
                    />
                    replays first
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input type="checkbox" checked={dryRun} onChange={(e) => setDryRun(e.target.checked)} className="accent-accent-purple" />
                    List only (walk the list, record nothing)
                  </label>
                  <label className="flex items-center gap-2">
                    Game UI language
                    <input
                      type="text"
                      value={lang}
                      onChange={(e) => setLang(e.target.value)}
                      className="w-20 bg-surface-3 border border-surface-4 rounded px-2 py-1 text-xs text-text-primary outline-none"
                    />
                    <span className="text-[10px] text-text-muted">auto, or a code calibrated with the CLI</span>
                  </label>
                </div>
              </details>

              <label className="flex items-start gap-2 text-xs text-text-primary cursor-pointer">
                <input
                  type="checkbox"
                  checked={acknowledged}
                  onChange={(e) => setAcknowledged(e.target.checked)}
                  className="accent-accent-purple mt-0.5"
                />
                The game is at the title screen with keyboard controls, and I won't use the game while it
                exports.
              </label>

              <div className="flex justify-end gap-2 pt-2">
                <button onClick={onCancel} className="btn-ghost">
                  Cancel
                </button>
                <button
                  onClick={handleStart}
                  disabled={!outputDir || !acknowledged || (mode === "since" && !since)}
                  className="btn-primary disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  {dryRun ? "List Replays" : "Start Export"}
                </button>
              </div>
            </>
          )}

          {phase !== "setup" && (
            <>
              {phase === "running" && (
                <div className="space-y-1">
                  <div className="text-accent-purple text-sm animate-pulse">
                    {stopping === "soft"
                      ? "Stopping after the current replay..."
                      : stopping === "hard"
                        ? "Stopping..."
                        : "Exporting - don't use the game window"}
                  </div>
                  <div className="text-[11px] text-text-muted truncate">{status}</div>
                  {captureInfo && <div className="text-[10px] text-text-muted">{captureInfo}</div>}
                </div>
              )}

              {phase === "analyzing" && (
                <div className="text-accent-green text-sm animate-pulse">{analyzeProgress}</div>
              )}

              {phase === "done" && summary && (
                <div className="text-xs text-text-secondary">
                  <span className="text-text-primary font-medium">
                    {summary.exported.length} exported
                    {summary.skipped.length > 0 && `, ${summary.skipped.length} already there`}
                  </span>{" "}
                  ({summary.reason}){summary.manifest && <div className="text-[10px] text-text-muted truncate">{summary.manifest}</div>}
                  {analyzeProgress && <div className="mt-1">{analyzeProgress}</div>}
                </div>
              )}

              {phase === "error" && (
                <div className="space-y-2">
                  <div className="text-p1 text-sm">Export stopped</div>
                  <div className="bg-surface-3 rounded-lg p-3 text-xs text-text-muted overflow-auto max-h-40 font-mono whitespace-pre-wrap">
                    {error}
                  </div>
                </div>
              )}

              {warnings.length > 0 && (
                <div className="bg-accent-gold/10 border border-accent-gold/30 rounded-lg p-2 text-[11px] text-text-secondary space-y-1">
                  {warnings.map((w) => (
                    <div key={w}>{w}</div>
                  ))}
                </div>
              )}

              {analyzeErrors.length > 0 && (
                <div className="bg-p1/10 border border-p1/30 rounded-lg p-2 text-[11px] text-text-secondary space-y-1">
                  {analyzeErrors.map((w) => (
                    <div key={w}>{w}</div>
                  ))}
                </div>
              )}

              <div>
                <div className="text-xs text-text-muted mb-1.5">
                  Replays {total !== null && `(${total} saved in the game)`}
                  {recordedCount > 0 && ` - ${recordedCount} recorded`}
                </div>
                <div className="max-h-72 overflow-y-auto space-y-1">
                  {rows.length === 0 && <div className="text-[11px] text-text-muted">Getting to the replay list...</div>}
                  {rows.map((r) => (
                    <div key={r.index} className="flex items-center gap-3 px-2.5 py-1.5 rounded-lg bg-surface-3 border border-surface-4/30">
                      <div className="text-[10px] text-text-muted w-6 text-right">{r.index + 1}</div>
                      <div className="flex-1 min-w-0">
                        <div className="text-xs text-text-primary truncate">
                          {r.p1 || r.p2 ? `${r.p1 || "?"} vs ${r.p2 || "?"}` : r.file}
                        </div>
                        <div className="text-[10px] text-text-muted truncate">
                          {r.date ? `${r.date} ${r.time ?? ""}` : "date not read"}
                          {r.duration !== undefined && ` | ${Math.round(r.duration)} s`}
                        </div>
                      </div>
                      <div className={`text-[10px] shrink-0 ${STATUS_CLASS[r.status]}`}>{STATUS_LABEL[r.status]}</div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="flex justify-end gap-2 pt-2">
                {phase === "running" && (
                  <button
                    onClick={handleStop}
                    disabled={stopping === "hard"}
                    className="btn-ghost disabled:opacity-30 disabled:cursor-not-allowed"
                    title={
                      stopping === "none"
                        ? "Finish the replay being recorded, return to the list, then stop"
                        : "Stop recording now (the file is finalised and marked partial)"
                    }
                  >
                    {stopping === "none" ? "Stop after this replay" : "Stop now"}
                  </button>
                )}
                {(phase === "done" || phase === "error") && (
                  <>
                    <button onClick={onCancel} className="btn-ghost">
                      Close
                    </button>
                    <button onClick={() => setPhase("setup")} className="btn-ghost">
                      Export more
                    </button>
                    {dbAfterErrors && (
                      <button onClick={() => onComplete(dbAfterErrors)} className="btn-ghost">
                        Open results
                      </button>
                    )}
                    {exportedFiles.length > 0 && !dbAfterErrors && (
                      <button onClick={handleAnalyze} className="btn-primary">
                        Analyze {exportedFiles.length} video{exportedFiles.length !== 1 ? "s" : ""}
                      </button>
                    )}
                  </>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
    </div>
  );
}
