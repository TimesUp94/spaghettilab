import { useState, useCallback, useEffect } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { listen } from "@tauri-apps/api/event";
import { convertFileSrc } from "@tauri-apps/api/core";
import { extractPreviewFrame, scanVod, cutVodSets, downloadVod } from "../api";
import type { VodRoiConfig, DetectedSetInfo } from "../types";
import { RoiPicker } from "./RoiPicker";

interface Props {
  onComplete: (outputDir: string) => void;
  onCancel: () => void;
}

type SplitStep = "select" | "downloading" | "roi" | "scanning" | "results" | "cutting" | "done" | "error";

function formatTime(secs: number): string {
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = Math.floor(secs % 60);
  return h > 0 ? `${h}:${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`
    : `${m}:${s.toString().padStart(2, "0")}`;
}

export function SplitVodView({ onComplete, onCancel }: Props) {
  const [step, setStep] = useState<SplitStep>("select");
  const [videoPath, setVideoPath] = useState<string | null>(null);
  const [previewSrc, setPreviewSrc] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState("");
  const [sets, setSets] = useState<DetectedSetInfo[]>([]);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [cutPaths, setCutPaths] = useState<string[]>([]);
  const [outputDir, setOutputDir] = useState<string | null>(null);
  const [videoDuration, setVideoDuration] = useState<number>(0);
  const [vodUrl, setVodUrl] = useState("");
  const [downloadProgress, setDownloadProgress] = useState("");

  // Video dimensions (assume 1920x1080 for now — could probe)
  const videoWidth = 1920;
  const videoHeight = 1080;

  // Listen for download progress events
  useEffect(() => {
    if (step !== "downloading") return;
    let cancelled = false;
    const unlistenPromise = listen<string>("vod-download-progress", (event) => {
      if (!cancelled) {
        setDownloadProgress(event.payload);
      }
    });
    return () => {
      cancelled = true;
      unlistenPromise.then((fn) => fn());
    };
  }, [step]);

  const handleDownloadVod = useCallback(async () => {
    const url = vodUrl.trim();
    if (!url) return;
    setStep("downloading");
    setDownloadProgress("Starting download...");
    setError(null);
    try {
      // Download to system temp directory
      const tmpDir = await import("@tauri-apps/api/path").then(m => m.tempDir());
      const filePath = await downloadVod(url, tmpDir);
      setVideoPath(filePath);
      setVodUrl("");
      // Auto-proceed to preview
      const framePath = await extractPreviewFrame(filePath, 30);
      setPreviewSrc(convertFileSrc(framePath));
      setStep("roi");
    } catch (err) {
      setError(String(err));
      setStep("error");
    }
  }, [vodUrl]);

  const handleSelectVideo = useCallback(async () => {
    const file = await open({
      multiple: false,
      filters: [{ name: "Video Files", extensions: ["mp4", "mkv", "avi"] }],
    });
    if (file) {
      setVideoPath(file as string);
    }
  }, []);

  const handleLoadPreview = useCallback(async () => {
    if (!videoPath) return;
    setError(null);
    try {
      // Extract a frame at 30s in (likely gameplay)
      const framePath = await extractPreviewFrame(videoPath, 30);
      setPreviewSrc(convertFileSrc(framePath));
      // Get video duration via ffprobe (invoke through extract at 0 is a workaround;
      // for now estimate from the scan or use a generous default)
      setStep("roi");
    } catch (err) {
      setError(String(err));
    }
  }, [videoPath]);

  const handleSeekPreview = useCallback(async (timestampSecs: number) => {
    if (!videoPath) return;
    try {
      const framePath = await extractPreviewFrame(videoPath, timestampSecs);
      // Append cache-buster to force browser to reload the image
      setPreviewSrc(convertFileSrc(framePath) + "?t=" + Date.now());
    } catch (err) {
      // Silently ignore seek errors (e.g. past end of video)
    }
  }, [videoPath]);

  // Listen for scan progress events
  useEffect(() => {
    if (step !== "scanning") return;
    let cancelled = false;
    const unlistenPromise = listen<string>("vod-scan-progress", (event) => {
      if (!cancelled) {
        const msg = event.payload;
        if (msg.startsWith("PROGRESS:")) {
          setProgress(msg.slice("PROGRESS:".length));
        }
      }
    });
    return () => {
      cancelled = true;
      unlistenPromise.then((fn) => fn());
    };
  }, [step]);

  const handleStartScan = useCallback(async (roiConfig: VodRoiConfig) => {
    if (!videoPath) return;
    setStep("scanning");
    setProgress("0");
    setError(null);
    try {
      const detected = await scanVod(videoPath, roiConfig);
      setSets(detected);
      // Auto-select only sets with detected player names
      setSelected(new Set(detected.filter((s) => s.p1_name && s.p2_name).map((s) => s.index)));
      setStep("results");
    } catch (err) {
      setError(String(err));
      setStep("error");
    }
  }, [videoPath]);

  const handleSelectOutput = useCallback(async () => {
    const dir = await open({ directory: true });
    if (dir) {
      setOutputDir(dir as string);
    }
  }, []);

  const handleCut = useCallback(async () => {
    if (!videoPath || !outputDir) return;
    setStep("cutting");
    setError(null);
    try {
      const tocut = sets
        .filter((s) => selected.has(s.index))
        .map((s) => ({
          index: s.index,
          start_secs: s.start_secs,
          end_secs: s.end_secs,
          p1_name: s.p1_name || null,
          p2_name: s.p2_name || null,
        }));
      const paths = await cutVodSets(videoPath, tocut, outputDir);
      setCutPaths(paths);
      setStep("done");
      onComplete(outputDir);
    } catch (err) {
      setError(String(err));
      setStep("error");
    }
  }, [videoPath, outputDir, sets, selected, onComplete]);

  const toggleSet = (index: number) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  };

  // ROI picker step
  if (step === "roi" && previewSrc) {
    return (
      <RoiPicker
        imageSrc={previewSrc}
        videoWidth={videoWidth}
        videoHeight={videoHeight}
        videoDurationSecs={videoDuration || 7200}
        onSeekPreview={handleSeekPreview}
        onConfirm={handleStartScan}
        onBack={() => setStep("select")}
      />
    );
  }

  return (
    <div className="h-screen flex items-center justify-center bg-surface-0">
      <div className="w-[560px] bg-surface-2 rounded-xl border border-surface-4/50 shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-surface-4/30">
          <h3 className="text-sm font-semibold text-text-primary">
            Split VOD
          </h3>
          <button
            onClick={onCancel}
            className="text-text-muted hover:text-text-primary cursor-pointer"
          >
            x
          </button>
        </div>

        <div className="p-6 space-y-4">
          {step === "downloading" ? (
            (() => {
              // Parse percentage from yt-dlp output like "[download]  45.2% of ~3.21GiB at 5.23MiB/s ETA 06:32"
              const pctMatch = downloadProgress.match(/([\d.]+)%/);
              const pct = pctMatch ? parseFloat(pctMatch[1]) : null;
              // Extract speed and ETA
              const speedMatch = downloadProgress.match(/at\s+([\d.]+\S+\/s)/);
              const etaMatch = downloadProgress.match(/ETA\s+(\S+)/);
              const sizeMatch = downloadProgress.match(/of\s+~?([\d.]+\S+)/);
              return (
                <div className="py-8 space-y-3">
                  <div className="text-accent-purple text-sm font-medium text-center">
                    Downloading VOD...
                  </div>
                  {/* Progress bar */}
                  <div className="w-full bg-surface-3 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full transition-all duration-300 ${pct != null ? "bg-accent-purple" : "bg-accent-purple animate-pulse"}`}
                      style={{ width: pct != null ? `${pct}%` : "50%" }}
                    />
                  </div>
                  {/* Stats row */}
                  <div className="flex justify-between text-xs text-text-muted px-1">
                    <span>{pct != null ? `${pct.toFixed(1)}%` : "Starting..."}</span>
                    {sizeMatch && <span>{sizeMatch[1]}</span>}
                    {speedMatch && <span>{speedMatch[1]}</span>}
                    {etaMatch && <span>ETA {etaMatch[1]}</span>}
                  </div>
                  {/* Raw line for non-download messages (e.g. "Installing yt-dlp...", "[info] ...") */}
                  {!pctMatch && downloadProgress && (
                    <div className="text-text-muted text-[10px] font-mono text-center truncate px-4">
                      {downloadProgress}
                    </div>
                  )}
                </div>
              );
            })()

          ) : step === "scanning" ? (
            <div className="text-center py-8">
              <div className="text-accent-purple text-sm animate-pulse mb-2">
                Scanning VOD for gameplay...
              </div>
              <div className="text-text-muted text-xs mb-4">
                {progress ? `${progress} seconds scanned` : "Starting..."}
              </div>
              <div className="w-full bg-surface-3 rounded-full h-1.5">
                <div className="bg-accent-purple h-1.5 rounded-full animate-pulse w-1/2" />
              </div>
            </div>
          ) : step === "cutting" ? (
            <div className="text-center py-8">
              <div className="text-accent-green text-sm animate-pulse mb-2">
                Cutting sets...
              </div>
              <div className="w-full bg-surface-3 rounded-full h-1.5">
                <div className="bg-accent-green h-1.5 rounded-full animate-pulse w-2/3" />
              </div>
            </div>
          ) : step === "done" ? (
            <div className="text-center py-8 space-y-3">
              <div className="text-accent-green text-sm font-medium">
                {cutPaths.length} sets saved
              </div>
              <div className="text-text-muted text-xs">
                {outputDir}
              </div>
              <button onClick={onCancel} className="btn-primary">
                Done
              </button>
            </div>
          ) : step === "error" ? (
            <div className="space-y-4">
              <div className="text-p1 text-sm text-center">Error</div>
              <div className="bg-surface-3 rounded-lg p-3 text-xs text-text-muted overflow-auto max-h-40 font-mono select-text">
                {error}
              </div>
              <div className="flex justify-center gap-2">
                <button onClick={onCancel} className="btn-ghost">Cancel</button>
                <button
                  onClick={() => error && navigator.clipboard.writeText(error)}
                  className="btn-ghost"
                >
                  Copy Error
                </button>
                <button onClick={() => setStep("select")} className="btn-primary">
                  Try Again
                </button>
              </div>
            </div>
          ) : step === "results" ? (
            <>
              <div className="text-xs text-text-muted mb-2">
                Found {sets.length} sets. Select which ones to cut:
              </div>
              <div className="space-y-1.5 max-h-[300px] overflow-y-auto">
                {sets.map((s) => (
                  <label
                    key={s.index}
                    className={`flex items-center gap-3 p-2.5 rounded-lg cursor-pointer transition-colors ${
                      selected.has(s.index)
                        ? "bg-accent-purple/10 border border-accent-purple/30"
                        : "bg-surface-3 border border-surface-4/30"
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={selected.has(s.index)}
                      onChange={() => toggleSet(s.index)}
                      className="accent-accent-purple"
                    />
                    <div className="flex-1">
                      <div className="text-xs text-text-primary font-medium">
                        {s.p1_name && s.p2_name ? `${s.p1_name} vs ${s.p2_name}` : `Set ${s.index}`}
                      </div>
                      <div className="text-[10px] text-text-muted">
                        {formatTime(s.start_secs)} - {formatTime(s.end_secs)}
                        {" | "}
                        {(s.gameplay_duration_secs / 60).toFixed(1)} min gameplay
                        {" | "}
                        {s.game_count} game{s.game_count !== 1 ? "s" : ""}
                      </div>
                    </div>
                  </label>
                ))}
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
                  <button onClick={handleSelectOutput} className="btn-primary !py-2">
                    Browse
                  </button>
                </div>
              </div>

              <div className="flex justify-end gap-2 pt-2">
                <button onClick={() => setStep("roi")} className="btn-ghost">
                  Back
                </button>
                <button
                  onClick={handleCut}
                  disabled={selected.size === 0 || !outputDir}
                  className="btn-primary disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  Cut {selected.size} Set{selected.size !== 1 ? "s" : ""}
                </button>
              </div>
            </>
          ) : (
            /* select step */
            <>
              {/* Download from URL */}
              <div>
                <label className="text-xs text-text-muted block mb-1.5">
                  Download from Twitch / YouTube
                </label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={vodUrl}
                    onChange={(e) => setVodUrl(e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter") handleDownloadVod(); }}
                    placeholder="https://www.twitch.tv/videos/..."
                    className="flex-1 bg-surface-3 border border-surface-4 rounded-lg px-3 py-2 text-xs text-text-primary placeholder:text-text-muted/50 outline-none focus:border-accent-purple/50"
                  />
                  <button
                    onClick={handleDownloadVod}
                    disabled={!vodUrl.trim()}
                    className="btn-primary !py-2 disabled:opacity-30 disabled:cursor-not-allowed"
                  >
                    Download
                  </button>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <div className="flex-1 h-px bg-surface-4/50" />
                <span className="text-[10px] text-text-muted">or</span>
                <div className="flex-1 h-px bg-surface-4/50" />
              </div>

              {/* Local file */}
              <div>
                <label className="text-xs text-text-muted block mb-1.5">
                  Local File
                </label>
                <div className="flex gap-2">
                  <div className="flex-1 bg-surface-3 border border-surface-4 rounded-lg px-3 py-2 text-xs text-text-secondary truncate">
                    {videoPath || "No file selected"}
                  </div>
                  <button onClick={handleSelectVideo} className="btn-primary !py-2">
                    Browse
                  </button>
                </div>
              </div>

              <div className="text-[10px] text-text-muted">
                Paste a Twitch or YouTube VOD link to download, or select a local file.
                Requires <span className="font-mono text-text-secondary">yt-dlp</span> for downloads.
              </div>

              <div className="flex justify-end gap-2 pt-2">
                <button onClick={onCancel} className="btn-ghost">Cancel</button>
                <button
                  onClick={handleLoadPreview}
                  disabled={!videoPath}
                  className="btn-primary disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  Next
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
