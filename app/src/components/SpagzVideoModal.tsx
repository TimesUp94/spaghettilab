import { useState, useEffect } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { listen } from "@tauri-apps/api/event";
import { downloadVod } from "../api";

interface Props {
  videoHint: string;
  videoUrl?: string;
  onLocalFile: (path: string) => void;
  onStreamUrl: (url: string) => void;
  onSkip: () => void;
  onCancel: () => void;
}

type Tab = "browse" | "download" | "stream" | "skip";

export function SpagzVideoModal({ videoHint, videoUrl, onLocalFile, onStreamUrl, onSkip, onCancel }: Props) {
  const hintIsUrl = videoHint.startsWith("http");
  const hasVideoUrl = !!videoUrl;
  const [activeTab, setActiveTab] = useState<Tab>(hasVideoUrl ? "download" : "browse");
  const [url, setUrl] = useState(videoUrl || (hintIsUrl ? videoHint : ""));
  const [streamUrl, setStreamUrl] = useState(hintIsUrl ? videoHint : "");
  const [downloading, setDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!downloading) return;
    let cancelled = false;
    const unlistenPromise = listen<string>("vod-download-progress", (event) => {
      if (!cancelled) setDownloadProgress(event.payload);
    });
    return () => {
      cancelled = true;
      unlistenPromise.then((fn) => fn());
    };
  }, [downloading]);

  const handleBrowse = async () => {
    const selected = await open({
      multiple: false,
      filters: [{ name: "Video Files", extensions: ["mp4", "mkv", "avi", "webm"] }],
    });
    if (selected) {
      onLocalFile(selected as string);
    }
  };

  const handleDownload = async () => {
    if (!url.trim()) return;
    setDownloading(true);
    setError(null);
    setDownloadProgress("");
    try {
      const tmpDir = await import("@tauri-apps/api/path").then(m => m.tempDir());
      const filePath = await downloadVod(url.trim(), tmpDir);
      onLocalFile(filePath);
    } catch (err) {
      setError(String(err));
      setDownloading(false);
    }
  };

  const handleStream = () => {
    if (!streamUrl.trim()) return;
    onStreamUrl(streamUrl.trim());
  };

  const tabs: { key: Tab; label: string }[] = [
    { key: "browse", label: "Browse File" },
    { key: "download", label: "Download VOD" },
    { key: "stream", label: "Stream URL" },
    { key: "skip", label: "No Video" },
  ];

  return (
    <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center">
      <div className="bg-surface-1 border border-surface-4/50 rounded-2xl shadow-2xl w-[520px] max-w-[90vw]">
        {/* Header */}
        <div className="px-6 pt-5 pb-3">
          <h2 className="text-lg font-semibold text-text-primary">Choose Video Source</h2>
          <p className="text-xs text-text-muted mt-1">
            This .spagz file contains analysis data without a video. Choose how to provide the video.
          </p>
          {videoHint && (
            <div className="mt-2 px-3 py-1.5 bg-surface-2 rounded-lg text-[11px] text-text-secondary font-mono truncate">
              Original: {videoHint}
            </div>
          )}
          {videoUrl && (
            <div className="mt-1 px-3 py-1.5 bg-surface-2 rounded-lg text-[11px] text-accent-purple font-mono truncate">
              URL: {videoUrl}
            </div>
          )}
        </div>

        {/* Tabs */}
        <div className="flex border-b border-surface-4/50 px-6">
          {tabs.map(({ key, label }) => (
            <button
              key={key}
              onClick={() => setActiveTab(key)}
              className={`px-3 py-2 text-xs font-medium border-b-2 transition-colors cursor-pointer ${
                activeTab === key
                  ? "text-accent-purple border-accent-purple"
                  : "text-text-muted border-transparent hover:text-text-secondary"
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="px-6 py-5 min-h-[140px]">
          {activeTab === "browse" && (
            <div className="flex flex-col items-center gap-3">
              <p className="text-sm text-text-secondary text-center">
                Select the video file on your computer.
              </p>
              <button
                onClick={handleBrowse}
                className="px-6 py-2.5 bg-accent-purple text-white rounded-lg hover:bg-accent-purple/90 transition-colors cursor-pointer text-sm font-medium"
              >
                Browse...
              </button>
            </div>
          )}

          {activeTab === "download" && (
            <div className="flex flex-col gap-3">
              <p className="text-sm text-text-secondary">
                Enter a YouTube or Twitch VOD URL to download.
              </p>
              <input
                type="text"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://youtube.com/watch?v=... or https://twitch.tv/videos/..."
                disabled={downloading}
                className="w-full px-3 py-2 bg-surface-2 border border-surface-4/50 rounded-lg text-sm text-text-primary placeholder:text-text-muted/50 focus:outline-none focus:border-accent-purple/50"
              />
              {downloading ? (
                <div className="flex items-center gap-3">
                  <div className="w-5 h-5 border-2 border-accent-purple/30 border-t-accent-purple rounded-full animate-spin shrink-0" />
                  <span className="text-xs text-text-secondary truncate">{downloadProgress || "Starting download..."}</span>
                </div>
              ) : (
                <button
                  onClick={handleDownload}
                  disabled={!url.trim()}
                  className="px-6 py-2.5 bg-accent-green text-white rounded-lg hover:bg-accent-green/90 transition-colors cursor-pointer text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed self-start"
                >
                  Download
                </button>
              )}
            </div>
          )}

          {activeTab === "stream" && (
            <div className="flex flex-col gap-3">
              <p className="text-sm text-text-secondary">
                Enter a direct video URL (.mp4). YouTube/Twitch URLs require downloading instead.
              </p>
              <input
                type="text"
                value={streamUrl}
                onChange={(e) => setStreamUrl(e.target.value)}
                placeholder="https://example.com/video.mp4"
                className="w-full px-3 py-2 bg-surface-2 border border-surface-4/50 rounded-lg text-sm text-text-primary placeholder:text-text-muted/50 focus:outline-none focus:border-accent-purple/50"
              />
              <button
                onClick={handleStream}
                disabled={!streamUrl.trim()}
                className="px-6 py-2.5 bg-p2 text-white rounded-lg hover:bg-p2/90 transition-colors cursor-pointer text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed self-start"
              >
                Use URL
              </button>
            </div>
          )}

          {activeTab === "skip" && (
            <div className="flex flex-col items-center gap-3">
              <p className="text-sm text-text-secondary text-center">
                Open without video. Stats, notes, and drawings will still be available.
              </p>
              <button
                onClick={onSkip}
                className="px-6 py-2.5 bg-surface-3 text-text-primary rounded-lg hover:bg-surface-4 transition-colors cursor-pointer text-sm font-medium"
              >
                Open Without Video
              </button>
            </div>
          )}

          {error && (
            <div className="mt-3 text-xs text-p1">{error}</div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 pb-5 flex justify-end">
          <button
            onClick={onCancel}
            disabled={downloading}
            className="px-4 py-1.5 text-xs text-text-muted hover:text-text-secondary transition-colors cursor-pointer disabled:opacity-50"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
