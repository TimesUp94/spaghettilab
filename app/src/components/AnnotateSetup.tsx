import { useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { detectVideoSource, type VideoSourceType } from "../lib/videoSourceDetect";
import { createQuickSession } from "../api";
import type { SpagzSession } from "../types";

interface Props {
  onSession: (session: SpagzSession, srcType: VideoSourceType, embedId?: string) => void;
  onCancel: () => void;
}

export function AnnotateSetup({ onSession, onCancel }: Props) {
  const [tab, setTab] = useState<"url" | "file">("url");
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const detected = url.trim() ? detectVideoSource(url.trim()) : null;
  const isValidUrl = detected && (detected.type === "youtube" || detected.type === "twitch");

  const handleUrlSubmit = async () => {
    if (!detected || !isValidUrl) return;
    setLoading(true);
    setError(null);
    try {
      const session = await createQuickSession("", url.trim());
      onSession(session, detected.type, detected.id);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  };

  const handleLocalFile = async () => {
    const selected = await open({
      multiple: false,
      filters: [{ name: "Video", extensions: ["mp4", "mkv", "avi", "webm", "mov"] }],
    });
    if (!selected) return;
    setLoading(true);
    setError(null);
    try {
      const session = await createQuickSession(selected as string);
      onSession(session, "file");
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-surface-1 rounded-2xl shadow-2xl max-w-md w-full mx-4 overflow-hidden">
        {/* Header */}
        <div className="px-6 pt-5 pb-3">
          <h2 className="text-lg font-semibold text-text-primary">Annotate Video</h2>
          <p className="text-xs text-text-muted mt-1">
            Add timestamped notes and drawings to any video
          </p>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 px-6 mb-4">
          <button
            onClick={() => setTab("url")}
            className={`px-3 py-1.5 text-xs rounded-lg transition-colors cursor-pointer ${
              tab === "url"
                ? "bg-accent-purple/15 text-accent-purple font-medium"
                : "text-text-muted hover:text-text-secondary"
            }`}
          >
            YouTube / Twitch
          </button>
          <button
            onClick={() => setTab("file")}
            className={`px-3 py-1.5 text-xs rounded-lg transition-colors cursor-pointer ${
              tab === "file"
                ? "bg-accent-purple/15 text-accent-purple font-medium"
                : "text-text-muted hover:text-text-secondary"
            }`}
          >
            Local File
          </button>
        </div>

        {/* Content */}
        <div className="px-6 pb-5">
          {tab === "url" && (
            <div className="space-y-3">
              <div>
                <input
                  type="text"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="Paste YouTube or Twitch VOD URL..."
                  className="w-full px-3 py-2 bg-surface-2 border border-surface-4 rounded-lg text-sm text-text-primary placeholder:text-text-muted/50 focus:outline-none focus:border-accent-purple/50"
                  onKeyDown={(e) => { if (e.key === "Enter" && isValidUrl) handleUrlSubmit(); }}
                  autoFocus
                />
                {url.trim() && (
                  <div className="mt-1.5 text-[10px]">
                    {isValidUrl ? (
                      <span className="text-accent-green">
                        {detected!.type === "youtube" ? "YouTube" : "Twitch"} video detected: {detected!.id}
                      </span>
                    ) : (
                      <span className="text-text-muted">
                        Paste a YouTube or Twitch VOD URL
                      </span>
                    )}
                  </div>
                )}
              </div>

              <button
                onClick={handleUrlSubmit}
                disabled={!isValidUrl || loading}
                className="btn-primary w-full !py-2 disabled:opacity-40"
              >
                {loading ? "Opening..." : "Open"}
              </button>
            </div>
          )}

          {tab === "file" && (
            <div className="space-y-3">
              <p className="text-xs text-text-secondary">
                Browse for a local video file (mp4, mkv, avi, webm, mov)
              </p>
              <button
                onClick={handleLocalFile}
                disabled={loading}
                className="btn-primary w-full !py-2 disabled:opacity-40"
              >
                {loading ? "Opening..." : "Browse File"}
              </button>
            </div>
          )}

          {error && (
            <div className="mt-3 px-3 py-2 bg-red-500/10 border border-red-500/20 rounded-lg text-xs text-red-400">
              {error}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 pb-5 flex justify-end">
          <button
            onClick={onCancel}
            className="btn-ghost text-xs"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
