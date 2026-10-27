// Annotate page: pick a YouTube / Twitch VOD or a local video to take notes on (no analysis).
import { useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { ArrowLeft, Link2, FolderOpen, CheckCircle2, Upload } from "lucide-react";
import { detectVideoSource, type VideoSourceType } from "../lib/videoSourceDetect";
import { createQuickSession } from "../api";
import type { SpagzSession } from "../types";
import { Button } from "./ui";

interface Props {
  onSession: (session: SpagzSession, srcType: VideoSourceType, embedId?: string) => void;
  onCancel: () => void;
  /** A video is being dragged over the window (a drop opens it here). */
  dragActive?: boolean;
}

export function AnnotateSetup({ onSession, onCancel, dragActive }: Props) {
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState<"url" | "file" | null>(null);
  const [error, setError] = useState<string | null>(null);

  const detected = url.trim() ? detectVideoSource(url.trim()) : null;
  const isValidUrl = detected && (detected.type === "youtube" || detected.type === "twitch");

  const handleUrlSubmit = async () => {
    if (!detected || !isValidUrl) return;
    setLoading("url");
    setError(null);
    try {
      const session = await createQuickSession("", url.trim());
      onSession(session, detected.type, detected.id);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(null);
    }
  };

  const handleLocalFile = async () => {
    const selected = await open({
      multiple: false,
      filters: [{ name: "Video", extensions: ["mp4", "mkv", "avi", "webm", "mov"] }],
    });
    if (!selected) return;
    setLoading("file");
    setError(null);
    try {
      const session = await createQuickSession(selected as string);
      onSession(session, "file");
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(null);
    }
  };

  return (
    <div className="h-full overflow-y-auto p-4 sm:p-8">
      <div className="w-full max-w-2xl mx-auto">
        <button onClick={onCancel} className="inline-flex items-center gap-1.5 text-sm text-text-secondary hover:text-text-primary mb-4 cursor-pointer">
          <ArrowLeft size={15} /> Library
        </button>
        <h1 className="text-2xl font-semibold tracking-tight">Annotate a video</h1>
        <p className="text-sm text-text-secondary mt-1 mb-6">
          Timestamped notes, drawings and the review checklist on any video, without running the analysis.
        </p>

        <div className="bg-surface-2 rounded-xl border border-surface-4/70 p-5">
          <div className="flex items-center gap-2 text-sm font-medium mb-1">
            <Link2 size={16} className="text-accent-purple" /> YouTube or Twitch VOD
          </div>
          <p className="text-xs text-text-secondary mb-3">The video is embedded; nothing is downloaded.</p>
          <div className="flex flex-col sm:flex-row gap-2">
            <input
              type="text"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="Paste a YouTube or Twitch link"
              className="flex-1 min-w-0 h-9 px-3 bg-surface-1 border border-surface-4 rounded-lg text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent-purple/60"
              onKeyDown={(e) => { if (e.key === "Enter" && isValidUrl) handleUrlSubmit(); }}
              autoFocus
            />
            <Button variant="primary" onClick={handleUrlSubmit} disabled={!isValidUrl} loading={loading === "url"}>
              Open
            </Button>
          </div>
          {url.trim() && (
            <div className="mt-2 text-xs">
              {isValidUrl ? (
                <span className="inline-flex items-center gap-1 text-accent-green">
                  <CheckCircle2 size={13} /> {detected!.type === "youtube" ? "YouTube" : "Twitch"} video {detected!.id}
                </span>
              ) : (
                <span className="text-text-muted">That doesn&apos;t look like a YouTube or Twitch video link.</span>
              )}
            </div>
          )}
        </div>

        <div className="flex items-center gap-3 my-4 text-xs text-text-muted">
          <div className="flex-1 h-px bg-surface-4" /> or <div className="flex-1 h-px bg-surface-4" />
        </div>

        <button
          onClick={handleLocalFile}
          disabled={loading !== null}
          className={`w-full rounded-xl border-2 border-dashed transition-colors cursor-pointer flex flex-col items-center justify-center text-center gap-2 py-8 px-4 disabled:opacity-50 disabled:cursor-not-allowed
            ${dragActive ? "border-accent-purple bg-accent-purple/10" : "border-surface-4 hover:border-accent-purple/50 hover:bg-surface-2"}`}
        >
          <div className={`w-10 h-10 rounded-full flex items-center justify-center ${dragActive ? "bg-accent-purple text-surface-0" : "bg-accent-purple/15 text-accent-purple"}`}>
            {dragActive ? <Upload size={18} /> : <FolderOpen size={18} />}
          </div>
          <div className="text-sm font-semibold text-text-primary">
            {loading === "file" ? "Opening…" : dragActive ? "Drop it to annotate" : "Drop a video file here"}
          </div>
          <div className="text-xs text-text-secondary">
            or <span className="text-accent-purple underline underline-offset-2">browse your files</span> · MP4, MKV, AVI, WebM or MOV
          </div>
        </button>

        {error && (
          <div className="mt-4 px-3 py-2 bg-p1/10 border border-p1/30 rounded-lg text-xs text-p1-light">{error}</div>
        )}
      </div>
    </div>
  );
}
