import { useRef, useEffect, useState, useCallback } from "react";
import type { RoundResult, DamageEvent, Match, Note, Drawing } from "../types";
import type { VideoSourceType } from "../lib/videoSourceDetect";
import { DrawingOverlay } from "./DrawingOverlay";
import { DrawingToolbar } from "./DrawingToolbar";
import { loadYouTubeApi } from "../lib/youtubeApi";
import { loadTwitchApi } from "../lib/twitchApi";
import {
  Play, Pause, RotateCcw, RotateCw, ChevronLeft, ChevronRight, Volume2, VolumeX,
  Pencil, Keyboard, VideoOff, FolderOpen,
} from "lucide-react";
import { Button, IconButton } from "./ui";

interface Props {
  src: string | null;
  srcType?: VideoSourceType;
  embedId?: string;
  seekToMs: number | null;
  onSeeked: () => void;
  durationMs: number;
  rounds: RoundResult[];
  damageEvents: DamageEvent[];
  selectedMatch: Match | null;
  onClearSelection: () => void;
  onLocateVideo?: () => void;
  onTimeUpdate?: (ms: number) => void;
  notes?: Note[];
  drawings?: Drawing[];
  onSaveDrawing?: (timestampMs: number, strokesJson: string) => void;
  onDeleteDrawing?: (drawingId: number) => void;
  /** Fill the parent's height: the video is fitted inside it instead of taking the full width. */
  fill?: boolean;
  /** The built-in progress bar; off when a ReviewTimeline sits under the player. */
  showProgress?: boolean;
}

function formatTime(s: number): string {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

export function VideoPlayer({
  src,
  srcType = "file",
  embedId,
  seekToMs,
  onSeeked,
  durationMs,
  rounds,
  damageEvents,
  selectedMatch,
  onClearSelection,
  onLocateVideo,
  onTimeUpdate,
  notes,
  drawings = [],
  onSaveDrawing,
  onDeleteDrawing,
  fill = false,
  showProgress = true,
}: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const embedRef = useRef<HTMLDivElement>(null);
  const embedPlayerRef = useRef<any>(null);
  const progressRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const videoContainerRef = useRef<HTMLDivElement>(null);
  const pendingSeekRef = useRef<number | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(durationMs / 1000);
  const [playing, setPlaying] = useState(false);
  const [volume, setVolume] = useState(0.5);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [embedReady, setEmbedReady] = useState(false);

  const isEmbed = srcType === "youtube" || srcType === "twitch";

  // ── Dispatch helpers ─────────────────────────────────────────────────

  const doPlay = useCallback(() => {
    if (!isEmbed) {
      videoRef.current?.play();
    } else if (srcType === "youtube") {
      embedPlayerRef.current?.playVideo();
    } else {
      embedPlayerRef.current?.play();
    }
  }, [isEmbed, srcType]);

  const doPause = useCallback(() => {
    if (!isEmbed) {
      videoRef.current?.pause();
    } else if (srcType === "youtube") {
      embedPlayerRef.current?.pauseVideo();
    } else {
      embedPlayerRef.current?.pause();
    }
  }, [isEmbed, srcType]);

  const doSeek = useCallback((seconds: number) => {
    if (!isEmbed) {
      if (videoRef.current) videoRef.current.currentTime = seconds;
    } else if (srcType === "youtube") {
      embedPlayerRef.current?.seekTo(seconds, true);
    } else {
      embedPlayerRef.current?.seek(seconds);
    }
  }, [isEmbed, srcType]);

  const doSetVolume = useCallback((v: number) => {
    if (!isEmbed) {
      if (videoRef.current) videoRef.current.volume = v;
    } else if (srcType === "youtube") {
      embedPlayerRef.current?.setVolume(v * 100);
    } else {
      embedPlayerRef.current?.setVolume(v);
    }
  }, [isEmbed, srcType]);

  const doSetPlaybackRate = useCallback((rate: number) => {
    if (!isEmbed) {
      if (videoRef.current) videoRef.current.playbackRate = rate;
    } else if (srcType === "youtube") {
      embedPlayerRef.current?.setPlaybackRate(rate);
    }
    // Twitch doesn't support playback rate
  }, [isEmbed, srcType]);

  const doGetCurrentTime = useCallback((): number => {
    if (!isEmbed) {
      return videoRef.current?.currentTime ?? 0;
    }
    return embedPlayerRef.current?.getCurrentTime?.() ?? 0;
  }, [isEmbed]);

  const doGetDuration = useCallback((): number => {
    if (!isEmbed) {
      return videoRef.current?.duration ?? 0;
    }
    return embedPlayerRef.current?.getDuration?.() ?? 0;
  }, [isEmbed]);

  const doIsPaused = useCallback((): boolean => {
    if (!isEmbed) {
      return videoRef.current?.paused ?? true;
    }
    if (srcType === "youtube") {
      const state = embedPlayerRef.current?.getPlayerState?.();
      return state !== 1; // 1 = PLAYING
    }
    return embedPlayerRef.current?.isPaused?.() ?? true;
  }, [isEmbed, srcType]);

  // ── YouTube embed setup ──────────────────────────────────────────────

  useEffect(() => {
    if (srcType !== "youtube" || !embedId) return;
    setEmbedReady(false);

    let destroyed = false;
    const el = embedRef.current;
    if (!el) return;

    // YouTube API replaces the target element with an iframe
    const target = document.createElement("div");
    target.style.width = "100%";
    target.style.height = "100%";
    el.appendChild(target);

    loadYouTubeApi().then(() => {
      if (destroyed) return;
      const player = new window.YT.Player(target, {
        width: "100%",
        height: "100%",
        videoId: embedId,
        playerVars: {
          autoplay: 0,
          controls: 0,
          modestbranding: 1,
          rel: 0,
          iv_load_policy: 3,
          disablekb: 1,
          fs: 0,
          playsinline: 1,
        },
        events: {
          onReady: () => {
            if (destroyed) return;
            embedPlayerRef.current = player;
            setDuration(player.getDuration());
            player.setVolume(volume * 100);
            setEmbedReady(true);
          },
          onStateChange: (e: any) => {
            if (destroyed) return;
            setPlaying(e.data === 1); // 1 = PLAYING
          },
        },
      });
    });

    return () => {
      destroyed = true;
      embedPlayerRef.current?.destroy?.();
      embedPlayerRef.current = null;
      // Clean up added child
      while (el.firstChild) el.removeChild(el.firstChild);
    };
  }, [srcType, embedId]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Twitch embed setup ───────────────────────────────────────────────

  useEffect(() => {
    if (srcType !== "twitch" || !embedId) return;
    setEmbedReady(false);

    let destroyed = false;
    const el = embedRef.current;
    if (!el) return;

    const target = document.createElement("div");
    target.style.width = "100%";
    target.style.height = "100%";
    el.appendChild(target);

    loadTwitchApi().then(() => {
      if (destroyed) return;
      const player = new window.Twitch.Player(target, {
        video: embedId,
        parent: [location.hostname],
        controls: false,
        autoplay: false,
        muted: false,
        width: "100%",
        height: "100%",
      });

      player.addEventListener(window.Twitch.Player.READY, () => {
        if (destroyed) return;
        embedPlayerRef.current = player;
        player.setVolume(volume);
        setEmbedReady(true);
        // Duration becomes available after a short delay
        setTimeout(() => {
          if (!destroyed) setDuration(player.getDuration());
        }, 1000);
      });

      player.addEventListener(window.Twitch.Player.PLAY, () => {
        if (!destroyed) setPlaying(true);
      });
      player.addEventListener(window.Twitch.Player.PAUSE, () => {
        if (!destroyed) setPlaying(false);
      });
    });

    return () => {
      destroyed = true;
      embedPlayerRef.current = null;
      while (el.firstChild) el.removeChild(el.firstChild);
    };
  }, [srcType, embedId]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Time polling for embeds ──────────────────────────────────────────

  useEffect(() => {
    if (!isEmbed || !embedReady) return;
    const interval = setInterval(() => {
      const t = doGetCurrentTime();
      setCurrentTime(t);
      onTimeUpdate?.(t * 1000);
      // Also refresh duration if it was 0
      if (duration === 0) {
        const d = doGetDuration();
        if (d > 0) setDuration(d);
      }
    }, 100);
    return () => clearInterval(interval);
  }, [isEmbed, embedReady, doGetCurrentTime, doGetDuration, onTimeUpdate, duration]);

  // ── Local video event listeners ──────────────────────────────────────

  useEffect(() => {
    if (isEmbed) return;
    const video = videoRef.current;
    if (seekToMs === null || !video) return;
    // a seek before the metadata has loaded is dropped by the browser: keep it for loadedmetadata
    if (video.readyState >= 1) video.currentTime = seekToMs / 1000;
    else pendingSeekRef.current = seekToMs / 1000;
    setCurrentTime(seekToMs / 1000);
    onTimeUpdate?.(seekToMs);
    onSeeked();
  }, [seekToMs, onSeeked, onTimeUpdate, isEmbed]);

  // Seek for embeds
  useEffect(() => {
    if (!isEmbed || !embedReady) return;
    if (seekToMs !== null) {
      doSeek(seekToMs / 1000);
      onSeeked();
    }
  }, [seekToMs, onSeeked, isEmbed, embedReady, doSeek]);

  useEffect(() => {
    if (isEmbed) return;
    const video = videoRef.current;
    if (!video) return;
    const handler = () => {
      setCurrentTime(video.currentTime);
      onTimeUpdate?.(video.currentTime * 1000);
    };
    video.addEventListener("timeupdate", handler);
    return () => video.removeEventListener("timeupdate", handler);
  }, [src, onTimeUpdate, isEmbed]);

  // timeupdate fires ~4x a second: while playing, also report every ~66 ms so the
  // timeline's playhead moves smoothly
  useEffect(() => {
    if (isEmbed || !playing) return;
    let raf = 0;
    let last = 0;
    const tick = (now: number) => {
      const video = videoRef.current;
      if (video && now - last > 66) {
        last = now;
        setCurrentTime(video.currentTime);
        onTimeUpdate?.(video.currentTime * 1000);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, isEmbed, onTimeUpdate]);

  useEffect(() => {
    if (isEmbed) return;
    const video = videoRef.current;
    if (!video) return;
    const handler = () => {
      setDuration(video.duration || durationMs / 1000);
      if (pendingSeekRef.current !== null) {
        video.currentTime = pendingSeekRef.current;
        pendingSeekRef.current = null;
      }
    };
    video.addEventListener("loadedmetadata", handler);
    return () => video.removeEventListener("loadedmetadata", handler);
  }, [src, durationMs, isEmbed]);

  // Volume + playback rate sync
  useEffect(() => { doSetVolume(volume); }, [volume, doSetVolume]);
  useEffect(() => { doSetPlaybackRate(playbackRate); }, [playbackRate, doSetPlaybackRate]);

  // Drawing state
  const [drawingMode, setDrawingMode] = useState(false);
  const [activeTool, setActiveTool] = useState<"pen" | "eraser">("pen");
  const [penColor, setPenColor] = useState("#ff3333");
  const [penSize, setPenSize] = useState(8);

  const togglePlay = useCallback(() => {
    if (drawingMode) return;
    if (doIsPaused()) {
      doPlay();
      setPlaying(true);
    } else {
      doPause();
      setPlaying(false);
    }
  }, [drawingMode, doIsPaused, doPlay, doPause]);

  const toggleDrawingMode = useCallback(() => {
    setDrawingMode((prev) => {
      if (!prev) {
        // Entering drawing mode — auto-pause
        if (!doIsPaused()) {
          doPause();
          setPlaying(false);
        }
      }
      return !prev;
    });
  }, [doIsPaused, doPause]);

  const handleClearDrawing = useCallback(() => {
    const currentMs = doGetCurrentTime() * 1000;
    const existing = drawings.find((d) => Math.abs(d.timestamp_ms - currentMs) < 50);
    if (existing && onDeleteDrawing) {
      onDeleteDrawing(existing.drawing_id);
    }
    if (onSaveDrawing) {
      onSaveDrawing(currentMs, "[]");
    }
    setDrawingMode(false);
    setTimeout(() => setDrawingMode(true), 0);
  }, [drawings, onDeleteDrawing, onSaveDrawing, doGetCurrentTime]);

  const handleDrawingDone = useCallback(() => {
    setDrawingMode(false);
  }, []);

  // Zoom range for progress bar
  const zoomStartS = selectedMatch ? selectedMatch.start_ms / 1000 : 0;
  const zoomEndS = selectedMatch ? selectedMatch.end_ms / 1000 : duration;
  const zoomDuration = zoomEndS - zoomStartS;

  const handleProgressClick = useCallback(
    (e: React.MouseEvent) => {
      const bar = progressRef.current;
      if (!bar) return;
      const rect = bar.getBoundingClientRect();
      const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      doSeek(zoomStartS + pct * zoomDuration);
    },
    [zoomStartS, zoomDuration, doSeek]
  );

  const skipSeconds = useCallback((delta: number) => {
    const t = doGetCurrentTime();
    const d = doGetDuration() || duration;
    doSeek(Math.max(0, Math.min(d, t + delta)));
  }, [doGetCurrentTime, doGetDuration, doSeek, duration]);

  const stepFrame = useCallback((dir: number) => {
    const v = videoRef.current;
    if (isEmbed || !v) return;
    if (!v.paused) {
      v.pause();
      setPlaying(false);
    }
    v.currentTime = Math.max(0, v.currentTime + dir / 30);
  }, [isEmbed]);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      switch (e.key) {
        case " ":
          e.preventDefault();
          togglePlay();
          break;
        case "ArrowLeft":
          e.preventDefault();
          skipSeconds(e.shiftKey ? -10 : -5);
          break;
        case "ArrowRight":
          e.preventDefault();
          skipSeconds(e.shiftKey ? 10 : 5);
          break;
        case "m":
          setVolume((v) => (v > 0 ? 0 : 0.5));
          break;
        case ",":
          if (!isEmbed && videoRef.current && videoRef.current.paused) {
            videoRef.current.currentTime -= 1 / 30;
          }
          break;
        case ".":
          if (!isEmbed && videoRef.current && videoRef.current.paused) {
            videoRef.current.currentTime += 1 / 30;
          }
          break;
        case "d":
          if (onSaveDrawing) toggleDrawingMode();
          break;
        case "Escape":
          if (drawingMode) setDrawingMode(false);
          break;
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [togglePlay, skipSeconds, toggleDrawingMode, drawingMode, onSaveDrawing, isEmbed]);

  // Map a time (seconds) to a 0-100% position within the zoom range
  const toBarPct = (s: number) =>
    zoomDuration > 0
      ? ((s - zoomStartS) / zoomDuration) * 100
      : 0;

  const progress = toBarPct(currentTime);

  // Filter rounds/damage to zoom range
  const visibleRounds = selectedMatch
    ? rounds.filter(
        (r) =>
          r.round_start_ms >= selectedMatch.start_ms &&
          r.round_end_ms <= selectedMatch.end_ms
      )
    : rounds;

  const visibleDamage = selectedMatch
    ? damageEvents.filter(
        (e) =>
          e.timestamp_ms >= selectedMatch.start_ms &&
          e.timestamp_ms <= selectedMatch.end_ms
      )
    : damageEvents;

  // Find current round
  const currentMs = currentTime * 1000;
  const currentRound = rounds.find(
    (r) => currentMs >= r.round_start_ms && currentMs <= r.round_end_ms
  );

  const hasVideo = isEmbed ? !!embedId : !!src;
  const rates = [0.25, 0.5, 1, 1.5, 2];

  return (
    <div ref={containerRef} className={`flex flex-col min-h-0 ${fill ? "h-full" : ""}`}>
      {/* Video: fitted inside the space it gets (16:9, never wider or taller) */}
      <div
        className={`relative bg-black rounded-lg overflow-hidden flex items-center justify-center ${fill ? "flex-1 min-h-[120px]" : "aspect-video"}`}
        style={fill ? { containerType: "size" } : undefined}
      >
        <div
          ref={videoContainerRef}
          className="relative aspect-video max-w-full max-h-full"
          style={fill ? { width: "min(100cqw, 100cqh * 16 / 9)" } : { width: "100%" }}
        >
        {hasVideo ? (
          <>
            {isEmbed ? (
              <div
                ref={embedRef}
                className="w-full h-full [&_iframe]:!w-full [&_iframe]:!h-full"
                onClick={drawingMode ? undefined : togglePlay}
              />
            ) : (
              <video
                ref={videoRef}
                src={src!}
                className="w-full h-full object-contain"
                onClick={drawingMode ? undefined : togglePlay}
                onPlay={() => setPlaying(true)}
                onPause={() => setPlaying(false)}
              />
            )}
            {/* Drawing overlay (always rendered for playback display; captures input only in drawingMode) */}
            {onSaveDrawing && (
              <DrawingOverlay
                currentTimeMs={currentTime * 1000}
                drawings={drawings}
                drawingMode={drawingMode}
                activeTool={activeTool}
                penColor={penColor}
                penSize={penSize}
                onSaveDrawing={onSaveDrawing}
                containerRef={videoContainerRef}
              />
            )}
            {drawingMode && (
              <DrawingToolbar
                activeTool={activeTool}
                penColor={penColor}
                penSize={penSize}
                onToolChange={setActiveTool}
                onColorChange={setPenColor}
                onSizeChange={setPenSize}
                onClear={handleClearDrawing}
                onDone={handleDrawingDone}
              />
            )}
            {currentRound && !drawingMode && (
              <div className="absolute top-2 left-2 bg-black/60 backdrop-blur-sm rounded-md px-2 py-1 text-[11px] font-medium text-text-primary pointer-events-none">
                Round {currentRound.round_index + 1}
                {currentRound.is_comeback && <span className="ml-1.5 text-accent-gold">Comeback</span>}
              </div>
            )}
            {drawingMode && (
              <div className="absolute bottom-2 left-2 bg-accent-gold/90 rounded-md px-2 py-1 text-[11px] text-black font-semibold pointer-events-none">
                Drawing — Esc to finish
              </div>
            )}
            {!playing && !drawingMode && (
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <div className="w-14 h-14 bg-black/50 backdrop-blur-sm rounded-full flex items-center justify-center">
                  <Play size={24} className="text-white ml-1" fill="currentColor" />
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-text-muted gap-2 p-4 text-center">
            <VideoOff size={28} className="opacity-50" />
            <div className="text-sm text-text-secondary">Video file not found</div>
            <div className="text-xs">The analysis is still available.</div>
            {onLocateVideo && (
              <Button size="sm" variant="primary" icon={FolderOpen} onClick={onLocateVideo} className="mt-1">
                Locate video
              </Button>
            )}
          </div>
        )}
        </div>
      </div>

      {/* Simple progress bar (when no separate timeline is shown) */}
      {showProgress && (
        <div
          ref={progressRef}
          className="relative h-5 mt-2 bg-surface-3 rounded cursor-pointer shrink-0"
          onClick={handleProgressClick}
        >
          <div
            className="absolute top-0 left-0 h-full bg-accent-purple/30 rounded-l"
            style={{ width: `${Math.max(0, Math.min(100, progress))}%` }}
          />
          {visibleRounds.map((r, i) => {
            const start = toBarPct(r.round_start_ms / 1000);
            const end = toBarPct(r.round_end_ms / 1000);
            return (
              <div
                key={`round-bg-${i}`}
                className={`absolute top-0 bottom-0 border-r border-text-muted/40 ${r.winner === "P1" ? "bg-p1/10" : "bg-p2/10"}`}
                style={{ left: `${start}%`, width: `${end - start}%` }}
              />
            );
          })}
          {visibleDamage
            .filter((e) => e.damage_pct > 0.3)
            .map((e, i) => (
              <div
                key={`dmg-${i}`}
                className={`absolute top-0 bottom-0 w-0.5 ${e.target_side === 1 ? "bg-p1/40" : "bg-p2/40"}`}
                style={{ left: `${toBarPct(e.timestamp_ms / 1000)}%` }}
              />
            ))}
          {notes?.map((note, i) => (
            <div
              key={`note-${i}`}
              className="absolute top-0 bottom-0 w-1 bg-accent-green/60 rounded"
              style={{ left: `${toBarPct(note.timestamp_ms / 1000)}%` }}
              title={note.text}
            />
          ))}
          {drawings.map((d, i) => (
            <div
              key={`draw-${i}`}
              className="absolute top-0 bottom-0 w-1 bg-accent-gold/60 rounded"
              style={{ left: `${toBarPct(d.timestamp_ms / 1000)}%` }}
              title="Drawing"
            />
          ))}
          <div
            className="absolute -top-0.5 -bottom-0.5 w-0.5 bg-text-primary rounded"
            style={{ left: `${Math.max(0, Math.min(100, progress))}%` }}
          />
        </div>
      )}

      {/* Controls */}
      <div className="flex items-center gap-1 pt-2 shrink-0 flex-wrap">
        <IconButton icon={RotateCcw} label="Back 5 s (←, Shift+← 10 s)" onClick={() => skipSeconds(-5)} />
        <IconButton icon={playing ? Pause : Play} label="Play / pause (Space)" onClick={togglePlay} size={36} className="!text-text-primary" />
        <IconButton icon={RotateCw} label="Forward 5 s (→, Shift+→ 10 s)" onClick={() => skipSeconds(5)} />
        {!isEmbed && (
          <>
            <IconButton icon={ChevronLeft} label="Previous frame (,)" onClick={() => stepFrame(-1)} size={28} />
            <IconButton icon={ChevronRight} label="Next frame (.)" onClick={() => stepFrame(1)} size={28} />
          </>
        )}
        <span className="text-xs font-mono tabular-nums text-text-secondary ml-1.5">
          {formatTime(currentTime)} <span className="text-text-muted">/ {formatTime(duration)}</span>
        </span>

        <div className="ml-auto flex items-center gap-1">
          {onSaveDrawing && (
            <Button
              size="sm"
              variant={drawingMode ? "primary" : "ghost"}
              icon={Pencil}
              onClick={toggleDrawingMode}
              title="Draw on the frame (D)"
            >
              <span className="hidden sm:inline">Draw</span>
            </Button>
          )}
          <select
            value={playbackRate}
            onChange={(e) => setPlaybackRate(parseFloat(e.target.value))}
            className="h-7 bg-surface-3 border border-surface-4 rounded-md text-xs text-text-secondary px-1.5 cursor-pointer focus:outline-none"
            title="Playback speed"
            disabled={srcType === "twitch"}
          >
            {rates.map((r) => (
              <option key={r} value={r}>{r}×</option>
            ))}
          </select>
          <IconButton
            icon={volume === 0 ? VolumeX : Volume2}
            label="Mute (M)"
            onClick={() => setVolume((v) => (v > 0 ? 0 : 0.5))}
            size={28}
          />
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={volume}
            onChange={(e) => setVolume(parseFloat(e.target.value))}
            className="w-16 h-1 accent-accent-purple hidden md:block"
            aria-label="Volume"
          />
          <IconButton
            icon={Keyboard}
            label={`Shortcuts: Space play/pause · ←/→ 5 s · Shift+←/→ 10 s${isEmbed ? "" : " · , . frame step"} · M mute${onSaveDrawing ? " · D draw" : ""}`}
            size={28}
            className="hidden md:inline-flex cursor-help"
          />
        </div>
      </div>
    </div>
  );
}
