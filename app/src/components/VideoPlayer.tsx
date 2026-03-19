import { useRef, useEffect, useState, useCallback } from "react";
import type { RoundResult, DamageEvent, Match, Note, Drawing } from "../types";
import { DrawingOverlay } from "./DrawingOverlay";
import { DrawingToolbar } from "./DrawingToolbar";

interface Props {
  src: string | null;
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
}

function formatTime(s: number): string {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

export function VideoPlayer({
  src,
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
}: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const progressRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const videoContainerRef = useRef<HTMLDivElement>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(durationMs / 1000);
  const [playing, setPlaying] = useState(false);
  const [volume, setVolume] = useState(0.5);
  const [playbackRate, setPlaybackRate] = useState(1);

  // Drawing state
  const [drawingMode, setDrawingMode] = useState(false);
  const [activeTool, setActiveTool] = useState<"pen" | "eraser">("pen");
  const [penColor, setPenColor] = useState("#ff3333");
  const [penSize, setPenSize] = useState(8);

  // Seek when seekToMs changes
  useEffect(() => {
    if (seekToMs !== null && videoRef.current) {
      videoRef.current.currentTime = seekToMs / 1000;
      onSeeked();
    }
  }, [seekToMs, onSeeked]);

  // Time update
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const handler = () => {
      setCurrentTime(video.currentTime);
      onTimeUpdate?.(video.currentTime * 1000);
    };
    video.addEventListener("timeupdate", handler);
    return () => video.removeEventListener("timeupdate", handler);
  }, [src, onTimeUpdate]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const handler = () => setDuration(video.duration || durationMs / 1000);
    video.addEventListener("loadedmetadata", handler);
    return () => video.removeEventListener("loadedmetadata", handler);
  }, [src, durationMs]);

  // Volume + playback rate
  useEffect(() => {
    if (videoRef.current) videoRef.current.volume = volume;
  }, [volume]);

  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = playbackRate;
  }, [playbackRate]);

  const togglePlay = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    if (drawingMode) return; // don't toggle play in drawing mode
    if (v.paused) {
      v.play();
      setPlaying(true);
    } else {
      v.pause();
      setPlaying(false);
    }
  }, [drawingMode]);

  const toggleDrawingMode = useCallback(() => {
    setDrawingMode((prev) => {
      if (!prev) {
        // Entering drawing mode — auto-pause
        const v = videoRef.current;
        if (v && !v.paused) {
          v.pause();
          setPlaying(false);
        }
      }
      return !prev;
    });
  }, []);

  const handleClearDrawing = useCallback(() => {
    // Find the drawing at current timestamp and delete it
    const currentMs = (videoRef.current?.currentTime ?? 0) * 1000;
    const existing = drawings.find((d) => Math.abs(d.timestamp_ms - currentMs) < 50);
    if (existing && onDeleteDrawing) {
      onDeleteDrawing(existing.drawing_id);
    }
    // Also save empty to clear active strokes (handled by DrawingOverlay exiting)
    if (onSaveDrawing) {
      onSaveDrawing(currentMs, "[]");
    }
    // Toggle off and back on to reset the overlay's active strokes
    setDrawingMode(false);
    setTimeout(() => setDrawingMode(true), 0);
  }, [drawings, onDeleteDrawing, onSaveDrawing]);

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
      const video = videoRef.current;
      if (!bar || !video) return;
      const rect = bar.getBoundingClientRect();
      const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      video.currentTime = zoomStartS + pct * zoomDuration;
    },
    [zoomStartS, zoomDuration]
  );

  const skipSeconds = useCallback((delta: number) => {
    const v = videoRef.current;
    if (v) v.currentTime = Math.max(0, Math.min(v.duration, v.currentTime + delta));
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Don't capture when typing in inputs
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
          if (videoRef.current && videoRef.current.paused) {
            videoRef.current.currentTime -= 1 / 30; // frame back
          }
          break;
        case ".":
          if (videoRef.current && videoRef.current.paused) {
            videoRef.current.currentTime += 1 / 30; // frame forward
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
  }, [togglePlay, skipSeconds, toggleDrawingMode, drawingMode, onSaveDrawing]);

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

  return (
    <div ref={containerRef} className="bg-surface-2 rounded-lg overflow-hidden border border-surface-4/50">
      {/* Video */}
      <div ref={videoContainerRef} className="relative bg-black aspect-video">
        {src ? (
          <>
            <video
              ref={videoRef}
              src={src}
              className="w-full h-full object-contain"
              onClick={drawingMode ? undefined : togglePlay}
              onPlay={() => setPlaying(true)}
              onPause={() => setPlaying(false)}
            />
            {/* Drawing overlay (always rendered for playback display; captures input only in drawingMode) */}
            {onSaveDrawing && (
              <DrawingOverlay
                videoRef={videoRef}
                drawings={drawings}
                drawingMode={drawingMode}
                activeTool={activeTool}
                penColor={penColor}
                penSize={penSize}
                onSaveDrawing={onSaveDrawing}
                containerRef={videoContainerRef}
              />
            )}
            {/* Drawing toolbar */}
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
            {/* Overlay: current round info */}
            {currentRound && !drawingMode && (
              <div className="absolute top-2 left-2 bg-black/60 backdrop-blur-sm rounded px-2 py-1 text-[10px] text-text-secondary pointer-events-none">
                Round {currentRound.round_index + 1}
                {currentRound.is_comeback && (
                  <span className="ml-1.5 text-accent-gold">COMEBACK</span>
                )}
              </div>
            )}
            {/* Drawing mode indicator */}
            {drawingMode && (
              <div className="absolute bottom-2 left-2 bg-accent-gold/80 rounded px-2 py-1 text-[10px] text-black font-medium pointer-events-none">
                Drawing Mode
              </div>
            )}
            {/* Play overlay when paused */}
            {!playing && !drawingMode && (
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <div className="w-12 h-12 bg-black/40 rounded-full flex items-center justify-center">
                  <span className="text-white/80 text-lg ml-0.5">{"\u25B6"}</span>
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center text-text-muted gap-2">
            <div className="text-2xl opacity-30">{"\uD83C\uDFAC"}</div>
            <div className="text-sm">Video file not found</div>
            <div className="text-[10px] text-text-muted/60">
              Statistics are still available below
            </div>
            {onLocateVideo && (
              <button
                onClick={onLocateVideo}
                className="btn-primary !py-1.5 !px-3 !text-xs mt-1"
              >
                Locate File
              </button>
            )}
          </div>
        )}
      </div>

      {/* Custom progress bar with round markers */}
      <div
        ref={progressRef}
        className="relative h-6 bg-surface-3 cursor-pointer group"
        onClick={handleProgressClick}
      >
        {/* Filled progress */}
        <div
          className="absolute top-0 left-0 h-full bg-accent-purple/40 transition-[width] duration-75"
          style={{ width: `${Math.max(0, Math.min(100, progress))}%` }}
        />

        {/* Round region shading */}
        {visibleRounds.map((r, i) => {
          const start = toBarPct(r.round_start_ms / 1000);
          const end = toBarPct(r.round_end_ms / 1000);
          const isP1Win = r.winner === "P1";
          return (
            <div
              key={`round-bg-${i}`}
              className={`absolute top-0 bottom-0 ${
                isP1Win ? "bg-p1/8" : "bg-p2/8"
              }`}
              style={{ left: `${start}%`, width: `${end - start}%` }}
            />
          );
        })}

        {/* Round boundary markers */}
        {visibleRounds.map((r, i) => {
          const pos = toBarPct(r.round_end_ms / 1000);
          return (
            <div
              key={`round-${i}`}
              className="absolute top-0 bottom-0 w-px bg-text-muted/40"
              style={{ left: `${pos}%` }}
            />
          );
        })}

        {/* Comeback markers */}
        {visibleRounds
          .filter((r) => r.is_comeback)
          .map((r, i) => {
            const pos = toBarPct(r.deficit_timestamp_ms / 1000);
            return (
              <div
                key={`cb-${i}`}
                className="absolute top-0 bottom-0 w-1 bg-accent-gold/60 rounded"
                style={{ left: `${pos}%` }}
                title={`Comeback: ${r.winner} (${(r.max_deficit * 100).toFixed(0)}% deficit)`}
              />
            );
          })}

        {/* High damage markers */}
        {visibleDamage
          .filter((e) => e.damage_pct > 0.30)
          .map((e, i) => {
            const pos = toBarPct(e.timestamp_ms / 1000);
            const color = e.target_side === 1 ? "bg-p1/40" : "bg-p2/40";
            return (
              <div
                key={`dmg-${i}`}
                className={`absolute top-0 bottom-0 w-0.5 ${color}`}
                style={{ left: `${pos}%` }}
              />
            );
          })}

        {/* Note markers */}
        {notes?.map((note, i) => {
          const pos = toBarPct(note.timestamp_ms / 1000);
          return (
            <div
              key={`note-${i}`}
              className="absolute top-0 bottom-0 w-1 bg-accent-green/50 rounded"
              style={{ left: `${pos}%` }}
              title={note.text}
            />
          );
        })}

        {/* Drawing markers */}
        {drawings.map((d, i) => {
          const pos = toBarPct(d.timestamp_ms / 1000);
          return (
            <div
              key={`draw-${i}`}
              className="absolute top-0 bottom-0 w-1 bg-accent-gold/50 rounded"
              style={{ left: `${pos}%` }}
              title="Drawing"
            />
          );
        })}

        {/* Playhead */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-text-primary"
          style={{ left: `${Math.max(0, Math.min(100, progress))}%` }}
        />
      </div>

      {/* Controls */}
      <div className="flex items-center gap-2 px-3 py-1.5 bg-surface-2">
        <button
          onClick={() => skipSeconds(-5)}
          className="btn-ghost !px-1.5 !py-1 text-xs"
          title="Rewind 5s (Left arrow)"
        >
          -5s
        </button>
        <button
          onClick={togglePlay}
          className="btn-ghost !px-2.5 !py-1 text-xs font-medium min-w-[40px]"
          title="Play/Pause (Space)"
        >
          {playing ? "||" : "\u25B6"}
        </button>
        <button
          onClick={() => skipSeconds(5)}
          className="btn-ghost !px-1.5 !py-1 text-xs"
          title="Forward 5s (Right arrow)"
        >
          +5s
        </button>

        <span className="text-[11px] font-mono text-text-secondary ml-1">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>

        {selectedMatch && (
          <button
            onClick={onClearSelection}
            className="text-[10px] text-accent-purple hover:text-accent-purple/80 transition-colors cursor-pointer px-1.5 py-0.5 rounded bg-accent-purple/10"
            title="Show full timeline"
          >
            G{selectedMatch.match_index + 1} &times;
          </button>
        )}

        {/* Draw toggle */}
        {onSaveDrawing && (
          <button
            onClick={toggleDrawingMode}
            className={`text-[10px] px-1.5 py-0.5 rounded cursor-pointer transition-colors ${
              drawingMode
                ? "bg-accent-gold/20 text-accent-gold"
                : "text-text-muted hover:text-text-secondary"
            }`}
            title="Toggle drawing mode (D)"
          >
            Draw
          </button>
        )}

        {/* Playback speed */}
        <div className="ml-auto flex items-center gap-3">
          <div className="flex items-center gap-1">
            {[0.5, 1, 1.5, 2].map((rate) => (
              <button
                key={rate}
                onClick={() => setPlaybackRate(rate)}
                className={`text-[10px] px-1.5 py-0.5 rounded cursor-pointer transition-colors ${
                  playbackRate === rate
                    ? "bg-accent-purple/20 text-accent-purple"
                    : "text-text-muted hover:text-text-secondary"
                }`}
              >
                {rate}x
              </button>
            ))}
          </div>

          <div className="flex items-center gap-1.5">
            <button
              onClick={() => setVolume((v) => (v > 0 ? 0 : 0.5))}
              className="btn-ghost !px-1 !py-0.5 text-[10px]"
              title="Toggle mute (M)"
            >
              {volume === 0 ? "\uD83D\uDD07" : "\uD83D\uDD0A"}
            </button>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={volume}
              onChange={(e) => setVolume(parseFloat(e.target.value))}
              className="w-14 h-1 accent-accent-purple"
            />
          </div>
        </div>
      </div>

      {/* Keyboard shortcuts hint */}
      <div className="px-3 py-1 bg-surface-3/30 text-[9px] text-text-muted flex gap-4 border-t border-surface-4/20">
        <span>Space: play/pause</span>
        <span>{"<-/->"}: seek 5s</span>
        <span>Shift+{"<-/->"}: seek 10s</span>
        <span>,/.: frame step</span>
        <span>M: mute</span>
        {onSaveDrawing && <span>D: draw</span>}
      </div>
    </div>
  );
}
