import { useRef, useEffect, useState, useCallback } from "react";
import type { RoundResult, DamageEvent } from "../types";

interface Props {
  src: string | null;
  seekToMs: number | null;
  onSeeked: () => void;
  durationMs: number;
  rounds: RoundResult[];
  damageEvents: DamageEvent[];
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
}: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const progressRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(durationMs / 1000);
  const [playing, setPlaying] = useState(false);
  const [volume, setVolume] = useState(0.5);
  const [playbackRate, setPlaybackRate] = useState(1);

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
    const handler = () => setCurrentTime(video.currentTime);
    video.addEventListener("timeupdate", handler);
    return () => video.removeEventListener("timeupdate", handler);
  }, [src]);

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
    if (v.paused) {
      v.play();
      setPlaying(true);
    } else {
      v.pause();
      setPlaying(false);
    }
  }, []);

  const handleProgressClick = useCallback(
    (e: React.MouseEvent) => {
      const bar = progressRef.current;
      const video = videoRef.current;
      if (!bar || !video) return;
      const rect = bar.getBoundingClientRect();
      const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      video.currentTime = pct * duration;
    },
    [duration]
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
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [togglePlay, skipSeconds]);

  const progress = duration > 0 ? currentTime / duration : 0;

  // Find current round
  const currentMs = currentTime * 1000;
  const currentRound = rounds.find(
    (r) => currentMs >= r.round_start_ms && currentMs <= r.round_end_ms
  );

  return (
    <div ref={containerRef} className="bg-surface-2 rounded-lg overflow-hidden border border-surface-4/50">
      {/* Video */}
      <div className="relative bg-black aspect-video">
        {src ? (
          <>
            <video
              ref={videoRef}
              src={src}
              className="w-full h-full object-contain"
              onClick={togglePlay}
              onPlay={() => setPlaying(true)}
              onPause={() => setPlaying(false)}
            />
            {/* Overlay: current round info */}
            {currentRound && (
              <div className="absolute top-2 left-2 bg-black/60 backdrop-blur-sm rounded px-2 py-1 text-[10px] text-text-secondary pointer-events-none">
                Round {currentRound.round_index + 1}
                {currentRound.is_comeback && (
                  <span className="ml-1.5 text-accent-gold">COMEBACK</span>
                )}
              </div>
            )}
            {/* Play overlay when paused */}
            {!playing && (
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
          style={{ width: `${progress * 100}%` }}
        />

        {/* Round region shading */}
        {rounds.map((r, i) => {
          const start = (r.round_start_ms / 1000 / duration) * 100;
          const end = (r.round_end_ms / 1000 / duration) * 100;
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
        {rounds.map((r, i) => {
          const pos = (r.round_end_ms / 1000 / duration) * 100;
          return (
            <div
              key={`round-${i}`}
              className="absolute top-0 bottom-0 w-px bg-text-muted/40"
              style={{ left: `${pos}%` }}
            />
          );
        })}

        {/* Comeback markers */}
        {rounds
          .filter((r) => r.is_comeback)
          .map((r, i) => {
            const pos = (r.deficit_timestamp_ms / 1000 / duration) * 100;
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
        {damageEvents
          .filter((e) => e.damage_pct > 0.30)
          .map((e, i) => {
            const pos = (e.timestamp_ms / 1000 / duration) * 100;
            const color = e.target_side === 1 ? "bg-p1/40" : "bg-p2/40";
            return (
              <div
                key={`dmg-${i}`}
                className={`absolute top-0 bottom-0 w-0.5 ${color}`}
                style={{ left: `${pos}%` }}
              />
            );
          })}

        {/* Playhead */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-text-primary"
          style={{ left: `${progress * 100}%` }}
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
      </div>
    </div>
  );
}
