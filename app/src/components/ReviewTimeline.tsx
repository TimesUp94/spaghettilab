// The review timeline under the video: rounds, health, tension and markers on one
// shared time axis, with the playhead. Click or drag anywhere on it to seek.
import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import type { DamageEvent, Drawing, FrameDataPoint, Match, Note, RoundResult } from "../types";
import { formatClock } from "../lib/matches";

interface Props {
  frameData: FrameDataPoint[];
  rounds: RoundResult[];
  matches: Match[];
  damageEvents: DamageEvent[];
  notes: Note[];
  drawings: Drawing[];
  durationMs: number;
  /** Visible time range; the whole replay when no game is selected. */
  range: { start: number; end: number };
  currentMs: number;
  onSeek: (ms: number) => void;
}

const P1 = "#e84040";
const P2 = "#4088e8";
const GOLD = "#f0c040";
const GREEN = "#40c878";

const ROUND_H = 18;
const HP_H = 64;
const TEN_H = 10;
const GAP = 4;
const AXIS_H = 14;

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [w, setW] = useState(0);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setW(Math.floor(e.contentRect.width)));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

/** Path through the points (null breaks the line). */
function linePath(xs: number[], ys: (number | null)[]): string {
  let d = "";
  let pen = false;
  for (let i = 0; i < xs.length; i++) {
    const y = ys[i];
    if (y === null) {
      pen = false;
      continue;
    }
    d += `${pen ? "L" : "M"}${xs[i].toFixed(1)},${y.toFixed(1)}`;
    pen = true;
  }
  return d;
}

export function ReviewTimeline({
  frameData,
  rounds,
  matches,
  damageEvents,
  notes,
  drawings,
  range,
  currentMs,
  onSeek,
}: Props) {
  const [wrapRef, width] = useWidth<HTMLDivElement>();
  const [hoverMs, setHoverMs] = useState<number | null>(null);
  const dragging = useRef(false);

  const span = Math.max(1, range.end - range.start);
  const x = useCallback((ms: number) => ((ms - range.start) / span) * width, [range.start, span, width]);

  const hpTop = ROUND_H + GAP;
  const t1Top = hpTop + HP_H + GAP;
  const t2Top = t1Top + TEN_H + 2;
  const axisTop = t2Top + TEN_H + 2;
  const height = axisTop + AXIS_H;

  // One sample per ~2 px: max-pool HP keeps short dips visible enough while staying light.
  const series = useMemo(() => {
    if (width <= 0 || frameData.length === 0) return null;
    const buckets = Math.max(50, Math.floor(width / 2));
    const acc = Array.from({ length: buckets }, () => ({ p1: null as number | null, p2: null as number | null, t1: null as number | null, t2: null as number | null }));
    for (const f of frameData) {
      if (f.timestamp_ms < range.start || f.timestamp_ms > range.end) continue;
      const b = Math.min(buckets - 1, Math.floor(((f.timestamp_ms - range.start) / span) * buckets));
      const a = acc[b];
      const clamp = (v: number | null) => (v === null ? null : Math.min(1, Math.max(0, v)));
      const p1 = clamp(f.p1_health_pct), p2 = clamp(f.p2_health_pct);
      // the bucket's last reading (HP only falls inside a round)
      if (p1 !== null) a.p1 = p1;
      if (p2 !== null) a.p2 = p2;
      if (f.p1_tension_pct !== null) a.t1 = clamp(f.p1_tension_pct);
      if (f.p2_tension_pct !== null) a.t2 = clamp(f.p2_tension_pct);
    }
    const xs = acc.map((_, i) => ((i + 0.5) / buckets) * width);
    return { xs, acc };
  }, [frameData, range.start, range.end, span, width]);

  const hpY = (v: number) => hpTop + (1 - v) * (HP_H - 2) + 1;

  const ticks = useMemo(() => {
    if (width <= 0) return [];
    const target = Math.max(2, Math.floor(width / 90));
    const steps = [5e3, 10e3, 15e3, 30e3, 60e3, 120e3, 300e3, 600e3, 900e3, 1800e3, 3600e3];
    const step = steps.find((s) => span / s <= target) ?? 3600e3;
    const out: number[] = [];
    for (let t = Math.ceil(range.start / step) * step; t <= range.end; t += step) out.push(t);
    return out;
  }, [width, span, range.start, range.end]);

  const msAt = useCallback(
    (clientX: number) => {
      const rect = wrapRef.current?.getBoundingClientRect();
      if (!rect || rect.width <= 0) return null;
      const f = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
      return range.start + f * span;
    },
    [range.start, span, wrapRef]
  );

  const onPointerDown = (e: React.PointerEvent) => {
    const ms = msAt(e.clientX);
    if (ms === null) return;
    dragging.current = true;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    onSeek(ms);
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const ms = msAt(e.clientX);
    setHoverMs(ms);
    if (dragging.current && ms !== null) onSeek(ms);
  };
  const onPointerUp = (e: React.PointerEvent) => {
    dragging.current = false;
    (e.currentTarget as HTMLElement).releasePointerCapture?.(e.pointerId);
  };

  const inRange = (ms: number) => ms >= range.start && ms <= range.end;
  const playX = x(Math.min(range.end, Math.max(range.start, currentMs)));
  const hoverRound = hoverMs !== null ? rounds.find((r) => hoverMs >= r.round_start_ms && hoverMs <= r.round_end_ms) : undefined;
  const matchStarts = new Set(matches.map((m) => m.start_ms));

  return (
    <div
      ref={wrapRef}
      className="relative select-none cursor-pointer touch-none"
      style={{ height }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerLeave={() => setHoverMs(null)}
    >
      {width > 0 && (
        <svg width={width} height={height} className="block overflow-visible">
          {/* lane backgrounds */}
          <rect x={0} y={hpTop} width={width} height={HP_H} rx={4} fill="#14141f" />
          <rect x={0} y={t1Top} width={width} height={TEN_H} rx={2} fill="#14141f" />
          <rect x={0} y={t2Top} width={width} height={TEN_H} rx={2} fill="#14141f" />
          {[0.5].map((v) => (
            <line key={v} x1={0} x2={width} y1={hpY(v)} y2={hpY(v)} stroke="#222233" strokeDasharray="2 4" />
          ))}

          {/* rounds lane: one block per round, coloured by its winner */}
          {rounds.map((r) => {
            if (r.round_end_ms < range.start || r.round_start_ms > range.end) return null;
            const x0 = Math.max(0, x(r.round_start_ms));
            const x1 = Math.min(width, x(r.round_end_ms));
            const w = Math.max(1, x1 - x0 - 1);
            const col = r.winner === "P1" ? P1 : r.winner === "P2" ? P2 : "#555570";
            return (
              <g key={`r${r.round_index}`}>
                <rect x={x0} y={0} width={w} height={ROUND_H} rx={3} fill={col} fillOpacity={r.winner_confident ? 0.28 : 0.12}
                  stroke={col} strokeOpacity={r.winner_confident ? 0.55 : 0.35} strokeDasharray={r.winner_confident ? undefined : "3 2"} />
                {w > 26 && (
                  <text x={x0 + 5} y={12.5} fontSize={10} fontWeight={600} fill="#e8e8f0" fillOpacity={0.9}>
                    R{r.round_index + 1}
                  </text>
                )}
                {r.is_comeback && (
                  <rect x={x0} y={hpTop} width={w} height={HP_H} fill={GOLD} fillOpacity={0.05} />
                )}
                {/* game boundary */}
                {matchStarts.has(r.round_start_ms) && x0 > 1 && (
                  <line x1={x0 - 1} x2={x0 - 1} y1={0} y2={axisTop} stroke="#8888aa" strokeOpacity={0.5} />
                )}
              </g>
            );
          })}

          {/* health */}
          {series && (
            <>
              <path d={linePath(series.xs, series.acc.map((a) => (a.p1 === null ? null : hpY(a.p1))))} fill="none" stroke={P1} strokeWidth={1.5} strokeLinejoin="round" />
              <path d={linePath(series.xs, series.acc.map((a) => (a.p2 === null ? null : hpY(a.p2))))} fill="none" stroke={P2} strokeWidth={1.5} strokeLinejoin="round" />
              {/* tension as bars */}
              {series.acc.map((a, i) =>
                a.t1 === null ? null : (
                  <rect key={`t1${i}`} x={series.xs[i] - 1} y={t1Top + TEN_H * (1 - a.t1)} width={2} height={TEN_H * a.t1} fill={P1} fillOpacity={0.55} />
                )
              )}
              {series.acc.map((a, i) =>
                a.t2 === null ? null : (
                  <rect key={`t2${i}`} x={series.xs[i] - 1} y={t2Top + TEN_H * (1 - a.t2)} width={2} height={TEN_H * a.t2} fill={P2} fillOpacity={0.55} />
                )
              )}
            </>
          )}

          {/* big hits */}
          {damageEvents.filter((e) => e.damage_pct >= 0.3 && inRange(e.timestamp_ms)).map((e) => (
            <line key={`d${e.event_id}`} x1={x(e.timestamp_ms)} x2={x(e.timestamp_ms)} y1={hpTop} y2={hpTop + HP_H}
              stroke={e.target_side === 1 ? P1 : P2} strokeOpacity={0.35} strokeWidth={2} />
          ))}

          {/* notes and drawings: small flags at the top of the health lane */}
          {notes.filter((n) => inRange(n.timestamp_ms)).map((n) => (
            <g key={`n${n.note_id}`}>
              <line x1={x(n.timestamp_ms)} x2={x(n.timestamp_ms)} y1={hpTop} y2={t2Top + TEN_H} stroke={GREEN} strokeOpacity={0.6} strokeDasharray="3 2" />
              <path d={`M${x(n.timestamp_ms)},${hpTop} l5,3 l-5,3 z`} fill={GREEN} />
            </g>
          ))}
          {drawings.filter((d) => inRange(d.timestamp_ms)).map((d) => (
            <circle key={`dr${d.drawing_id}`} cx={x(d.timestamp_ms)} cy={hpTop + 3} r={3} fill={GOLD} />
          ))}

          {/* time axis */}
          {ticks.map((t) => (
            <text key={t} x={x(t)} y={axisTop + 11} fontSize={9} fill="#555570" textAnchor="middle">
              {formatClock(t)}
            </text>
          ))}

          {/* hover line */}
          {hoverMs !== null && (
            <line x1={x(hoverMs)} x2={x(hoverMs)} y1={0} y2={axisTop} stroke="#e8e8f0" strokeOpacity={0.25} />
          )}

          {/* playhead */}
          <line x1={playX} x2={playX} y1={0} y2={axisTop} stroke="#e8e8f0" strokeWidth={1.5} />
          <circle cx={playX} cy={0} r={3.5} fill="#e8e8f0" />
        </svg>
      )}

      {/* hover tooltip */}
      {hoverMs !== null && width > 0 && (
        <div
          className="absolute -top-7 px-1.5 py-0.5 rounded bg-surface-4 text-[10px] text-text-primary font-mono whitespace-nowrap pointer-events-none -translate-x-1/2"
          style={{ left: Math.min(width - 30, Math.max(30, x(hoverMs))) }}
        >
          {formatClock(hoverMs)}
          {hoverRound && <span className="text-text-secondary"> · R{hoverRound.round_index + 1}</span>}
        </div>
      )}
    </div>
  );
}
