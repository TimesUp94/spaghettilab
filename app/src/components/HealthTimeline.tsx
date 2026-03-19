import { useMemo, useCallback } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  ReferenceArea,
} from "recharts";
import type {
  FrameDataPoint,
  RoundResult,
  DamageEvent,
  Highlight,
  Match,
  Note,
} from "../types";

interface Props {
  frameData: FrameDataPoint[];
  rounds: RoundResult[];
  damageEvents: DamageEvent[];
  highlights: Highlight[];
  notes: Note[];
  onSeek: (ms: number) => void;
  selectedMatch: Match | null;
  onClearSelection: () => void;
}

interface ChartPoint {
  time_s: number;
  p1: number | null;
  p2: number | null;
  t1: number | null;
  t2: number | null;
}

interface TensionSpend {
  time_s: number;
  side: "p1" | "p2";
  drop: number; // how much tension was spent (0-1)
}

function formatTimeAxis(s: number): string {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

export function HealthTimeline({
  frameData,
  rounds,
  damageEvents,
  highlights,
  notes,
  onSeek,
  selectedMatch,
  onClearSelection,
}: Props) {
  // Downsample to ~800 points, zoom-aware
  const chartData = useMemo(() => {
    if (frameData.length === 0) return [];
    let data = frameData;
    if (selectedMatch) {
      const pad =
        ((selectedMatch.end_ms - selectedMatch.start_ms) / 1000) * 0.03;
      const lo = selectedMatch.start_ms / 1000 - pad;
      const hi = selectedMatch.end_ms / 1000 + pad;
      data = frameData.filter((f) => {
        const s = f.timestamp_ms / 1000;
        return s >= lo && s <= hi;
      });
    }
    const step = Math.max(1, Math.floor(data.length / 800));
    const points: ChartPoint[] = [];
    let lastP1: number | null = null;
    let lastP2: number | null = null;
    for (let i = 0; i < data.length; i += step) {
      const f = data[i];
      const p1 =
        f.p1_health_pct !== null
          ? Math.min(f.p1_health_pct / 0.875, 1.0)
          : null;
      const p2 =
        f.p2_health_pct !== null
          ? Math.min(f.p2_health_pct / 0.875, 1.0)
          : null;
      // Forward-fill: hold last valid health value through gaps
      if (p1 !== null) lastP1 = p1;
      if (p2 !== null) lastP2 = p2;
      points.push({
        time_s: f.timestamp_ms / 1000,
        p1: p1 ?? lastP1,
        p2: p2 ?? lastP2,
        t1: f.p1_tension_pct,
        t2: f.p2_tension_pct,
      });
    }
    return points;
  }, [frameData, selectedMatch]);

  // Detect tension spend events: significant drops between consecutive points
  const tensionSpends = useMemo(() => {
    if (chartData.length < 2) return [];
    const spends: TensionSpend[] = [];
    const MIN_DROP = 0.12; // ignore tiny jitter
    for (let i = 1; i < chartData.length; i++) {
      const prev = chartData[i - 1];
      const cur = chartData[i];
      if (prev.t1 !== null && cur.t1 !== null) {
        const drop = prev.t1 - cur.t1;
        if (drop >= MIN_DROP) {
          spends.push({ time_s: cur.time_s, side: "p1", drop });
        }
      }
      if (prev.t2 !== null && cur.t2 !== null) {
        const drop = prev.t2 - cur.t2;
        if (drop >= MIN_DROP) {
          spends.push({ time_s: cur.time_s, side: "p2", drop });
        }
      }
    }
    return spends;
  }, [chartData]);

  const handleClick = useCallback(
    (data: any) => {
      if (data?.activePayload?.[0]) {
        const time_s = data.activePayload[0].payload.time_s;
        onSeek(time_s * 1000);
      }
    },
    [onSeek]
  );

  // Filter rounds to visible range when zoomed
  const visibleRounds = useMemo(() => {
    if (!selectedMatch) return rounds;
    return rounds.filter(
      (r) =>
        r.round_start_ms >= selectedMatch.start_ms &&
        r.round_end_ms <= selectedMatch.end_ms
    );
  }, [rounds, selectedMatch]);

  const comebacks = visibleRounds.filter((r) => r.is_comeback);

  // Compute chart time range for overlay positioning
  const chartTimeRange = useMemo(() => {
    if (chartData.length < 2) return { min: 0, max: 1 };
    return { min: chartData[0].time_s, max: chartData[chartData.length - 1].time_s };
  }, [chartData]);

  // YAxis width (32) + left margin (4) = 36px offset for chart area
  const chartLeftPx = 36;
  const chartRightPx = 4; // right margin

  if (chartData.length === 0) {
    return (
      <div className="bg-surface-2 rounded-lg border border-surface-4/50 h-28 flex items-center justify-center text-text-muted text-sm">
        No frame data available
      </div>
    );
  }

  return (
    <div className="bg-surface-2 rounded-lg border border-surface-4/50 p-2">
      <div className="flex items-center justify-between px-2 mb-1">
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-text-muted uppercase tracking-wider font-medium">
            Health & Tension {visibleRounds.length > 0 && `(${visibleRounds.length} rounds)`}
          </span>
          {selectedMatch && (
            <>
              <span className="text-[10px] text-accent-purple font-medium">
                Game {selectedMatch.match_index + 1}
              </span>
              <button
                onClick={onClearSelection}
                className="text-[10px] text-text-muted hover:text-text-secondary transition-colors cursor-pointer px-1"
                title="Show full timeline"
              >
                Show all
              </button>
            </>
          )}
        </div>
        <div className="flex items-center gap-3 text-[10px]">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-p1" />
            <span className="text-text-muted">P1</span>
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-p2" />
            <span className="text-text-muted">P2</span>
          </span>
          {comebacks.length > 0 && (
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-sm bg-accent-gold/60" />
              <span className="text-text-muted">Comeback</span>
            </span>
          )}
          {notes.length > 0 && (
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-sm bg-accent-green/60" />
              <span className="text-text-muted">Notes</span>
            </span>
          )}
        </div>
      </div>
      <div className="relative">
      <ResponsiveContainer width="100%" height={120}>
        <AreaChart
          data={chartData}
          onClick={handleClick}
          margin={{ top: 2, right: 4, bottom: 0, left: 4 }}
        >
          <defs>
            <linearGradient id="p1Gradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#e84040" stopOpacity={0.4} />
              <stop offset="100%" stopColor="#e84040" stopOpacity={0.05} />
            </linearGradient>
            <linearGradient id="p2Gradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#4088e8" stopOpacity={0.4} />
              <stop offset="100%" stopColor="#4088e8" stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <XAxis
            dataKey="time_s"
            tickFormatter={formatTimeAxis}
            stroke="#555570"
            tick={{ fontSize: 9, fill: "#555570" }}
            tickLine={false}
            axisLine={{ stroke: "#222233" }}
          />
          <YAxis
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
            stroke="#555570"
            tick={{ fontSize: 9, fill: "#555570" }}
            tickLine={false}
            axisLine={false}
            width={32}
          />
          <Tooltip
            contentStyle={{
              background: "#14141f",
              border: "1px solid #222233",
              borderRadius: "6px",
              fontSize: "11px",
              padding: "6px 8px",
            }}
            labelFormatter={(v: number) => formatTimeAxis(v)}
            formatter={(value: any, name: string) => {
              const pct = ((value as number) * 100).toFixed(0);
              return [`${pct}%`, name === "p1" ? "P1 HP" : "P2 HP"];
            }}
          />

          {/* Note markers */}
          {notes.map((note, i) => (
            <ReferenceLine
              key={`note-${i}`}
              x={note.timestamp_ms / 1000}
              stroke="#40c878"
              strokeDasharray="4 2"
              strokeOpacity={0.7}
            />
          ))}

          {/* Comeback regions */}
          {comebacks.map((r, i) => (
            <ReferenceArea
              key={`cb-${i}`}
              x1={r.round_start_ms / 1000}
              x2={r.round_end_ms / 1000}
              fill="#f0c040"
              fillOpacity={0.06}
              stroke="#f0c040"
              strokeOpacity={0.15}
            />
          ))}

          <Area
            type="monotone"
            dataKey="p1"
            stroke="#e84040"
            strokeWidth={1.5}
            fill="url(#p1Gradient)"
            dot={false}
            connectNulls={true}
            isAnimationActive={false}
          />
          <Area
            type="monotone"
            dataKey="p2"
            stroke="#4088e8"
            strokeWidth={1.5}
            fill="url(#p2Gradient)"
            dot={false}
            connectNulls={true}
            isAnimationActive={false}
          />

          {/* Round end markers removed — using overlay divs instead */}
        </AreaChart>
      </ResponsiveContainer>
      {/* Round end marker overlays — positioned over chart area */}
      <div style={{ position: "absolute", top: 0, bottom: 0, left: chartLeftPx, right: chartRightPx, pointerEvents: "none", zIndex: 10 }}>
        {visibleRounds.map((r, i) => {
          const t = r.round_end_ms / 1000;
          const range = chartTimeRange.max - chartTimeRange.min;
          if (range <= 0) return null;
          const pct = ((t - chartTimeRange.min) / range) * 100;
          if (pct < 0 || pct > 100) return null;
          return (
            <div
              key={`rm-${i}`}
              style={{
                position: "absolute",
                left: `${pct}%`,
                top: 0,
                bottom: 0,
                width: 2,
                backgroundColor: "#f0c040",
              }}
            >
              <span style={{ position: "absolute", top: -1, left: 4, fontSize: 9, fontWeight: 700, color: "#f0c040", whiteSpace: "nowrap" }}>
                {r.winner}
              </span>
            </div>
          );
        })}
      </div>
      </div>

      {/* P1 Tension gauge chart */}
      <div className="flex items-center gap-1 px-2 mt-1">
        <span className="w-1.5 h-1.5 rounded-full bg-p1" />
        <span className="text-[9px] text-text-muted">P1 Tension</span>
        {tensionSpends.filter(s => s.side === "p1").length > 0 && (
          <span className="text-[9px] text-text-muted ml-1">
            <span className="text-p1">▼</span> = spend
          </span>
        )}
      </div>
      <ResponsiveContainer width="100%" height={44}>
        <AreaChart
          data={chartData}
          onClick={handleClick}
          margin={{ top: 2, right: 4, bottom: 0, left: 4 }}
        >
          <defs>
            <linearGradient id="t1Gradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#e84040" stopOpacity={0.3} />
              <stop offset="100%" stopColor="#e84040" stopOpacity={0.03} />
            </linearGradient>
          </defs>
          <XAxis dataKey="time_s" hide />
          <YAxis domain={[0, 1]} hide width={32} />
          <Tooltip
            contentStyle={{
              background: "#14141f",
              border: "1px solid #222233",
              borderRadius: "6px",
              fontSize: "11px",
              padding: "6px 8px",
            }}
            labelFormatter={(v: number) => formatTimeAxis(v)}
            formatter={(value: any) => {
              const pct = ((value as number) * 100).toFixed(0);
              return [`${pct}%`, "P1 Tension"];
            }}
          />
          {/* Round end markers */}
          {visibleRounds.map((r, i) => (
            <ReferenceLine
              key={`t1rb-${i}`}
              x={r.round_end_ms / 1000}
              stroke="#f0c040"
              strokeWidth={2}
              strokeOpacity={0.8}
            />
          ))}
          {/* Tension spend markers for P1 */}
          {tensionSpends
            .filter((s) => s.side === "p1")
            .map((s, i) => (
              <ReferenceLine
                key={`ts1-${i}`}
                x={s.time_s}
                stroke="#e84040"
                strokeWidth={1.5}
                strokeOpacity={0.7}
              />
            ))}
          <Area
            type="monotone"
            dataKey="t1"
            stroke="#e84040"
            strokeWidth={1}
            fill="url(#t1Gradient)"
            dot={false}
            connectNulls={false}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>

      {/* P2 Tension gauge chart */}
      <div className="flex items-center gap-1 px-2">
        <span className="w-1.5 h-1.5 rounded-full bg-p2" />
        <span className="text-[9px] text-text-muted">P2 Tension</span>
        {tensionSpends.filter(s => s.side === "p2").length > 0 && (
          <span className="text-[9px] text-text-muted ml-1">
            <span className="text-p2">▼</span> = spend
          </span>
        )}
      </div>
      <ResponsiveContainer width="100%" height={44}>
        <AreaChart
          data={chartData}
          onClick={handleClick}
          margin={{ top: 2, right: 4, bottom: 0, left: 4 }}
        >
          <defs>
            <linearGradient id="t2Gradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#4088e8" stopOpacity={0.3} />
              <stop offset="100%" stopColor="#4088e8" stopOpacity={0.03} />
            </linearGradient>
          </defs>
          <XAxis dataKey="time_s" hide />
          <YAxis domain={[0, 1]} hide width={32} />
          <Tooltip
            contentStyle={{
              background: "#14141f",
              border: "1px solid #222233",
              borderRadius: "6px",
              fontSize: "11px",
              padding: "6px 8px",
            }}
            labelFormatter={(v: number) => formatTimeAxis(v)}
            formatter={(value: any) => {
              const pct = ((value as number) * 100).toFixed(0);
              return [`${pct}%`, "P2 Tension"];
            }}
          />
          {/* Round end markers */}
          {visibleRounds.map((r, i) => (
            <ReferenceLine
              key={`t2rb-${i}`}
              x={r.round_end_ms / 1000}
              stroke="#f0c040"
              strokeWidth={2}
              strokeOpacity={0.8}
            />
          ))}
          {/* Tension spend markers for P2 */}
          {tensionSpends
            .filter((s) => s.side === "p2")
            .map((s, i) => (
              <ReferenceLine
                key={`ts2-${i}`}
                x={s.time_s}
                stroke="#4088e8"
                strokeWidth={1.5}
                strokeOpacity={0.7}
              />
            ))}
          <Area
            type="monotone"
            dataKey="t2"
            stroke="#4088e8"
            strokeWidth={1}
            fill="url(#t2Gradient)"
            dot={false}
            connectNulls={false}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
