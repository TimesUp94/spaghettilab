import { useState, useMemo } from "react";
import type { DamageEvent } from "../types";

interface Props {
  events: DamageEvent[];
  onSeek: (ms: number) => void;
}

function formatTime(ms: number): string {
  const s = ms / 1000;
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(1);
  return `${m}:${sec.padStart(4, "0")}`;
}

type SortKey = "time" | "damage" | "target";

export function DamageLog({ events, onSeek }: Props) {
  const [sortKey, setSortKey] = useState<SortKey>("time");
  const [filterSide, setFilterSide] = useState<number | null>(null);

  const sortedEvents = useMemo(() => {
    let filtered = filterSide
      ? events.filter((e) => e.target_side === filterSide)
      : events;

    return [...filtered].sort((a, b) => {
      switch (sortKey) {
        case "damage":
          return b.damage_pct - a.damage_pct;
        case "target":
          return a.target_side - b.target_side;
        default:
          return a.timestamp_ms - b.timestamp_ms;
      }
    });
  }, [events, sortKey, filterSide]);

  return (
    <div>
      {/* Filters */}
      <div className="flex items-center gap-2 mb-3">
        <span className="text-[10px] text-text-muted uppercase tracking-wider">
          Filter:
        </span>
        {[
          { label: "All", value: null },
          { label: "P1", value: 1 },
          { label: "P2", value: 2 },
        ].map(({ label, value }) => (
          <button
            key={label}
            className={`text-[10px] px-2 py-1 rounded cursor-pointer transition-colors ${
              filterSide === value
                ? "bg-accent-purple/20 text-accent-purple"
                : "text-text-muted hover:text-text-secondary"
            }`}
            onClick={() => setFilterSide(value)}
          >
            {label}
          </button>
        ))}
        <span className="text-text-muted text-[10px] ml-2">|</span>
        <span className="text-[10px] text-text-muted uppercase tracking-wider ml-2">
          Sort:
        </span>
        {(["time", "damage"] as const).map((key) => (
          <button
            key={key}
            className={`text-[10px] px-2 py-1 rounded cursor-pointer transition-colors ${
              sortKey === key
                ? "bg-accent-purple/20 text-accent-purple"
                : "text-text-muted hover:text-text-secondary"
            }`}
            onClick={() => setSortKey(key)}
          >
            {key === "time" ? "Time" : "Damage"}
          </button>
        ))}
        <span className="text-text-muted text-[10px] ml-auto">
          {sortedEvents.length} events
        </span>
      </div>

      {/* Table */}
      <div className="bg-surface-2 rounded-lg border border-surface-4/50 overflow-hidden">
        <table className="w-full text-xs">
          <thead>
            <tr className="bg-surface-3/50 text-text-muted text-[10px] uppercase tracking-wider">
              <th className="text-left py-2 px-3 font-medium">Time</th>
              <th className="text-left py-2 px-3 font-medium">Target</th>
              <th className="text-right py-2 px-3 font-medium">Damage</th>
              <th className="text-right py-2 px-3 font-medium">Pre HP</th>
              <th className="text-right py-2 px-3 font-medium">Post HP</th>
              <th className="py-2 px-3 w-24"></th>
            </tr>
          </thead>
          <tbody>
            {sortedEvents.map((e, i) => {
              const isP1 = e.target_side === 1;
              const barWidth = Math.min(e.damage_pct * 100 * 2, 100);
              return (
                <tr
                  key={i}
                  className="border-t border-surface-4/20 hover:bg-surface-3/30 cursor-pointer transition-colors"
                  onClick={() => onSeek(e.timestamp_ms)}
                >
                  <td className="py-1.5 px-3 font-mono text-text-secondary">
                    {formatTime(e.timestamp_ms)}
                  </td>
                  <td className="py-1.5 px-3">
                    <span
                      className={`font-medium ${
                        isP1 ? "text-p1-light" : "text-p2-light"
                      }`}
                    >
                      {isP1 ? "P1" : "P2"}
                    </span>
                  </td>
                  <td className="py-1.5 px-3 text-right">
                    <div className="flex items-center justify-end gap-2">
                      <div className="w-16 h-1.5 bg-surface-4/30 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full ${
                            isP1 ? "bg-p1/60" : "bg-p2/60"
                          }`}
                          style={{ width: `${barWidth}%` }}
                        />
                      </div>
                      <span className="font-mono text-text-primary w-10 text-right">
                        {(e.damage_pct * 100).toFixed(1)}%
                      </span>
                    </div>
                  </td>
                  <td className="py-1.5 px-3 text-right font-mono text-text-secondary">
                    {(e.pre_health_pct * 100).toFixed(0)}%
                  </td>
                  <td className="py-1.5 px-3 text-right font-mono text-text-secondary">
                    {(e.post_health_pct * 100).toFixed(0)}%
                  </td>
                  <td className="py-1.5 px-3 text-center">
                    <button
                      className="btn-ghost !px-2 !py-0.5 text-[10px]"
                      onClick={(ev) => {
                        ev.stopPropagation();
                        onSeek(e.timestamp_ms);
                      }}
                    >
                      Jump
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
