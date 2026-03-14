import type { Highlight } from "../types";

interface Props {
  highlights: Highlight[];
  onSeek: (ms: number) => void;
  onExport: (startMs: number, endMs: number) => void;
}

function formatTime(ms: number): string {
  const s = ms / 1000;
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

const HIGHLIGHT_STYLES: Record<
  string,
  { bg: string; border: string; badge: string; badgeBg: string }
> = {
  comeback: {
    bg: "hover:bg-accent-gold/5",
    border: "border-accent-gold/20",
    badge: "text-accent-gold",
    badgeBg: "bg-accent-gold/15",
  },
  close_round: {
    bg: "hover:bg-p1/5",
    border: "border-p1/20",
    badge: "text-p1-light",
    badgeBg: "bg-p1/15",
  },
  big_damage: {
    bg: "hover:bg-accent-purple/5",
    border: "border-accent-purple/20",
    badge: "text-accent-purple",
    badgeBg: "bg-accent-purple/15",
  },
  perfect: {
    bg: "hover:bg-accent-green/5",
    border: "border-accent-green/20",
    badge: "text-accent-green",
    badgeBg: "bg-accent-green/15",
  },
};

const KIND_LABELS: Record<string, string> = {
  comeback: "COMEBACK",
  close_round: "CLOSE",
  big_damage: "BIG HIT",
  perfect: "DOMINANT",
};

export function HighlightsPanel({ highlights, onSeek, onExport }: Props) {
  if (highlights.length === 0) {
    return (
      <div className="text-text-muted text-sm text-center py-8">
        No highlights detected
      </div>
    );
  }

  // Group by kind
  const grouped: Record<string, Highlight[]> = {};
  for (const h of highlights) {
    if (!grouped[h.kind]) grouped[h.kind] = [];
    grouped[h.kind].push(h);
  }

  const order = ["comeback", "close_round", "big_damage", "perfect"];

  return (
    <div className="space-y-4">
      {order
        .filter((kind) => grouped[kind]?.length)
        .map((kind) => {
          const items = grouped[kind];
          const style = HIGHLIGHT_STYLES[kind] || HIGHLIGHT_STYLES.big_damage;
          const label = KIND_LABELS[kind] || kind.toUpperCase();

          return (
            <div key={kind}>
              <div className="flex items-center gap-2 mb-2">
                <span
                  className={`text-[10px] font-semibold uppercase tracking-wider ${style.badge}`}
                >
                  {label}
                </span>
                <span className="text-text-muted text-[10px]">
                  ({items.length})
                </span>
              </div>
              <div className="space-y-1.5">
                {items.map((h, i) => (
                  <div
                    key={i}
                    className={`flex items-center gap-3 px-3 py-2.5 rounded-lg bg-surface-2 border ${style.border} ${style.bg} transition-all duration-150 cursor-pointer`}
                    onClick={() => onSeek(h.timestamp_ms)}
                  >
                    <span
                      className={`text-[10px] font-semibold px-1.5 py-0.5 rounded ${style.badgeBg} ${style.badge} shrink-0`}
                    >
                      {(h.severity * 100).toFixed(0)}%
                    </span>

                    <div className="flex-1 min-w-0">
                      <div className="text-xs text-text-primary truncate">
                        {h.label}
                      </div>
                      <div className="text-[10px] text-text-muted mt-0.5 truncate">
                        {h.details}
                      </div>
                    </div>

                    <span className="text-[10px] text-text-muted font-mono shrink-0">
                      {formatTime(h.timestamp_ms)} - {formatTime(h.end_ms)}
                    </span>

                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onExport(h.timestamp_ms, h.end_ms);
                      }}
                      className="btn-ghost !px-2 !py-1 text-[10px] shrink-0"
                    >
                      Export
                    </button>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
    </div>
  );
}
