import type { RoundResult } from "../types";

interface Props {
  rounds: RoundResult[];
  onSeek: (ms: number) => void;
  onExport: (startMs: number, endMs: number) => void;
}

function formatTime(ms: number): string {
  const s = ms / 1000;
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

export function RoundBreakdown({ rounds, onSeek, onExport }: Props) {
  if (rounds.length === 0) {
    return (
      <div className="text-text-muted text-sm text-center py-8">
        No rounds detected
      </div>
    );
  }

  return (
    <div className="space-y-1.5">
      {rounds.map((r, i) => {
        const dur = ((r.round_end_ms - r.round_start_ms) / 1000).toFixed(0);
        const isP1Win = r.winner === "P1";

        return (
          <div
            key={i}
            className={`flex items-center gap-3 px-3 py-2 rounded-lg bg-surface-2 border transition-all duration-150 cursor-pointer hover:bg-surface-3 ${
              r.is_comeback
                ? "border-accent-gold/20 glow-gold"
                : "border-surface-4/30"
            }`}
            onClick={() => onSeek(r.round_start_ms)}
          >
            {/* Round number */}
            <span className="text-text-muted text-xs font-mono w-4 shrink-0">
              {i + 1}
            </span>

            {/* Winner indicator */}
            <div
              className={`w-1.5 h-8 rounded-full ${
                isP1Win ? "bg-p1" : "bg-p2"
              }`}
            />

            {/* Round info */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span
                  className={`text-xs font-medium ${
                    isP1Win ? "text-p1-light" : "text-p2-light"
                  }`}
                >
                  {r.winner} wins
                </span>
                {r.is_comeback && (
                  <span className="text-[10px] bg-accent-gold/15 text-accent-gold px-1.5 py-0.5 rounded font-medium">
                    COMEBACK {(r.max_deficit * 100).toFixed(0)}%
                  </span>
                )}
                {r.winner_final_hp < 0.10 && !r.is_comeback && (
                  <span className="text-[10px] bg-p1/10 text-p1-light px-1.5 py-0.5 rounded">
                    CLOSE
                  </span>
                )}
                {r.winner_final_hp > 0.90 && (
                  <span className="text-[10px] bg-accent-green/10 text-accent-green px-1.5 py-0.5 rounded">
                    DOMINANT
                  </span>
                )}
              </div>
              <div className="text-[10px] text-text-muted mt-0.5">
                {formatTime(r.round_start_ms)} - {formatTime(r.round_end_ms)}
              </div>
            </div>

            {/* Stats */}
            <div className="flex items-center gap-4 text-xs">
              <div className="text-center">
                <div className="text-text-muted text-[10px]">Final HP</div>
                <div className="font-mono text-text-primary">
                  {(r.winner_final_hp * 100).toFixed(0)}%
                </div>
              </div>
              <div className="text-center">
                <div className="text-text-muted text-[10px]">Min HP</div>
                <div className="font-mono text-text-primary">
                  {(r.winner_min_hp * 100).toFixed(0)}%
                </div>
              </div>
              <div className="text-center">
                <div className="text-text-muted text-[10px]">Duration</div>
                <div className="font-mono text-text-primary">{dur}s</div>
              </div>
            </div>

            {/* Export button */}
            <button
              onClick={(e) => {
                e.stopPropagation();
                onExport(r.round_start_ms, r.round_end_ms);
              }}
              className="btn-ghost !px-2 !py-1 text-[10px] shrink-0"
              title="Export round clip"
            >
              Export
            </button>
          </div>
        );
      })}
    </div>
  );
}
