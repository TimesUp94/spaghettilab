import type { Match } from "../types";

interface Props {
  matches: Match[];
  onSeek: (ms: number) => void;
  onExport: (startMs: number, endMs: number) => void;
  selectedMatchIndex: number | null;
}

function formatTime(ms: number): string {
  const s = ms / 1000;
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

export function MatchList({
  matches,
  onSeek,
  onExport,
  selectedMatchIndex,
}: Props) {
  if (matches.length === 0) {
    return (
      <div className="text-text-muted text-sm text-center py-8">
        No matches detected
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {matches.map((m) => {
        const dur = ((m.end_ms - m.start_ms) / 1000).toFixed(0);
        const isP1Win = m.winner === "P1";
        const hasComeback = m.rounds.some((r) => r.is_comeback);
        const isSelected = selectedMatchIndex === m.match_index;

        return (
          <div
            key={m.match_index}
            className={`rounded-lg bg-surface-2 border transition-all duration-150 ${
              isSelected
                ? "border-accent-purple/50 ring-1 ring-accent-purple/20"
                : hasComeback
                  ? "border-accent-gold/20"
                  : "border-surface-4/30"
            }`}
          >
            {/* Match header */}
            <div
              className="flex items-center gap-3 px-3 py-2.5 cursor-pointer hover:bg-surface-3 rounded-t-lg"
              onClick={() => onSeek(m.rounds[0].round_start_ms)}
            >
              <span className="text-text-muted text-xs font-mono w-6 shrink-0">
                G{m.match_index + 1}
              </span>

              {/* Score */}
              <div className="flex items-center gap-2">
                <span
                  className={`text-sm font-bold font-mono ${
                    isP1Win ? "text-p1" : "text-text-secondary"
                  }`}
                >
                  {m.p1_rounds_won}
                </span>
                <span className="text-text-muted text-[10px]">-</span>
                <span
                  className={`text-sm font-bold font-mono ${
                    !isP1Win ? "text-p2" : "text-text-secondary"
                  }`}
                >
                  {m.p2_rounds_won}
                </span>
              </div>

              {/* Winner label */}
              <span
                className={`text-xs font-medium ${
                  isP1Win ? "text-p1-light" : "text-p2-light"
                }`}
              >
                {m.winner} wins
              </span>

              {hasComeback && (
                <span className="text-[10px] bg-accent-gold/15 text-accent-gold px-1.5 py-0.5 rounded font-medium">
                  COMEBACK
                </span>
              )}

              <div className="flex-1" />

              <span className="text-[10px] text-text-muted font-mono">
                {formatTime(m.start_ms)} - {formatTime(m.end_ms)}
              </span>

              <span className="text-[10px] text-text-muted font-mono">
                {dur}s
              </span>

              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onExport(m.start_ms, m.end_ms);
                }}
                className="btn-ghost !px-2 !py-1 text-[10px] shrink-0"
                title="Export match clip"
              >
                Export
              </button>
            </div>

            {/* Round details within match */}
            <div className="border-t border-surface-4/20 px-3 py-1.5 space-y-1">
              {m.rounds.map((r, ri) => {
                const rDur = (
                  (r.round_end_ms - r.round_start_ms) /
                  1000
                ).toFixed(0);
                const rIsP1 = r.winner === "P1";

                return (
                  <div
                    key={ri}
                    className="flex items-center gap-2 py-1 px-2 rounded hover:bg-surface-3 cursor-pointer transition-colors text-xs"
                    onClick={() => onSeek(r.round_start_ms)}
                  >
                    <span className="text-text-muted text-[10px] font-mono w-6">
                      R{ri + 1}
                    </span>
                    <div
                      className={`w-1 h-4 rounded-full ${
                        rIsP1 ? "bg-p1" : "bg-p2"
                      }`}
                    />
                    <span
                      className={`text-[11px] ${
                        rIsP1 ? "text-p1-light" : "text-p2-light"
                      }`}
                    >
                      {r.winner}
                    </span>
                    {r.is_comeback && (
                      <span className="text-[9px] bg-accent-gold/15 text-accent-gold px-1 py-0.5 rounded">
                        COMEBACK
                      </span>
                    )}
                    {r.winner_final_hp < 0.10 && !r.is_comeback && (
                      <span className="text-[9px] bg-p1/10 text-p1-light px-1 py-0.5 rounded">
                        CLOSE
                      </span>
                    )}
                    {r.winner_final_hp > 0.90 && (
                      <span className="text-[9px] bg-accent-green/10 text-accent-green px-1 py-0.5 rounded">
                        DOM
                      </span>
                    )}
                    <div className="flex-1" />
                    <span className="text-[10px] text-text-muted font-mono">
                      HP {(r.winner_final_hp * 100).toFixed(0)}%
                    </span>
                    <span className="text-[10px] text-text-muted font-mono">
                      {rDur}s
                    </span>
                    <span className="text-[10px] text-text-muted font-mono">
                      {formatTime(r.round_start_ms)} - {formatTime(r.round_end_ms)}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}
