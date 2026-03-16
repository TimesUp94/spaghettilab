import type { Replay } from "../types";

interface Props {
  replays: Replay[];
  selectedReplay: Replay | null;
  onSelect: (r: Replay) => void;
  onOpenDb: () => void;
  onReload?: () => void;
  reloading?: boolean;
  onReanalyze?: () => void;
  reanalyzing?: boolean;
  onAnalyzeNew?: () => void;
}

function formatDuration(ms: number): string {
  const s = ms / 1000;
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

function extractSetInfo(replayId: string): {
  number: string;
  players: string;
  bracket: string;
} {
  // Try to parse format like "01_WRI_Slycoops_vs_Leftover"
  const match = replayId.match(
    /^(\d+)_([A-Za-z]+)_(.+?)_vs_(.+?)$/
  );
  if (match) {
    return {
      number: match[1],
      players: `${match[3]} vs ${match[4]}`,
      bracket: match[2],
    };
  }
  return { number: "", players: replayId, bracket: "" };
}

export function Sidebar({ replays, selectedReplay, onSelect, onReload, reloading, onReanalyze, reanalyzing, onAnalyzeNew }: Props) {
  return (
    <aside className="w-[260px] bg-surface-1 border-r border-surface-4/50 flex flex-col shrink-0">
      <div className="p-3 border-b border-surface-4/50 flex items-center justify-between">
        <h2 className="text-xs font-semibold text-text-secondary uppercase tracking-wider">
          Sets ({replays.length})
        </h2>
        <div className="flex items-center gap-2">
          {onReanalyze && (
            <button
              onClick={onReanalyze}
              disabled={reanalyzing || reloading}
              className="text-[10px] text-text-muted hover:text-accent-green transition-colors cursor-pointer disabled:opacity-50"
              title="Re-run full CV analysis (Python + round detection)"
            >
              {reanalyzing ? "Analyzing..." : "Reanalyze"}
            </button>
          )}
          {onReload && (
            <button
              onClick={onReload}
              disabled={reloading || reanalyzing}
              className="text-[10px] text-text-muted hover:text-accent-purple transition-colors cursor-pointer disabled:opacity-50"
              title="Re-run round detection only (fast)"
            >
              {reloading ? "..." : "Reload"}
            </button>
          )}
        </div>
      </div>
      {onAnalyzeNew && (
        <div className="p-2 border-b border-surface-4/50">
          <button
            onClick={onAnalyzeNew}
            className="w-full px-3 py-2 text-xs font-medium text-accent-green bg-accent-green/8 border border-accent-green/20
                       rounded-lg hover:bg-accent-green/15 hover:border-accent-green/35 transition-all cursor-pointer"
          >
            + Analyze Video
          </button>
        </div>
      )}
      <div className="flex-1 overflow-y-auto">
        {replays.map((r) => {
          const isSelected = selectedReplay?.replay_id === r.replay_id;
          const info = extractSetInfo(r.replay_id);
          return (
            <button
              key={r.replay_id}
              onClick={() => onSelect(r)}
              className={`w-full text-left px-3 py-2.5 border-b border-surface-4/30 transition-all duration-150 cursor-pointer
                ${
                  isSelected
                    ? "bg-accent-purple/10 border-l-2 border-l-accent-purple"
                    : "hover:bg-surface-2 border-l-2 border-l-transparent"
                }`}
            >
              <div className="flex items-center gap-2">
                {info.number && (
                  <span className="text-text-muted text-[10px] font-mono w-5 shrink-0">
                    {info.number}
                  </span>
                )}
                <div className="min-w-0 flex-1">
                  <div
                    className={`text-xs truncate ${
                      isSelected ? "text-text-primary" : "text-text-secondary"
                    }`}
                  >
                    {info.players}
                  </div>
                  <div className="flex items-center gap-2 mt-0.5">
                    {info.bracket && (
                      <span className="text-[10px] text-accent-purple/70 font-medium">
                        {info.bracket}
                      </span>
                    )}
                    <span className="text-[10px] text-text-muted font-mono">
                      {formatDuration(r.duration_ms)}
                    </span>
                  </div>
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </aside>
  );
}
