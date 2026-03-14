import type { Replay } from "../types";

interface Props {
  replays: Replay[];
  selectedReplay: Replay | null;
  onSelect: (r: Replay) => void;
  onOpenDb: () => void;
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

export function Sidebar({ replays, selectedReplay, onSelect }: Props) {
  return (
    <aside className="w-[260px] bg-surface-1 border-r border-surface-4/50 flex flex-col shrink-0">
      <div className="p-3 border-b border-surface-4/50">
        <h2 className="text-xs font-semibold text-text-secondary uppercase tracking-wider">
          Sets ({replays.length})
        </h2>
      </div>
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
