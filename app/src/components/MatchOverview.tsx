import type { MatchStats, Replay } from "../types";

interface Props {
  stats: MatchStats;
  replay: Replay;
}

function StatRow({
  label,
  value,
  sub,
}: {
  label: string;
  value: string;
  sub?: string;
}) {
  return (
    <div className="flex justify-between items-baseline py-1">
      <span className="text-text-secondary text-xs">{label}</span>
      <div className="text-right">
        <span className="text-text-primary text-xs font-mono font-medium">
          {value}
        </span>
        {sub && (
          <span className="text-text-muted text-[10px] ml-1">{sub}</span>
        )}
      </div>
    </div>
  );
}

export function MatchOverview({ stats, replay }: Props) {
  const p1Name = replay.replay_id.match(/_([^_]+)_vs_/)?.[1] || "P1";
  const p2Name = replay.replay_id.match(/_vs_(.+)$/)?.[1] || "P2";

  return (
    <div className="bg-surface-2 rounded-lg border border-surface-4/50 h-full flex flex-col">
      {/* Header with round score */}
      <div className="p-3 border-b border-surface-4/30">
        <div className="flex items-center justify-between">
          <div className="text-center flex-1">
            <div className="text-p1-light text-xs font-medium truncate">
              {p1Name}
            </div>
            <div className="text-2xl font-bold text-p1 font-mono">
              {stats.p1_round_wins}
            </div>
          </div>
          <div className="text-text-muted text-xs px-2">vs</div>
          <div className="text-center flex-1">
            <div className="text-p2-light text-xs font-medium truncate">
              {p2Name}
            </div>
            <div className="text-2xl font-bold text-p2 font-mono">
              {stats.p2_round_wins}
            </div>
          </div>
        </div>
      </div>

      {/* Stats grid */}
      <div className="p-3 flex-1 overflow-y-auto space-y-0.5">
        <StatRow
          label="Total Rounds"
          value={stats.total_rounds.toString()}
        />
        <StatRow
          label="Duration"
          value={`${Math.floor(stats.duration_s / 60)}:${Math.floor(stats.duration_s % 60)
            .toString()
            .padStart(2, "0")}`}
        />

        <div className="h-px bg-surface-4/30 my-1.5" />

        <StatRow
          label="Damage Events"
          value={stats.total_damage_events.toString()}
        />
        <StatRow
          label={`${p1Name} Dmg Taken`}
          value={`${(stats.p1_damage_taken * 100).toFixed(0)}%`}
        />
        <StatRow
          label={`${p2Name} Dmg Taken`}
          value={`${(stats.p2_damage_taken * 100).toFixed(0)}%`}
        />
        <StatRow
          label={`${p1Name} Biggest Hit`}
          value={`${(stats.p1_biggest_hit * 100).toFixed(0)}%`}
        />
        <StatRow
          label={`${p2Name} Biggest Hit`}
          value={`${(stats.p2_biggest_hit * 100).toFixed(0)}%`}
        />

        <div className="h-px bg-surface-4/30 my-1.5" />

        <StatRow
          label="Avg Round"
          value={`${stats.avg_round_duration_s.toFixed(0)}s`}
          sub={`${stats.shortest_round_s.toFixed(0)}-${stats.longest_round_s.toFixed(0)}s`}
        />
        <StatRow
          label="Avg Winner HP"
          value={`${(stats.avg_winner_final_hp * 100).toFixed(0)}%`}
        />
        <StatRow
          label="Comebacks"
          value={stats.comeback_count.toString()}
        />
        <StatRow
          label="Close Rounds"
          value={stats.close_rounds.toString()}
          sub="<20% HP"
        />
      </div>
    </div>
  );
}
