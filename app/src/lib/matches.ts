import type { DamageEvent, Match, MatchStats, Replay, RoundResult } from "../types";

/** Group rounds into matches (first-to-2 round wins). */
export function groupRoundsIntoMatches(rounds: RoundResult[]): Match[] {
  const matches: Match[] = [];
  let currentRounds: RoundResult[] = [];
  let p1Wins = 0;
  let p2Wins = 0;

  const flushMatch = () => {
    if (currentRounds.length > 0) {
      matches.push({
        match_index: matches.length,
        rounds: currentRounds,
        winner: p1Wins >= 2 ? "P1" : p2Wins >= 2 ? "P2" : p1Wins > p2Wins ? "P1" : p2Wins > p1Wins ? "P2" : "??",
        p1_rounds_won: p1Wins,
        p2_rounds_won: p2Wins,
        start_ms: currentRounds[0].round_start_ms,
        end_ms: currentRounds[currentRounds.length - 1].round_end_ms,
      });
      currentRounds = [];
      p1Wins = 0;
      p2Wins = 0;
    }
  };

  for (const round of rounds) {
    // Force new match if backend detected a match boundary (timer=99, both 0 wins)
    if (round.is_match_start && currentRounds.length > 0) {
      flushMatch();
    }

    currentRounds.push(round);
    if (round.winner === "P1") p1Wins++;
    else if (round.winner === "P2") p2Wins++;

    if (p1Wins >= 2 || p2Wins >= 2) {
      flushMatch();
    }
  }

  // Remaining rounds that didn't complete a match
  flushMatch();

  return matches;
}

/** The replay-wide stats narrowed to one game. */
export function statsForMatch(stats: MatchStats, match: Match, damageEvents: DamageEvent[]): MatchStats {
  const matchDamage = damageEvents.filter(
    (e) => e.timestamp_ms >= match.start_ms && e.timestamp_ms <= match.end_ms
  );
  const p1Hits = matchDamage.filter((e) => e.target_side === 1);
  const p2Hits = matchDamage.filter((e) => e.target_side === 2);
  const durations = match.rounds.map((r) => (r.round_end_ms - r.round_start_ms) / 1000);
  const winnerHps = match.rounds.map((r) => r.winner_final_hp);
  const mean = (xs: number[]) => (xs.length > 0 ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
  return {
    ...stats,
    total_rounds: match.rounds.length,
    p1_round_wins: match.p1_rounds_won,
    p2_round_wins: match.p2_rounds_won,
    total_damage_events: matchDamage.length,
    p1_damage_taken: p1Hits.reduce((s, e) => s + e.damage_pct, 0),
    p2_damage_taken: p2Hits.reduce((s, e) => s + e.damage_pct, 0),
    p1_biggest_hit: p1Hits.length > 0 ? Math.max(...p1Hits.map((e) => e.damage_pct)) : 0,
    p2_biggest_hit: p2Hits.length > 0 ? Math.max(...p2Hits.map((e) => e.damage_pct)) : 0,
    avg_round_duration_s: mean(durations),
    longest_round_s: durations.length > 0 ? Math.max(...durations) : 0,
    shortest_round_s: durations.length > 0 ? Math.min(...durations) : 0,
    comeback_count: match.rounds.filter((r) => r.is_comeback).length,
    close_rounds: match.rounds.filter((r) => r.winner_final_hp < 0.2).length,
    avg_winner_final_hp: mean(winnerHps),
    duration_s: (match.end_ms - match.start_ms) / 1000,
  };
}

/** m:ss (h:mm:ss past an hour). */
export function formatClock(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return h > 0
    ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`
    : `${m}:${String(s).padStart(2, "0")}`;
}

/** "12 min" / "1 h 05 min" */
export function formatLength(ms: number): string {
  const min = Math.round(ms / 60000);
  if (min < 60) return `${Math.max(1, min)} min`;
  return `${Math.floor(min / 60)} h ${String(min % 60).padStart(2, "0")} min`;
}

/** A readable name for a replay: its video file name without folder and extension. */
export function replayTitle(r: Pick<Replay, "replay_id" | "video_path">): string {
  const base = (r.video_path || "").split(/[/\\]/).pop() || "";
  const name = base.replace(/\.[a-z0-9]{2,4}$/i, "").replace(/[_]+/g, " ").trim();
  return name || r.replay_id;
}

export function fileName(path: string | null | undefined): string {
  return (path || "").split(/[/\\]/).pop() || "";
}

/** Tournament set ids like "01_WRI_Slycoops_vs_Leftover" → number, bracket and players. */
export function setInfo(replayId: string): { number: string; bracket: string; p1: string | null; p2: string | null } {
  const m = replayId.match(/^(\d+)_([A-Za-z]+)_(.+?)_vs_(.+?)$/);
  if (m) return { number: m[1], bracket: m[2], p1: m[3], p2: m[4] };
  const vs = replayId.match(/^(?:.*?_)?([^_]+)_vs_(.+)$/);
  if (vs) return { number: "", bracket: "", p1: vs[1], p2: vs[2] };
  return { number: "", bracket: "", p1: null, p2: null };
}
