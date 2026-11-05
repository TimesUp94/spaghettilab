// Hit counts next to damage instances and Lab openers. The event's counts come from the backend (combos.rs,
// `combo_hits`); here is how they read, and the link from a Lab opener to its combo.
import type { Combo, DamageEvent } from "../types";

/** "7 hits"; "3+2 hits" when the counter restarted inside one damage event; "" without a count. */
export function hitsLabel(hits: number[] | undefined | null): string {
  return hits && hits.length > 0 ? `${hits.join("+")} hits` : "";
}

/** What a damage row says about its combo. */
export function eventHitsLabel(e: DamageEvent): string {
  const label = hitsLabel(e.combo_hits);
  return label && e.combo_continued ? `↳ ${label}` : label;
}

export function eventHitsTitle(e: DamageEvent): string {
  if (!e.combo_hits?.length) return "";
  if (e.combo_continued) return `The same ${e.combo_hits[0]}-hit combo as the damage before it, from the game's hit counter`;
  if (e.combo_hits.length > 1) return `The hit counter restarted inside this damage: ${e.combo_hits.join(" hits, then ")} hits`;
  return `From the game's hit counter`;
}

// An opener is the combo's first hit, and the counter appears from the second on (LAB_RESULT.md `combo_hits`,
// the same numbers as lab_combos.py).
const OPENER_BEFORE_S = 0.5;
const OPENER_AFTER_S = 3;

/**
 * Opener time -> its combo's hits, for results made before `combo_hits` existed: the first combo of the same
 * attacker starting in the window, and only if no later opener of that attacker is nearer to it.
 */
export function openerHitsFromCombos<O extends { time_s: number; attacker: string }>(openers: O[], combos: Combo[]): Map<O, number> {
  const out = new Map<O, number>();
  for (const side of ["P1", "P2"]) {
    const own = openers.filter((o) => o.attacker === side).sort((a, b) => a.time_s - b.time_s);
    const list = combos.filter((c) => c.attacker_side === (side === "P1" ? 1 : 2));
    own.forEach((o, i) => {
      const next = own[i + 1]?.time_s ?? Infinity;
      const c = list.find((c) => c.start_ms / 1000 >= o.time_s - OPENER_BEFORE_S && c.start_ms / 1000 <= Math.min(o.time_s + OPENER_AFTER_S, next - OPENER_BEFORE_S));
      if (c) out.set(o, c.max_hits);
    });
  }
  return out;
}
