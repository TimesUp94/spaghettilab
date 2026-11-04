// The lab result (lab_result.json, schema "spaghetti-lab-result/1", see
// tools/ggst_logger/model/LAB_RESULT.md) and its client-side search presets. The presets mirror
// search_index.py and must return the same moments as the web UI (FGCScoreboard): keep them exactly
// as LAB_RESULT.md defines them.

export type Side = "P1" | "P2";
export type Outcome = "counter" | "hit" | "block" | "whiff" | "interrupted" | "teched" | "none";
export type StateClass = "neutral" | "attacking" | "blockstun" | "hitstun" | "knockdown" | "airborne";

export interface LabOpener {
  time_s: number;
  video_frame?: number;
  attacker: Side;
  attacker_char: string;
  defender_char?: string;
  detector_score?: number;
  category: string;
  category_p?: number;
  move: string;
  move_p: number;
  move_top3?: [string, number][];
  normal?: string | null;
  hit_kind: "counter" | "punish" | "none" | string;
  hit_kind_p?: number;
}

export interface LabEvent {
  t: number;
  frame: number;
  side: Side;
  char: string;
  category: string;
  move: string;
  move_p: number;
  normal?: string | null;
  outcome: Outcome | string;
  outcome_p: number | null;
  opener: boolean;
  hit_kind: string | null;
}

export interface LabResult {
  schema: string;
  video?: string;
  duration_s?: number;
  fps?: number;
  characters: { P1: string; P2: string };
  capabilities: { openers: boolean; transcript: boolean; facts: boolean; ai_summary?: boolean };
  openers: LabOpener[];
  transcript: { events: LabEvent[] } | null;
  states: { hz: number; P1: StateClass[]; P2: StateClass[] } | null;
  facts: { md?: { it?: string; en?: string }; caveats?: unknown; [k: string]: unknown } | null;
  prompts?: { it?: string; en?: string };
  timings_s?: Record<string, number>;
  /** optional: an LLM's summary + coaching notes from the facts, checked against them (LAB_RESULT.md "ai_summary") */
  ai_summary?: AiSummary | null;
}

export interface AiSummaryLang {
  /** only when grounded */
  text: string | null;
  grounded: boolean;
  errors: string[];
  warnings: string[];
  usage?: { model?: string; input_tokens?: number; output_tokens?: number } | null;
}

export interface AiSummary {
  provider: string;
  model?: string | null;
  langs: Partial<Record<LabLang, AiSummaryLang>>;
}

/** The AI summary of one language, as the UI shows it: the text only when grounded. */
export type AiShown =
  | { kind: "grounded"; text: string; warnings: string[]; model: string | null }
  | { kind: "withheld"; errors: string[]; authHint: boolean };

/** Errors meaning the API key is missing or refused (the run needs ANTHROPIC_API_KEY). */
export const isAuthError = (e: string) => /api[_ -]?key|authenticat|unauthori[sz]ed|\b401\b|x-api-key/i.test(e);

/** null when the result has no AI summary for this language (then nothing is shown). */
export function aiSummaryFor(r: LabResult, lang: LabLang): AiShown | null {
  const l = r.ai_summary?.langs?.[lang];
  if (!l) return null;
  const errors = Array.isArray(l.errors) ? l.errors.map(String) : [];
  if (l.grounded === true && typeof l.text === "string" && l.text.trim()) {
    return { kind: "grounded", text: l.text, warnings: Array.isArray(l.warnings) ? l.warnings.map(String) : [], model: r.ai_summary?.model ?? null };
  }
  return { kind: "withheld", errors, authHint: errors.some(isAuthError) };
}

export interface LabOptions {
  transcript?: "auto" | "on" | "off";
  light?: boolean;
  device?: "auto" | "cuda" | "cpu";
  video_path?: string;
  models_dir?: string;
  /** match title in the facts (the replay's name) */
  title?: string;
  /** "claude": also an AI summary from the facts (needs ANTHROPIC_API_KEY); "none": never */
  llm?: "none" | "claude";
}

export interface LabProgress {
  replay_id: string;
  stage: string;
  line: string;
}

/** Errors meaning the lab isn't set up (lab.rs NOT_INSTALLED). */
export const NOT_INSTALLED = "Lab analysis is not installed";

/** Below this the model's guess is shown as uncertain. */
export const UNCERTAIN_P = 0.5;

export const OUTCOMES: Outcome[] = ["counter", "hit", "block", "whiff", "interrupted", "teched", "none"];

/** Outcome colours (timeline lanes and chips). */
export const OUTCOME_COLOR: Record<string, string> = {
  counter: "#f0c040",
  hit: "#40c878",
  block: "#6bb0ff",
  whiff: "#8888aa",
  interrupted: "#e87040",
  teched: "#c080f0",
  none: "#444458",
};

/** The stages lab_analyze.py prints (`stage: hud|detect|moves|transcript|facts|summary|done`), as the UI names them. */
export const STAGES: { key: string; label: string }[] = [
  { key: "hud", label: "Reading the HUD" },
  { key: "detect", label: "Recognising moves" },
  { key: "moves", label: "Recognising moves" },
  { key: "transcript", label: "Transcript" },
  { key: "facts", label: "Summary" },
  { key: "summary", label: "AI summary" },
  { key: "done", label: "Done" },
];

export function stageLabel(stage: string): string {
  return STAGES.find((s) => s.key === stage)?.label ?? stage;
}

/** The ordered, de-duplicated steps shown in the progress list. */
export function stageSteps(withTranscript: boolean, withAi = false): string[] {
  const steps = ["Reading the HUD", "Recognising moves"];
  if (withTranscript) steps.push("Transcript");
  steps.push("Summary");
  if (withAi) steps.push("AI summary");
  return steps;
}

export const isUncertain = (p: number | null | undefined) => p != null && p < UNCERTAIN_P;

/** An opener seen as an event (LAB_RESULT.md, the openers fallback). */
export function openerAsEvent(o: LabOpener): LabEvent {
  const counter = o.hit_kind === "counter";
  return {
    t: o.time_s,
    frame: Math.round(o.time_s * 60),
    side: o.attacker,
    char: o.attacker_char,
    category: o.category,
    move: o.move,
    move_p: o.move_p,
    normal: o.normal ?? null,
    outcome: counter ? "counter" : "hit",
    outcome_p: counter ? (o.hit_kind_p ?? null) : null,
    opener: true,
    hit_kind: o.hit_kind,
  };
}

/** The list the search runs on: the transcript, else the openers as events. Sorted by frame. */
export function searchEvents(r: LabResult): LabEvent[] {
  const evs = r.transcript ? [...r.transcript.events] : r.openers.map(openerAsEvent);
  return evs.sort((a, b) => a.frame - b.frame);
}

export type OutcomeWant = "any" | "connected" | Outcome;

export function outcomeMatches(e: LabEvent, want: OutcomeWant): boolean {
  if (want === "any") return true;
  if (want === "connected") return e.outcome === "hit" || e.outcome === "counter";
  return e.outcome === want;
}

export function own(events: LabEvent[], side: Side): LabEvent[] {
  return events.filter((e) => e.side === side).sort((a, b) => a.frame - b.frame);
}

const byFrame = <T extends { frame: number }>(xs: T[]) => xs.sort((a, b) => a.frame - b.frame);

/** One search hit: the event to jump to, plus what led to it (presets 3 and 4). */
export interface Hit {
  e: LabEvent;
  /** after blocked: the blocked move */
  prev?: LabEvent;
  /** after knockdown: frames since the wake-up */
  sinceWake?: number;
}

/** 1. Move + outcome. */
export function presetMoveOutcome(
  events: LabEvent[],
  q: { side: Side | "any"; move: string; outcome: OutcomeWant; openersOnly?: boolean }
): Hit[] {
  return byFrame(
    events.filter(
      (e) =>
        (q.side === "any" || e.side === q.side) &&
        (!q.move || e.move === q.move) &&
        outcomeMatches(e, q.outcome) &&
        (!q.openersOnly || e.opener)
    )
  ).map((e) => ({ e }));
}

/** 2. Counter-hit by: the opponent's moves that counter-hit the victim. */
export function presetCounterBy(events: LabEvent[], q: { victim: Side; move: string }): Hit[] {
  return byFrame(events.filter((e) => e.side !== q.victim && e.outcome === "counter" && (!q.move || e.move === q.move))).map((e) => ({ e }));
}

/** 3. After blocked: the side's next own event within 40 frames of one of its blocked moves. */
export function presetAfterBlock(events: LabEvent[], side: Side): Hit[] {
  const ev = own(events, side);
  const out: Hit[] = [];
  for (let k = 1; k < ev.length; k++) {
    const p = ev[k - 1], r = ev[k];
    if (p.outcome === "block" && r.frame - p.frame <= 40) out.push({ e: r, prev: p });
  }
  return out;
}

/** The 60 fps frames at which a side gets up from a knockdown (search_index._state_runs). */
export function wakeUps(cls: StateClass[]): number[] {
  const wakes: number[] = [];
  let i = 0;
  while (i < cls.length) {
    if (cls[i] === "knockdown") {
      let j = i;
      while (j < cls.length && cls[j] === "knockdown") j++;
      if (j - i >= 2 && j < cls.length && cls[j] !== "hitstun") wakes.push(8 * j);
      i = j;
    } else i++;
  }
  return wakes;
}

/** 4. After knockdown: the side's first own event after waking up (needs the states). */
export function presetAfterKnockdown(events: LabEvent[], states: LabResult["states"], side: Side): Hit[] {
  if (!states) return [];
  const wakes = wakeUps(states[side] ?? []);
  const ev = own(events, side);
  const out: Hit[] = [];
  let w = 0;
  for (let k = 0; k < ev.length; k++) {
    const r = ev[k];
    const p = k > 0 ? ev[k - 1] : null;
    while (w < wakes.length && wakes[w] < r.frame - 60) w++;
    if (w < wakes.length && wakes[w] - 8 <= r.frame && r.frame <= wakes[w] + 60 && (p === null || p.frame < wakes[w] - 8)) {
      out.push({ e: r, sinceWake: r.frame - wakes[w] });
      w++;
    }
  }
  return out;
}

/** Distinct moves of a side (or both), most used first, without "(rare)". */
export function moveChoices(events: LabEvent[], side: Side | "any"): string[] {
  const n = new Map<string, number>();
  for (const e of events) {
    if ((side !== "any" && e.side !== side) || e.move === "(rare)") continue;
    n.set(e.move, (n.get(e.move) ?? 0) + 1);
  }
  return [...n.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).map(([m]) => m);
}

/** Free filters, ANDed; empty values mean any. */
export interface FreeFilter {
  side: Side | "any";
  char: string;
  move: string;
  category: string;
  outcome: OutcomeWant;
}

export function freeFilter(events: LabEvent[], f: FreeFilter): Hit[] {
  return byFrame(
    events.filter(
      (e) =>
        (f.side === "any" || e.side === f.side) &&
        (!f.char || e.char === f.char) &&
        (!f.move || e.move === f.move) &&
        (!f.category || e.category === f.category) &&
        outcomeMatches(e, f.outcome)
    )
  ).map((e) => ({ e }));
}

/** Seconds to seek to for an event: a second before it, so the lead-up is visible. */
export const seekBefore = (t: number) => Math.max(0, t - 1);

/** m:ss.s */
export function clockTenths(s: number): string {
  const m = Math.floor(s / 60);
  const r = s - m * 60;
  return `${m}:${r.toFixed(1).padStart(4, "0")}`;
}

export type LabLang = "it" | "en";
const LANG_KEY = "spaghettiLab.lab.lang";
export function initialLang(): LabLang {
  const saved = localStorage.getItem(LANG_KEY);
  if (saved === "it" || saved === "en") return saved;
  return navigator.language?.toLowerCase().startsWith("it") ? "it" : "en";
}
export function saveLang(l: LabLang) {
  localStorage.setItem(LANG_KEY, l);
}

const MODELS_KEY = "spaghettiLab.lab.modelsDir";
export const loadModelsDir = () => localStorage.getItem(MODELS_KEY) ?? "";
export const saveModelsDir = (d: string) => (d.trim() ? localStorage.setItem(MODELS_KEY, d.trim()) : localStorage.removeItem(MODELS_KEY));
