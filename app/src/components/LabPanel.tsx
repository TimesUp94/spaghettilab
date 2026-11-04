// The "Lab" tab of the review screen: analyses the replay on the Spaghetti Showdown servers
// (run_lab_analysis_remote, behind a sign-in; see docs/desktop-login.md) or, only with SPAGHETTI_LAB_MODELS
// set, with the local research models (run_lab_analysis). Then shows the result: combo openers, the move
// transcript, a search over it and the match summary. Cached results stay readable signed out.
// The same panels exist on the website (FGCScoreboard); both follow tools/ggst_logger/model/LAB_RESULT.md.
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { listen } from "@tauri-apps/api/event";
import {
  FlaskConical, Play, Square, Trash2, RefreshCw, Swords, ListTree, Search, FileText, Copy, Check,
  AlertTriangle, Cpu, Settings2, Info, Sparkles,
} from "lucide-react";
import { Badge, Button, EmptyState, Spinner } from "./ui";
import {
  cancelLabAnalysis, cancelLabAnalysisRemote, deleteLabResult, getLabResult, labAccount, labAnalysisRunning, labConfig, labOpenWeb,
  labQuota, labRemoteCancelPhase, labRemoteResume, labRemoteRunning, labRemoteTakeError, labServiceStatus, labSignIn, labSignInCancel, labSignInReopen,
  labSignOut, runLabAnalysis, runLabAnalysisRemote,
} from "../api";
import {
  NOT_INSTALLED, OUTCOME_COLOR, aiSummaryFor, OUTCOMES, cancelNote, clockTenths, freeFilter, initialLang, isUncertain, loadModelsDir,
  moveChoices, presetAfterBlock, presetAfterKnockdown, presetCounterBy, presetMoveOutcome, saveLang, saveModelsDir,
  searchEvents, seekBefore, stageLabel, stageSteps, toRemoteError, wakeUps,
  type Hit, type LabAccount, type LabConfig, type LabEvent, type LabLang, type LabOpener, type LabOptions, type LabProgress,
  type LabQuota, type LabResult, type LabServiceStatus, type OutcomeWant, type RemoteError, type RemoteRunning, type Side,
} from "../lib/lab";
import { AccountBar, HostedForm, HostedProgress, RemoteErrorNote } from "./LabHosted";

// ── state ────────────────────────────────────────────────────────────────────

export interface LabState {
  result: LabResult | null;
  loading: boolean;
  running: boolean;
  /** stage labels seen so far in this run (in order) */
  stages: string[];
  lastLine: string;
  error: string | null;
  run: (o: LabOptions) => void;
  cancel: () => void;
  remove: () => void;

  // the hosted analysis and its sign-in
  /** null until known */
  config: LabConfig | null;
  account: LabAccount | null;
  quota: LabQuota | null;
  service: LabServiceStatus | null;
  /** the run going on is the hosted one */
  hosted: boolean;
  progress: LabProgress | null;
  /** what the server charged this run: "free", "credit", "reused", or null before it said */
  source: string | null;
  cancelling: boolean;
  remoteError: RemoteError | null;
  /** a quiet line about how the last run ended */
  note: string | null;
  signingIn: boolean;
  loginError: RemoteError | null;
  /** Account, service and quota, read when the Lab tab opens, on window focus and after a run. */
  refreshRemote: () => void;
  runRemote: (again: boolean) => void;
  /** What a cancel means right now, from the server. */
  cancelPhase: () => Promise<RemoteRunning["phase"]>;
  cancelRemote: () => void;
  signIn: () => void;
  signInReopen: () => void;
  signInCancel: () => void;
  signOut: () => void;
  openWeb: (page: "support" | "terms" | "privacy") => void;
}

/** The lab result of a replay and the run / cancel / delete actions. */
export function useLab(replayId: string, dbPath: string | null, videoPath: string | null, title?: string): LabState {
  const [result, setResult] = useState<LabResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [stages, setStages] = useState<string[]>([]);
  const [lastLine, setLastLine] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [config, setConfig] = useState<LabConfig | null>(null);
  const [account, setAccount] = useState<LabAccount | null>(null);
  const [quota, setQuota] = useState<LabQuota | null>(null);
  const [service, setService] = useState<LabServiceStatus | null>(null);
  const [hosted, setHosted] = useState(false);
  const [progress, setProgress] = useState<LabProgress | null>(null);
  const [source, setSource] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const [remoteError, setRemoteError] = useState<RemoteError | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [signingIn, setSigningIn] = useState(false);
  const [loginError, setLoginError] = useState<RemoteError | null>(null);
  /** the replay on screen, for answers that arrive after the user moved on */
  const idRef = useRef(replayId);
  idRef.current = replayId;
  /** false once the review screen is gone: an answer that arrives then is for a later visit */
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);
  const onScreen = useCallback((id: string) => mounted.current && idRef.current === id, []);
  /** this screen's own command is waiting for the run to end */
  const ownRun = useRef(false);
  const followTimer = useRef<number | undefined>(undefined);

  const load = useCallback(async () => {
    try {
      setResult(await getLabResult(replayId));
    } catch (e) {
      setResult(null);
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [replayId]);

  useEffect(() => {
    labConfig().then(setConfig).catch(() => setConfig({ local_models: false, web_url: "" }));
  }, []);

  const refreshRemote = useCallback(async () => {
    const acc = await labAccount().catch(() => null);
    if (acc) setAccount(acc);
    labServiceStatus().then(setService).catch(() => setService({ available: false }));
    if (!acc?.signed_in) {
      setQuota(null);
      return;
    }
    try {
      setQuota(await labQuota());
    } catch (e) {
      // only the answer "signed out" ends the session; the rest keeps the last known line
      if (toRemoteError(e).code === "signed_out") {
        setQuota(null);
        setAccount(await labAccount().catch(() => null));
      }
    }
  }, []);

  /** How a hosted run ended, whichever way its end was noticed. */
  const showRemoteEnd = useCallback(
    (e: RemoteError) => {
      if (e.quota) setQuota(e.quota);
      if (e.code === "cancelled") {
        setNote(cancelNote(e));
        return;
      }
      setRemoteError(e);
      if (e.code === "signed_out") labAccount().then(setAccount).catch(() => {});
    },
    []
  );

  /** A run this screen didn't start (it was closed meanwhile, or a saved job was resumed). */
  const follow = useCallback(() => {
    if (followTimer.current !== undefined) return;
    setRunning(true);
    followTimer.current = window.setInterval(async () => {
      const id = await labAnalysisRunning().catch(() => null);
      if (id === idRef.current) return;
      window.clearInterval(followTimer.current);
      followTimer.current = undefined;
      setRunning(false);
      setHosted(false);
      setProgress(null);
      setCancelling(false);
      load();
      const e = await labRemoteTakeError(idRef.current).catch(() => null);
      if (e) showRemoteEnd(e);
      refreshRemote();
    }, 2000);
  }, [load, refreshRemote, showRemoteEnd]);

  useEffect(() => {
    setResult(null);
    setError(null);
    setRemoteError(null);
    setNote(null);
    setStages([]);
    setLastLine("");
    setProgress(null);
    setSource(null);
    setCancelling(false);
    setHosted(false);
    setRunning(false);
    setLoading(true);
    load();
    // a run started earlier (before leaving the screen) is still going: follow it until it ends
    labAnalysisRunning()
      .then(async (id) => {
        if (id !== replayId) {
          // how a run ended while this screen was closed, or a saved job resumed in the background
          const e = await labRemoteTakeError(replayId).catch(() => null);
          if (e && onScreen(replayId)) showRemoteEnd(e);
          return;
        }
        const remote = await labRemoteRunning().catch(() => null);
        if (remote?.replay_id === replayId) {
          setHosted(true);
          setProgress({ replay_id: replayId, stage: remote.stage, line: "", position: remote.position ?? undefined });
          setStages([stageLabel(remote.stage)]);
        }
        follow();
      })
      .catch(() => {});
    return () => {
      window.clearInterval(followTimer.current);
      followTimer.current = undefined;
    };
  }, [replayId, load, follow, onScreen, showRemoteEnd]);

  useEffect(() => {
    const un = listen<LabProgress>("lab-progress", (ev) => {
      const p = ev.payload;
      if (!p || p.replay_id !== replayId) return;
      const label = stageLabel(p.stage);
      setLastLine(p.line);
      setProgress(p);
      if (p.source) setSource(p.source);
      else if (p.stage === "start" && p.line === "reused") setSource("reused");
      if (p.stage !== "done") setStages((s) => (s.includes(label) ? s : [...s, label]));
    });
    const quotaEvents = listen<LabQuota>("lab-quota", (ev) => {
      if (ev.payload) setQuota(ev.payload);
    });
    // a saved job picked up in the background
    const ended = listen<{ replay_id: string }>("lab-remote-ended", (ev) => {
      if (ev.payload?.replay_id !== replayId) return;
      load();
      labRemoteTakeError(replayId).then((e) => e && showRemoteEnd(e)).catch(() => {});
    });
    // progress of a run this screen didn't start: a saved job resumed in the background
    const resumed = listen<LabProgress>("lab-progress", (ev) => {
      const p = ev.payload;
      if (!p || p.replay_id !== replayId || p.stage === "done" || ownRun.current || followTimer.current !== undefined) return;
      labRemoteRunning()
        .then((r) => {
          if (r?.replay_id !== replayId || ownRun.current) return;
          setHosted(true);
          follow();
        })
        .catch(() => {});
    });
    return () => {
      un.then((f) => f());
      quotaEvents.then((f) => f());
      ended.then((f) => f());
      resumed.then((f) => f());
    };
  }, [replayId, load, follow, showRemoteEnd]);

  const run = useCallback(
    async (o: LabOptions) => {
      if (!dbPath) return;
      setRunning(true);
      ownRun.current = true;
      setError(null);
      setRemoteError(null);
      setNote(null);
      setStages([]);
      setLastLine("");
      try {
        const r = await runLabAnalysis(dbPath, replayId, { ...o, video_path: videoPath ?? undefined, title: o.title ?? title });
        setResult(r);
      } catch (e) {
        const msg = String(e);
        if (msg !== "Cancelled") setError(msg);
      } finally {
        ownRun.current = false;
        setRunning(false);
      }
    },
    [dbPath, replayId, videoPath, title]
  );

  const runRemote = useCallback(
    async (again: boolean) => {
      if (!dbPath) return;
      setRunning(true);
      setHosted(true);
      ownRun.current = true;
      setError(null);
      setRemoteError(null);
      setNote(null);
      setStages([]);
      setLastLine("");
      setProgress(null);
      setSource(null);
      setCancelling(false);
      try {
        const r = await runLabAnalysisRemote(dbPath, replayId, { again, video_path: videoPath ?? undefined });
        if (onScreen(replayId)) setResult(r);
      } catch (err) {
        // Kept for a screen that was closed. This one takes it only if it's still here to show it: otherwise
        // it stays until the user comes back to this replay.
        if (onScreen(replayId)) {
          labRemoteTakeError(replayId).catch(() => {});
          showRemoteEnd(toRemoteError(err));
        }
      } finally {
        ownRun.current = false;
        if (onScreen(replayId)) {
          setRunning(false);
          setHosted(false);
          setProgress(null);
          setCancelling(false);
          refreshRemote();
        }
      }
    },
    [dbPath, replayId, videoPath, onScreen, showRemoteEnd, refreshRemote]
  );

  const cancelPhase = useCallback(
    () => labRemoteCancelPhase(replayId).then((a) => a.phase).catch((): RemoteRunning["phase"] => "unknown"),
    [replayId]
  );

  const cancelRemote = useCallback(() => {
    setCancelling(true);
    cancelLabAnalysisRemote(replayId).catch(() => setCancelling(false));
  }, [replayId]);

  const signIn = useCallback(async () => {
    setSigningIn(true);
    setLoginError(null);
    setRemoteError(null);
    try {
      setAccount(await labSignIn());
      refreshRemote();
      // a job saved for this player, its upload still to delete, a result to collect
      labRemoteResume().catch(() => {});
    } catch (err) {
      const e = toRemoteError(err);
      if (e.code !== "cancelled") setLoginError(e);
    } finally {
      setSigningIn(false);
    }
  }, [refreshRemote]);

  const signInReopen = useCallback(() => {
    labSignInReopen().catch((e) => setLoginError(toRemoteError(e)));
  }, []);

  const signInCancel = useCallback(() => {
    labSignInCancel().catch(() => {});
  }, []);

  const signOut = useCallback(async () => {
    try {
      setAccount(await labSignOut());
      setQuota(null);
      setRemoteError(null);
      setNote(null);
    } catch (e) {
      setRemoteError(toRemoteError(e));
    }
  }, []);

  const openWeb = useCallback((page: "support" | "terms" | "privacy") => {
    labOpenWeb(page).catch((e) => setRemoteError(toRemoteError(e)));
  }, []);

  const cancel = useCallback(() => {
    cancelLabAnalysis(replayId).catch((e) => setError(String(e)));
  }, [replayId]);

  const remove = useCallback(async () => {
    try {
      await deleteLabResult(replayId);
      setResult(null);
    } catch (e) {
      setError(String(e));
    }
  }, [replayId]);

  return {
    result, loading, running, stages, lastLine, error, run, cancel, remove,
    config, account, quota, service, hosted, progress, source, cancelling, remoteError, note, signingIn, loginError,
    refreshRemote, runRemote, cancelPhase, cancelRemote, signIn, signInReopen, signInCancel, signOut, openWeb,
  };
}

/** At most every 30 s: focus changes come in bursts and each refresh is a few requests. */
const FOCUS_REFRESH_MS = 30_000;

// ── small pieces ─────────────────────────────────────────────────────────────

const sideTone = (s: string) => (s === "P1" ? "p1" : "p2") as "p1" | "p2";

function Seg<K extends string>({
  value, options, onChange, label,
}: { value: K; options: { key: K; label: string; disabled?: boolean; title?: string }[]; onChange: (k: K) => void; label: string }) {
  return (
    <div className="inline-flex rounded-lg bg-surface-2 border border-surface-4 p-0.5 gap-0.5" role="group" aria-label={label}>
      {options.map((o) => (
        <button
          key={o.key}
          aria-pressed={value === o.key}
          disabled={o.disabled}
          title={o.title}
          onClick={() => onChange(o.key)}
          className={`h-6 px-2 rounded-md text-[11px] font-medium whitespace-nowrap cursor-pointer transition-colors disabled:opacity-40 disabled:cursor-not-allowed
            ${value === o.key ? "bg-accent-purple/15 text-accent-purple" : "text-text-secondary hover:text-text-primary"}`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

function Select({ value, onChange, label, children }: { value: string; onChange: (v: string) => void; label: string; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-1 min-w-0">
      <span className="text-[10px] uppercase tracking-wider text-text-muted font-semibold">{label}</span>
      <select
        aria-label={label}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="h-7 rounded-md bg-surface-2 border border-surface-4 text-xs px-1.5 text-text-primary min-w-0"
      >
        {children}
      </select>
    </label>
  );
}

function OutcomeChip({ outcome, p }: { outcome: string; p?: number | null }) {
  const unsure = isUncertain(p);
  return (
    <span
      className={`inline-flex items-center gap-1 text-[10px] font-medium ${unsure ? "opacity-60 italic" : ""}`}
      title={p != null ? `${outcome} (p ${p.toFixed(2)})${unsure ? ", uncertain" : ""}` : outcome}
    >
      <i className="w-2 h-2 rounded-full shrink-0" style={{ background: OUTCOME_COLOR[outcome] ?? "#555570" }} />
      {outcome}
      {unsure && "?"}
    </span>
  );
}

function MoveName({ move, p }: { move: string; p: number }) {
  const unsure = isUncertain(p);
  return (
    <span
      className={`font-mono text-xs ${unsure ? "text-text-secondary italic" : "text-text-primary font-semibold"}`}
      title={`${move} (p ${p.toFixed(2)})${unsure ? ", uncertain" : ""}`}
      data-uncertain={unsure || undefined}
    >
      {move}
      {unsure && <span className="text-accent-gold not-italic">?</span>}
    </span>
  );
}

/** One clickable row: time, side, content. */
function Row({ t, side, onSeek, children, label }: { t: number; side: string; onSeek: (ms: number) => void; children: ReactNode; label: string }) {
  return (
    <button
      onClick={() => onSeek(seekBefore(t) * 1000)}
      aria-label={label}
      className="w-full flex items-center gap-2 px-2 py-1.5 rounded-md hover:bg-surface-3 text-left cursor-pointer"
    >
      <span className="font-mono text-[11px] text-text-secondary w-11 shrink-0 tabular-nums">{clockTenths(t)}</span>
      <Badge tone={sideTone(side)}>{side}</Badge>
      <div className="flex items-center gap-2 min-w-0 flex-1 flex-wrap">{children}</div>
    </button>
  );
}

function EventRow({ e, onSeek, extra }: { e: LabEvent; onSeek: (ms: number) => void; extra?: ReactNode }) {
  return (
    <Row t={e.t} side={e.side} onSeek={onSeek} label={`${clockTenths(e.t)} ${e.side} ${e.move} ${e.outcome}`}>
      {extra}
      <MoveName move={e.move} p={e.move_p} />
      <span className="text-[10px] text-text-muted">{e.category}</span>
      <span className="ml-auto flex items-center gap-1.5">
        {e.opener && <Badge tone="accent">opener</Badge>}
        <OutcomeChip outcome={e.outcome} p={e.outcome_p} />
      </span>
    </Row>
  );
}

const CAP = 300;
function Capped<T>({ items, render, empty }: { items: T[]; render: (x: T, i: number) => ReactNode; empty: string }) {
  const [all, setAll] = useState(false);
  useEffect(() => setAll(false), [items]);
  if (items.length === 0) return <div className="text-xs text-text-muted px-2 py-4 text-center">{empty}</div>;
  const shown = all ? items : items.slice(0, CAP);
  return (
    <div className="flex flex-col">
      {shown.map(render)}
      {items.length > shown.length && (
        <button onClick={() => setAll(true)} className="text-xs text-accent-purple hover:underline py-2 cursor-pointer">
          Show all {items.length}
        </button>
      )}
    </div>
  );
}

// ── sections ─────────────────────────────────────────────────────────────────

function OpenersList({ r, onSeek }: { r: LabResult; onSeek: (ms: number) => void }) {
  const [side, setSide] = useState<Side | "any">("any");
  const list = useMemo(() => [...r.openers].sort((a, b) => a.time_s - b.time_s).filter((o) => side === "any" || o.attacker === side), [r.openers, side]);
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <Seg label="Openers side" value={side} onChange={setSide} options={[{ key: "any", label: "Both" }, { key: "P1", label: `P1 ${r.characters.P1}` }, { key: "P2", label: `P2 ${r.characters.P2}` }]} />
        <span className="ml-auto text-[11px] text-text-muted" data-testid="lab-openers-count">{list.length} openers</span>
      </div>
      <Capped
        items={list}
        empty="No combo openers found."
        render={(o: LabOpener, i) => (
          <Row key={i} t={o.time_s} side={o.attacker} onSeek={onSeek} label={`${clockTenths(o.time_s)} ${o.attacker} ${o.move} opener`}>
            <MoveName move={o.move} p={o.move_p} />
            <span className="text-[10px] text-text-muted">{o.category}</span>
            {o.move_top3 && isUncertain(o.move_p) && (
              <span className="text-[10px] text-text-muted truncate" title="the model's other guesses">
                or {o.move_top3.slice(1).map(([m, p]) => `${m} ${Math.round(p * 100)}%`).join(", ")}
              </span>
            )}
            <span className="ml-auto">
              {o.hit_kind === "counter" || o.hit_kind === "punish" ? (
                <Badge tone="gold" className={isUncertain(o.hit_kind_p) ? "opacity-60 italic" : ""}>
                  {o.hit_kind}
                  {isUncertain(o.hit_kind_p) && "?"}
                </Badge>
              ) : null}
            </span>
          </Row>
        )}
      />
    </div>
  );
}

function MovesList({ r, onSeek, local }: { r: LabResult; onSeek: (ms: number) => void; local: boolean }) {
  const [side, setSide] = useState<Side | "any">("any");
  const [outcome, setOutcome] = useState<OutcomeWant>("any");
  const [move, setMove] = useState("");
  const events = useMemo(() => searchEvents(r), [r]);
  const moves = useMemo(() => moveChoices(events, side), [events, side]);
  const list = useMemo(() => freeFilter(events, { side, char: "", move, category: "", outcome }).map((h) => h.e), [events, side, move, outcome]);
  if (!r.transcript) return <NoTranscript local={local} />;
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-end gap-2 flex-wrap">
        <Seg label="Moves side" value={side} onChange={(s) => { setSide(s); setMove(""); }} options={[{ key: "any", label: "Both" }, { key: "P1", label: "P1" }, { key: "P2", label: "P2" }]} />
        <Select label="Move" value={move} onChange={setMove}>
          <option value="">any</option>
          {moves.map((m) => <option key={m} value={m}>{m}</option>)}
        </Select>
        <Select label="Outcome" value={outcome} onChange={(v) => setOutcome(v as OutcomeWant)}>
          <option value="any">any</option>
          <option value="connected">connected</option>
          {OUTCOMES.map((o) => <option key={o} value={o}>{o}</option>)}
        </Select>
        <span className="ml-auto text-[11px] text-text-muted" data-testid="lab-moves-count">{list.length} moves</span>
      </div>
      <Capped items={list} empty="No moves match." render={(e, i) => <EventRow key={i} e={e} onSeek={onSeek} />} />
    </div>
  );
}

type Preset = "move_outcome" | "counter_by" | "after_block" | "after_kd" | "free";

function SearchPanel({ r, onSeek }: { r: LabResult; onSeek: (ms: number) => void }) {
  const events = useMemo(() => searchEvents(r), [r]);
  const fallback = !r.transcript;
  const [preset, setPreset] = useState<Preset>("move_outcome");
  const [side, setSide] = useState<Side | "any">("P1");
  const [move, setMove] = useState("");
  const [outcome, setOutcome] = useState<OutcomeWant>("any");
  const [openersOnly, setOpenersOnly] = useState(false);
  const [char, setChar] = useState("");
  const [category, setCategory] = useState("");

  const sideOf: Side = side === "any" ? "P1" : side;
  const victimMoves = useMemo(() => moveChoices(events.filter((e) => e.side !== sideOf), "any"), [events, sideOf]);
  const sideMoves = useMemo(() => moveChoices(events, preset === "free" ? side : side), [events, side, preset]);
  const chars = useMemo(() => [...new Set(events.map((e) => e.char))].sort(), [events]);
  const categories = useMemo(() => [...new Set(events.map((e) => e.category))].sort(), [events]);

  const hits: Hit[] = useMemo(() => {
    switch (preset) {
      case "move_outcome": return presetMoveOutcome(events, { side, move, outcome, openersOnly });
      case "counter_by": return presetCounterBy(events, { victim: sideOf, move });
      case "after_block": return fallback ? [] : presetAfterBlock(events, sideOf);
      case "after_kd": return presetAfterKnockdown(events, r.states, sideOf);
      case "free": return freeFilter(events, { side, char, move, category, outcome });
    }
  }, [preset, events, side, sideOf, move, outcome, openersOnly, char, category, fallback, r.states]);

  const wakes = useMemo(() => (preset === "after_kd" && r.states ? wakeUps(r.states[sideOf] ?? []) : []), [preset, r.states, sideOf]);

  const pick = (p: Preset) => {
    setPreset(p);
    setMove("");
    if (p === "free") setSide("any");
    else if (p !== "move_outcome" && side === "any") setSide("P1");
  };
  const presets: { key: Preset; label: string; disabled?: boolean; title?: string }[] = [
    { key: "move_outcome", label: "Move + outcome" },
    { key: "counter_by", label: "Counter-hit by" },
    { key: "after_block", label: "After blocked", disabled: fallback, title: fallback ? "Needs the transcript" : "What a side does after one of its moves was blocked" },
    { key: "after_kd", label: "After knockdown", disabled: !r.states, title: !r.states ? "Needs the transcript (states)" : "A side's first move after waking up" },
    { key: "free", label: "Filters" },
  ];
  const sideOpts = (any: boolean) => [
    ...(any ? [{ key: "any" as const, label: "Both" }] : []),
    { key: "P1" as const, label: `P1 ${r.characters.P1}` },
    { key: "P2" as const, label: `P2 ${r.characters.P2}` },
  ];

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap gap-1" role="group" aria-label="Search preset">
        {presets.map((p) => (
          <button
            key={p.key}
            aria-pressed={preset === p.key}
            disabled={p.disabled}
            title={p.title}
            onClick={() => pick(p.key)}
            className={`h-7 px-2.5 rounded-md text-xs font-medium cursor-pointer transition-colors disabled:opacity-40 disabled:cursor-not-allowed
              ${preset === p.key ? "bg-accent-purple/15 text-accent-purple" : "text-text-secondary hover:bg-surface-3"}`}
          >
            {p.label}
          </button>
        ))}
      </div>

      <div className="flex items-end gap-2 flex-wrap rounded-lg bg-surface-2/60 border border-surface-4/60 p-2">
        {(preset === "move_outcome" || preset === "free") && (
          <Seg label="Side" value={side} onChange={(s) => { setSide(s); setMove(""); }} options={sideOpts(true)} />
        )}
        {(preset === "counter_by") && (
          <div className="flex flex-col gap-1">
            <span className="text-[10px] uppercase tracking-wider text-text-muted font-semibold">Counter-hit</span>
            <Seg label="Victim" value={sideOf} onChange={(s) => { setSide(s); setMove(""); }} options={sideOpts(false)} />
          </div>
        )}
        {(preset === "after_block" || preset === "after_kd") && (
          <Seg label="Side" value={sideOf} onChange={setSide} options={sideOpts(false)} />
        )}
        {preset === "free" && (
          <>
            <Select label="Character" value={char} onChange={setChar}>
              <option value="">any</option>
              {chars.map((c) => <option key={c} value={c}>{c}</option>)}
            </Select>
            <Select label="Category" value={category} onChange={setCategory}>
              <option value="">any</option>
              {categories.map((c) => <option key={c} value={c}>{c}</option>)}
            </Select>
          </>
        )}
        {(preset === "move_outcome" || preset === "counter_by" || preset === "free") && (
          <Select label="Move" value={move} onChange={setMove}>
            <option value="">any</option>
            {(preset === "counter_by" ? victimMoves : sideMoves).map((m) => <option key={m} value={m}>{m}</option>)}
          </Select>
        )}
        {(preset === "move_outcome" || preset === "free") && (
          <Select label="Outcome" value={outcome} onChange={(v) => setOutcome(v as OutcomeWant)}>
            <option value="any">any</option>
            <option value="connected">connected</option>
            {OUTCOMES.map((o) => <option key={o} value={o}>{o}</option>)}
          </Select>
        )}
        {preset === "move_outcome" && (
          <label className="flex items-center gap-1.5 text-xs text-text-secondary h-7 cursor-pointer">
            <input type="checkbox" checked={openersOnly} onChange={(e) => setOpenersOnly(e.target.checked)} className="accent-[#e8c840]" />
            openers only
          </label>
        )}
      </div>

      <div className="flex items-center gap-2 text-[11px] text-text-secondary">
        <span className="font-semibold text-text-primary" data-testid="lab-search-count">
          {hits.length} {hits.length === 1 ? "moment" : "moments"}
        </span>
        {preset === "after_kd" && (
          <span>· {wakes.length} wake-ups, {wakes.length - hits.length} without a move (waited / blocked)</span>
        )}
        {fallback && <span className="ml-auto text-text-muted">openers only</span>}
      </div>

      <Capped
        items={hits}
        empty="Nothing matches."
        render={(h, i) => (
          <EventRow
            key={i}
            e={h.e}
            onSeek={onSeek}
            extra={
              h.prev ? (
                <span className="text-[10px] text-text-muted font-mono">{h.prev.move} blocked →</span>
              ) : h.sinceWake !== undefined ? (
                <span className="text-[10px] text-text-muted font-mono">+{h.sinceWake}f</span>
              ) : undefined
            }
          />
        )}
      />
    </div>
  );
}

/** facts.md and the AI summary: headings, bullets, numbered lists and bold, as React nodes (never HTML);
 * every m:ss.s timestamp seeks the video. */
function Markdown({ text, onSeek }: { text: string; onSeek: (ms: number) => void }) {
  const inline = (s: string, key: string): ReactNode[] => {
    const out: ReactNode[] = [];
    const re = /(\*\*[^*]+\*\*|`[^`]+`|\b\d{1,2}:\d{2}\.\d\b)/g;
    let last = 0;
    let m: RegExpExecArray | null;
    let i = 0;
    while ((m = re.exec(s))) {
      if (m.index > last) out.push(s.slice(last, m.index));
      const tok = m[0];
      if (tok.startsWith("**")) out.push(<strong key={`${key}b${i}`} className="text-text-primary">{tok.slice(2, -2)}</strong>);
      else if (tok.startsWith("`")) out.push(<code key={`${key}c${i}`} className="font-mono text-[11px]">{tok.slice(1, -1)}</code>);
      else {
        const [mm, ss] = tok.split(":");
        const sec = Number(mm) * 60 + Number(ss);
        out.push(
          <button key={`${key}t${i}`} onClick={() => onSeek(seekBefore(sec) * 1000)} className="font-mono text-accent-purple hover:underline cursor-pointer" title={`Go to ${tok}`}>
            {tok}
          </button>
        );
      }
      last = m.index + tok.length;
      i++;
    }
    if (last < s.length) out.push(s.slice(last));
    return out;
  };
  const blocks: ReactNode[] = [];
  let bullets: ReactNode[] = [];
  let ordered = false;
  const flush = (k: number) => {
    if (bullets.length)
      blocks.push(
        ordered
          ? <ol key={`o${k}`} className="list-decimal pl-4 flex flex-col gap-1 marker:text-text-muted">{bullets}</ol>
          : <ul key={`u${k}`} className="list-disc pl-4 flex flex-col gap-1 marker:text-text-muted">{bullets}</ul>
      );
    bullets = [];
  };
  text.split(/\r?\n/).forEach((line, k) => {
    const l = line.trimEnd();
    const h = /^(#{1,4})\s+(.*)$/.exec(l);
    const b = /^\s*[-*]\s+(.*)$/.exec(l);
    const n = /^\s*\d{1,2}[.)]\s+(.*)$/.exec(l);
    if (b || n) {
      if (bullets.length && ordered !== !!n) flush(k);
      ordered = !!n;
      bullets.push(<li key={k}>{inline((b ?? n)![1], `l${k}`)}</li>);
      return;
    }
    flush(k);
    if (h) {
      const lvl = h[1].length;
      const cls = lvl === 1 ? "text-sm font-semibold text-text-primary" : lvl === 2 ? "text-[11px] font-semibold uppercase tracking-wider text-text-secondary mt-2" : "text-xs font-semibold text-text-primary mt-1";
      blocks.push(<div key={k} className={cls} role="heading" aria-level={lvl}>{inline(h[2], `h${k}`)}</div>);
    } else if (l.trim()) {
      blocks.push(<p key={k}>{inline(l, `p${k}`)}</p>);
    }
  });
  flush(-1);
  return <div className="flex flex-col gap-1.5 text-xs text-text-secondary leading-relaxed">{blocks}</div>;
}

async function copyText(text: string) {
  try {
    await navigator.clipboard.writeText(text);
    return;
  } catch {
    /* not a secure context: fall back */
  }
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.select();
  document.execCommand("copy");
  ta.remove();
}

const AI_LABEL: Record<LabLang, string> = {
  it: "Scritto dall'IA dai fatti della partita",
  en: "Written by AI from the match facts",
};
const AI_TEXT: Record<LabLang, { withheld: string; auth: string; details: string }> = {
  it: {
    withheld: "Il riepilogo dell'IA è stato nascosto: non corrispondeva ai fatti della partita. I fatti qui sotto restano validi.",
    auth: "Nessun riepilogo dell'IA: la chiave API di Claude manca o è stata rifiutata. Imposta ANTHROPIC_API_KEY nell'ambiente da cui parte l'app, poi riesegui.",
    details: "Dettagli",
  },
  en: {
    withheld: "The AI summary was withheld: it didn't match the match facts. The facts below are unaffected.",
    auth: "No AI summary: the Claude API key is missing or was refused. Set ANTHROPIC_API_KEY in the environment the app starts from, then run again.",
    details: "Details",
  },
};

/** The AI summary of one language (LAB_RESULT.md "ai_summary"): its text only when grounded, else why it
 * was withheld. Nothing when the result has none. */
function AiSummaryBlock({ r, lang, onSeek, local }: { r: LabResult; lang: LabLang; onSeek: (ms: number) => void; local: boolean }) {
  const ai = aiSummaryFor(r, lang);
  if (!ai) return null;
  if (ai.kind === "withheld") {
    return (
      <div className="flex flex-col gap-1.5 rounded-lg bg-surface-2/60 border border-surface-4/60 px-2.5 py-2 text-[11px] text-text-secondary" data-testid="lab-ai-withheld">
        <div className="flex items-start gap-2">
          <Sparkles size={13} className="mt-0.5 shrink-0 text-text-muted" />
          <span>
            {ai.authHint && local
              ? AI_TEXT[lang].auth
              : AI_TEXT[lang].withheld}
          </span>
        </div>
        {ai.errors.length > 0 && (
          <details className="pl-5">
            <summary className="cursor-pointer text-text-muted hover:text-text-primary">{AI_TEXT[lang].details} ({ai.errors.length})</summary>
            <ul className="list-disc pl-4 mt-1 flex flex-col gap-0.5 font-mono text-[10px] break-words">
              {ai.errors.map((e, i) => <li key={i}>{e}</li>)}
            </ul>
          </details>
        )}
      </div>
    );
  }
  return (
    <section className="flex flex-col gap-2 rounded-lg bg-accent-purple/5 border border-accent-purple/25 px-2.5 py-2" data-testid="lab-ai-summary" aria-label="AI summary">
      <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider font-semibold text-accent-purple">
        <Sparkles size={12} />
        <span>{AI_LABEL[lang]}</span>
        {ai.model && <span className="ml-auto normal-case tracking-normal font-normal text-text-muted">{ai.model}</span>}
      </div>
      <Markdown text={ai.text} onSeek={onSeek} />
      {ai.warnings.length > 0 && (
        <details className="text-[10px] text-text-muted">
          <summary className="cursor-pointer hover:text-text-primary">
            {lang === "it" ? "Avvisi del controllo" : "Check warnings"} ({ai.warnings.length})
          </summary>
          <ul className="list-disc pl-4 mt-1 flex flex-col gap-0.5 break-words">
            {ai.warnings.map((w, i) => <li key={i}>{w}</li>)}
          </ul>
        </details>
      )}
    </section>
  );
}

function SummaryPanel({ r, onSeek, local }: { r: LabResult; onSeek: (ms: number) => void; local: boolean }) {
  const [lang, setLang] = useState<LabLang>(initialLang);
  const [copied, setCopied] = useState(false);
  const md = r.facts?.md?.[lang] ?? r.facts?.md?.en ?? r.facts?.md?.it ?? "";
  const prompt = r.prompts?.[lang] ?? r.prompts?.en ?? "";
  const hasAi = !!r.ai_summary;
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-2 flex-wrap">
        <Seg label="Summary language" value={lang} onChange={(l) => { setLang(l); saveLang(l); }} options={[{ key: "en", label: "English" }, { key: "it", label: "Italiano" }]} />
        <Button
          size="sm"
          icon={copied ? Check : Copy}
          className="ml-auto"
          disabled={!prompt}
          onClick={async () => {
            await copyText(prompt);
            setCopied(true);
            window.setTimeout(() => setCopied(false), 1500);
          }}
        >
          {copied ? "Copied" : "Copy prompt"}
        </Button>
      </div>
      <AiSummaryBlock r={r} lang={lang} onSeek={onSeek} local={local} />
      <div className="flex items-start gap-2 rounded-lg bg-surface-2/60 border border-surface-4/60 px-2.5 py-2 text-[11px] text-text-secondary">
        <Info size={13} className="mt-0.5 shrink-0 text-text-muted" />
        {hasAi ? (
          <span>
            The facts the models read from the video, the source of truth for the AI summary above; no AI wrote them.
          </span>
        ) : (
          <span>
            {local
              ? "These are the facts the models read from the video; no AI wrote them and nothing is sent anywhere."
              : "These are the facts the models read from the video; no AI wrote them."}{" "}
            "Copy prompt" copies them with instructions, to paste into a chatbot of your choice for a written summary.
          </span>
        )}
      </div>
      {md ? <Markdown text={md} onSeek={onSeek} /> : <div className="text-xs text-text-muted">No summary in this result.</div>}
    </div>
  );
}

function NoTranscript({ local }: { local: boolean }) {
  if (!local) {
    return (
      <div className="flex items-start gap-2 rounded-lg bg-surface-2/60 border border-surface-4/60 px-2.5 py-2 text-[11px] text-text-secondary" data-testid="lab-no-transcript">
        <Info size={13} className="mt-0.5 shrink-0 text-text-muted" />
        <span>This analysis has no move transcript. Openers, search over the openers and the summary still work.</span>
      </div>
    );
  }
  return (
    <div className="flex items-start gap-2 rounded-lg bg-accent-gold/10 border border-accent-gold/30 px-2.5 py-2 text-[11px] text-text-secondary" data-testid="lab-no-transcript">
      <Cpu size={13} className="mt-0.5 shrink-0 text-accent-gold" />
      <span>
        No move transcript in this result: it needs a GPU (CUDA), so it was skipped. Openers, search over the openers and the
        summary still work. Run again with Transcript "On" on a PC with a GPU for every move.
      </span>
    </div>
  );
}

// ── the panel ────────────────────────────────────────────────────────────────

type Section = "openers" | "moves" | "search" | "summary";

export function LabPanel({ lab, onSeek, canRun }: { lab: LabState; onSeek: (ms: number) => void; canRun: boolean }) {
  const [section, setSection] = useState<Section>("openers");
  const [transcript, setTranscript] = useState<"auto" | "on" | "off">("auto");
  const [light, setLight] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [modelsDir, setModelsDir] = useState(loadModelsDir);
  const [aiSummary, setAiSummary] = useState(false);
  const [modeChoice, setModeChoice] = useState<"local" | "hosted">("local");
  const r = lab.result;
  const localAllowed = !!lab.config?.local_models;
  const mode = localAllowed ? modeChoice : "hosted";

  // The Lab tab is where the account matters: nothing goes on the network before it's opened
  const { refreshRemote } = lab;
  const lastRefresh = useRef(0);
  useEffect(() => {
    lastRefresh.current = Date.now();
    refreshRemote();
    const onFocus = () => {
      if (Date.now() - lastRefresh.current < FOCUS_REFRESH_MS) return;
      lastRefresh.current = Date.now();
      refreshRemote();
    };
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, [refreshRemote]);

  const start = () => lab.run({ transcript, light, llm: aiSummary ? "claude" : "none", models_dir: modelsDir.trim() || undefined });

  const notInstalled = lab.error?.startsWith(NOT_INSTALLED);

  const runForm = (
    <div className="flex flex-col gap-2.5">
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-[11px] text-text-secondary">Transcript</span>
        <Seg label="Transcript" value={transcript} onChange={setTranscript} options={[
          { key: "auto", label: "Auto", title: "Only with a GPU" },
          { key: "on", label: "On", title: "Every move (slow without a GPU)" },
          { key: "off", label: "Off", title: "Openers and summary only" },
        ]} />
        <label className="flex items-center gap-1.5 text-[11px] text-text-secondary cursor-pointer" title="Smaller models: faster on a CPU, a little less accurate">
          <input type="checkbox" checked={light} onChange={(e) => setLight(e.target.checked)} className="accent-[#e8c840]" />
          Light
        </label>
        <label
          className="flex items-center gap-1.5 text-[11px] text-text-secondary cursor-pointer"
          title="Also a written summary and coaching notes by Claude, from the facts only (sends the facts to Anthropic; needs ANTHROPIC_API_KEY in the environment)"
        >
          <input type="checkbox" checked={aiSummary} onChange={(e) => setAiSummary(e.target.checked)} className="accent-[#e8c840]" />
          AI summary (Claude)
        </label>
        <button onClick={() => setShowSettings((s) => !s)} className="ml-auto text-text-muted hover:text-text-primary cursor-pointer" aria-label="Lab settings" title="Models folder">
          <Settings2 size={14} />
        </button>
      </div>
      {showSettings && (
        <label className="flex flex-col gap-1">
          <span className="text-[10px] uppercase tracking-wider text-text-muted font-semibold">Models folder</span>
          <input
            value={modelsDir}
            onChange={(e) => setModelsDir(e.target.value)}
            onBlur={() => saveModelsDir(modelsDir)}
            placeholder="default: SPAGHETTI_LAB_MODELS, else the research data folder"
            className="h-7 rounded-md bg-surface-2 border border-surface-4 text-xs px-2 text-text-primary"
          />
        </label>
      )}
      <Button variant="primary" size="sm" icon={r ? RefreshCw : Play} onClick={start} disabled={!canRun} title={canRun ? undefined : "Needs the local video"}>
        {r ? "Run again" : "Run lab analysis"}
      </Button>
    </div>
  );

  const progress = (
    <div className="flex flex-col gap-2 rounded-lg bg-surface-2 border border-surface-4 p-3" data-testid="lab-progress">
      <div className="flex items-center gap-2">
        <Spinner size={14} className="text-accent-purple" />
        <span className="text-xs font-medium">Lab analysis running…</span>
        <Button size="sm" variant="ghost" icon={Square} className="ml-auto" onClick={lab.cancel}>Cancel</Button>
      </div>
      <ol className="flex flex-col gap-1">
        {stageSteps(transcript !== "off", aiSummary).map((s) => {
          const seen = lab.stages.includes(s);
          const current = lab.stages[lab.stages.length - 1] === s;
          return (
            <li key={s} className={`flex items-center gap-2 text-[11px] ${current ? "text-text-primary" : seen ? "text-text-secondary" : "text-text-muted"}`}>
              {current ? <Spinner size={10} className="text-accent-purple" /> : seen ? <Check size={11} className="text-accent-green" /> : <i className="w-2.5 h-2.5 rounded-full border border-surface-4" />}
              {s}
            </li>
          );
        })}
      </ol>
      {lab.lastLine && <div className="text-[10px] font-mono text-text-muted truncate">{lab.lastLine}</div>}
    </div>
  );

  const error = lab.error && (
    <div className={`flex items-start gap-2 rounded-lg px-2.5 py-2 text-[11px] border ${notInstalled ? "bg-surface-2 border-surface-4 text-text-secondary" : "bg-p1/10 border-p1/30 text-p1-light"}`} role="alert">
      <AlertTriangle size={13} className="mt-0.5 shrink-0" />
      <div className="min-w-0">
        <div className="font-semibold">{notInstalled ? "The lab models aren't installed on this PC" : "The lab analysis failed"}</div>
        <div className="whitespace-pre-wrap break-words mt-0.5 max-h-40 overflow-y-auto">{lab.error}</div>
      </div>
    </div>
  );

  const modeSwitch = localAllowed && !lab.running && (
    <div className="flex items-center gap-2">
      <span className="text-[11px] text-text-secondary">Run on</span>
      <Seg label="Analysis mode" value={mode} onChange={setModeChoice} options={[
        { key: "local", label: "local models", title: "This PC's research models: no sign-in, nothing spent" },
        { key: "hosted", label: "server", title: "The Spaghetti Showdown servers: sign-in, spends an analysis" },
      ]} />
    </div>
  );
  const form = (
    <div className="flex flex-col gap-2.5">
      {modeSwitch}
      {mode === "local" ? runForm : <HostedForm lab={lab} canRun={canRun} />}
    </div>
  );
  const runningCard = lab.hosted ? <HostedProgress lab={lab} /> : progress;
  const remoteNotes = (
    <>
      {lab.remoteError && <RemoteErrorNote e={lab.remoteError} lab={lab} />}
      {lab.note && <div className="text-[11px] text-text-secondary" role="status" data-testid="lab-note">{lab.note}</div>}
    </>
  );
  const showAccount = mode === "hosted" && lab.account?.signed_in;

  if (lab.loading || !lab.config) {
    return <div className="flex items-center gap-2 text-xs text-text-secondary p-4"><Spinner size={14} /> Loading the lab result…</div>;
  }

  if (!r) {
    return (
      <div className="flex flex-col gap-3">
        <EmptyState icon={FlaskConical} title="Lab analysis">
          {localAllowed
            ? "Research models read the video: combo openers, every move and its outcome (with a GPU), a search over them and a summary of the match."
            : "Research models read the video: combo openers, a search over them and a summary of the match."}{" "}
          Experimental: treat it as what the model saw, not a full record.
        </EmptyState>
        {showAccount && <AccountBar lab={lab} />}
        {lab.running ? runningCard : form}
        {error}
        {remoteNotes}
      </div>
    );
  }

  const nEvents = r.transcript?.events.length ?? 0;
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-2 flex-wrap text-[11px] text-text-secondary">
        <FlaskConical size={14} className="text-accent-purple" />
        <span><span className="text-p1-light font-semibold">{r.characters.P1}</span> vs <span className="text-p2-light font-semibold">{r.characters.P2}</span></span>
        <span className="text-text-muted">· {r.openers.length} openers{r.transcript ? ` · ${nEvents} moves` : ""}</span>
        {!lab.running && (
          <span className="ml-auto flex items-center gap-0.5">
            <Button size="sm" variant="ghost" icon={Settings2} onClick={() => setShowSettings((s) => !s)} aria-label="Run options">
              <span className="hidden sm:inline">{mode === "local" ? "Run again" : lab.account?.signed_in ? "Analyse again" : "Sign in to analyse again"}</span>
            </Button>
            <Button size="sm" variant="danger" icon={Trash2} onClick={lab.remove} aria-label="Delete the lab result" title="Delete the lab result" />
          </span>
        )}
      </div>
      {showAccount && <AccountBar lab={lab} />}
      {lab.running ? runningCard : (showSettings || !!lab.remoteError) && <div className="rounded-lg bg-surface-2/60 border border-surface-4/60 p-2.5">{form}</div>}
      {error}
      {remoteNotes}
      {!r.transcript && section !== "moves" && <NoTranscript local={localAllowed} />}

      <div className="flex flex-wrap gap-1" role="group" aria-label="Lab section">
        {([
          { key: "openers", label: "Openers", icon: Swords, n: r.openers.length },
          { key: "moves", label: "Moves", icon: ListTree, n: nEvents },
          { key: "search", label: "Search", icon: Search },
          { key: "summary", label: "Summary", icon: FileText },
        ] as { key: Section; label: string; icon: typeof Swords; n?: number }[]).map((s) => (
          <button
            key={s.key}
            aria-pressed={section === s.key}
            onClick={() => setSection(s.key)}
            className={`inline-flex items-center gap-1.5 h-7 px-2.5 rounded-md text-xs font-medium cursor-pointer transition-colors
              ${section === s.key ? "bg-accent-purple/15 text-accent-purple" : "text-text-secondary hover:bg-surface-3"}`}
          >
            <s.icon size={13} />
            {s.label}
            {s.n ? <span className="text-[10px] tabular-nums text-text-muted">{s.n}</span> : null}
          </button>
        ))}
      </div>

      {section === "openers" && <OpenersList r={r} onSeek={onSeek} />}
      {section === "moves" && <MovesList r={r} onSeek={onSeek} local={localAllowed} />}
      {section === "search" && <SearchPanel r={r} onSeek={onSeek} />}
      {section === "summary" && <SummaryPanel r={r} onSeek={onSeek} local={localAllowed} />}
    </div>
  );
}
