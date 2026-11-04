// The hosted side of the Lab tab: sign-in panel, account and quota line, the form that spends an analysis,
// the progress of a run on the server and its errors. State and commands are in useLab (LabPanel.tsx); the
// server decides everything about money, and this file only reports it (docs/desktop-login.md).
import { useState } from "react";
import { AlertTriangle, Check, ExternalLink, Info, LogIn, LogOut, Play, RefreshCw, Square } from "lucide-react";
import { Button, Spinner } from "./ui";
import {
  NOTICE_TEXT, acceptNotice, cancelWarning, closeIsSafe, hostedSteps, initialLang, noticeAccepted, quotaEmpty, quotaLine, resetDate, spendNote,
  stageLabel, type RemoteError, type RemoteRunning,
} from "../lib/lab";
import type { LabState } from "./LabPanel";

function Links({ lab }: { lab: LabState }) {
  const link = "text-accent-purple hover:underline cursor-pointer";
  return (
    <span className="inline-flex items-center gap-2">
      <button className={link} onClick={() => lab.openWeb("terms")}>Terms</button>
      <button className={link} onClick={() => lab.openWeb("privacy")}>Privacy</button>
    </span>
  );
}

function SupportButton({ lab }: { lab: LabState }) {
  return (
    <Button size="sm" icon={ExternalLink} onClick={() => lab.openWeb("support")}>
      Get credits on the website
    </Button>
  );
}

const ServiceDown = () => (
  <div className="flex items-start gap-2 rounded-lg bg-surface-2 border border-surface-4 px-2.5 py-2 text-[11px] text-text-secondary" role="status" data-testid="lab-service-down">
    <Info size={13} className="mt-0.5 shrink-0 text-text-muted" />
    <span>The analysis is not available right now. Please try again later; results you already have stay readable.</span>
  </div>
);

/** Signed out: what the sign-in is for and what leaves this PC. Nothing else in the app needs an account. */
export function SignInPanel({ lab }: { lab: LabState }) {
  const expired = lab.remoteError?.code === "signed_out";
  return (
    <div className="flex flex-col gap-2 rounded-lg bg-surface-2 border border-surface-4 p-3 text-[11px] text-text-secondary" data-testid="lab-sign-in">
      <div className="text-xs font-semibold text-text-primary">Sign in to analyse on the server</div>
      <p>
        The analysis runs on the Spaghetti Showdown servers and needs a spaghettishowdown.it account. You sign in with Discord in
        your browser; the app never sees your password. Nothing else in the app needs an account.
      </p>
      <p>
        <strong className="text-text-primary">What leaves this PC:</strong> a 720p copy of the video, without audio, goes to the
        server for the analysis, and the app deletes it afterwards (if it's closed or offline then, the next time it runs signed
        in). The result comes back to this PC.
      </p>
      <p>
        Each account has free analyses every week, then credits; contributions and credits are handled on the website, under the
        Terms. <Links lab={lab} />
      </p>
      {lab.signingIn ? (
        <div className="flex items-center gap-2 flex-wrap" role="status">
          <Spinner size={12} className="text-accent-purple" />
          <span>Waiting for the browser…</span>
          <Button size="sm" variant="ghost" onClick={lab.signInReopen}>Open the page again</Button>
          <Button size="sm" variant="ghost" onClick={lab.signInCancel}>Cancel sign-in</Button>
        </div>
      ) : (
        <div>
          <Button variant="primary" size="sm" icon={LogIn} onClick={lab.signIn}>Sign in with spaghettishowdown.it</Button>
        </div>
      )}
      {expired && !lab.signingIn && <div role="status">You've been signed out. Please sign in again.</div>}
      {lab.account?.store_unavailable && (
        <div role="status" data-testid="lab-store-unavailable">
          The system's credential store can't be read right now, so an earlier sign-in can't be checked. It isn't lost: try again in a moment.
        </div>
      )}
      {lab.loginError && (
        <div className="flex items-start gap-2 text-p1-light" role="alert" data-testid="lab-login-error">
          <AlertTriangle size={13} className="mt-0.5 shrink-0" />
          <span>{lab.loginError.message}</span>
        </div>
      )}
      {lab.service && !lab.service.available && <ServiceDown />}
    </div>
  );
}

/** Who is signed in and what's left. */
export function AccountBar({ lab }: { lab: LabState }) {
  const a = lab.account;
  if (!a?.signed_in) return null;
  const name = a.nickname ?? "Signed in";
  const reset = resetDate(lab.quota?.week_resets_at);
  return (
    <div className="flex items-center gap-2 flex-wrap text-[11px] text-text-secondary" data-testid="lab-account">
      {a.avatar ? (
        <img src={a.avatar} alt="" className="w-5 h-5 rounded-full" referrerPolicy="no-referrer" />
      ) : (
        <span className="w-5 h-5 rounded-full bg-surface-4 text-text-primary flex items-center justify-center text-[10px] font-semibold" aria-hidden>
          {name.slice(0, 1).toUpperCase()}
        </span>
      )}
      <span className="font-medium text-text-primary" data-testid="lab-nickname">{name}</span>
      {lab.quota && (
        <span data-testid="lab-quota" title={reset ? `Free analyses come back on ${reset} (Europe/Rome)` : undefined}>
          · {quotaLine(lab.quota)}
        </span>
      )}
      {!lab.running && (
        <Button size="sm" variant="ghost" icon={LogOut} className="ml-auto" onClick={lab.signOut}>Sign out</Button>
      )}
      {a.session_only && (
        <span className="w-full text-text-muted">The system's credential store isn't available, so this sign-in ends when the app closes.</span>
      )}
    </div>
  );
}

/** Ended on these, the server may hold a job for this replay: the next press goes on with it, and spends nothing new. */
const WAITING_ON_SERVER = ["start_unconfirmed", "result_pending", "result_write", "cancel_unconfirmed"];

/** Starting a run: says what it will use, confirms a repeat, shows the upload notice once. */
export function HostedForm({ lab, canRun }: { lab: LabState; canRun: boolean }) {
  const [ask, setAsk] = useState<null | "again" | "notice">(null);
  // not known yet: no sign-in panel flashing past a signed-in user
  if (!lab.account) return null;
  if (!lab.account.signed_in) return <SignInPanel lab={lab} />;

  const again = !!lab.result;
  const waiting = WAITING_ON_SERVER.includes(lab.remoteError?.code ?? "");
  const go = () => {
    setAsk(null);
    lab.runRemote(again);
  };
  const afterConfirm = () => (noticeAccepted() ? go() : setAsk("notice"));
  const down = !!lab.service && !lab.service.available;
  const empty = quotaEmpty(lab.quota);
  const spend = spendNote(lab.quota);
  const reset = resetDate(lab.quota?.week_resets_at);

  return (
    <div className="flex flex-col gap-2 text-[11px] text-text-secondary" data-testid="lab-hosted-form">
      {down && <ServiceDown />}
      {!down && empty && !waiting && (
        <div className="flex flex-col gap-1.5 rounded-lg bg-surface-2 border border-surface-4 px-2.5 py-2" data-testid="lab-exhausted">
          <span>No analyses left this week{reset ? `: free analyses come back on ${reset}` : ""}. You can also get credits on the website.</span>
          <div><SupportButton lab={lab} /></div>
        </div>
      )}
      {!down && !empty && spend && !waiting && <div data-testid="lab-spend">{spend} The server decides when it starts.</div>}

      {ask === "again" && (
        <div className="flex flex-col gap-2 rounded-lg bg-accent-gold/10 border border-accent-gold/30 px-2.5 py-2" data-testid="lab-again-confirm">
          <span>This replay already has a result. Analysing it again spends another analysis, even for the same video. The current result stays until the new one is ready.</span>
          <div className="flex items-center gap-2">
            <Button size="sm" variant="primary" onClick={afterConfirm}>Spend an analysis</Button>
            <Button size="sm" variant="ghost" onClick={() => setAsk(null)}>Keep this result</Button>
          </div>
        </div>
      )}
      {ask === "notice" && (
        <div className="flex flex-col gap-2 rounded-lg bg-surface-2 border border-surface-4 p-3" data-testid="lab-notice">
          <div className="text-xs font-semibold text-text-primary">Before the first upload</div>
          {NOTICE_TEXT[initialLang()].map((p, i) => <p key={i}>{p}</p>)}
          <div><Links lab={lab} /></div>
          <div className="flex items-center gap-2">
            <Button size="sm" variant="primary" icon={Check} onClick={() => { acceptNotice(); go(); }}>I understand, analyse</Button>
            <Button size="sm" variant="ghost" onClick={() => setAsk(null)}>Not now</Button>
          </div>
        </div>
      )}
      {!ask && (
        <div>
          <Button
            variant="primary"
            size="sm"
            icon={again || waiting ? RefreshCw : Play}
            disabled={!waiting && (!canRun || down || empty)}
            title={canRun || waiting ? undefined : "Needs the local video"}
            onClick={() => (waiting ? go() : again ? setAsk("again") : afterConfirm())}
          >
            {waiting ? "Try again" : again ? "Analyse again" : "Analyse this replay"}
          </Button>
        </div>
      )}
      {!ask && <div className="text-text-muted">An analysis usually takes 5 to 15 minutes. You can keep using the app. Once it has started on the server you can also close the app: the result is collected when it opens again.</div>}
    </div>
  );
}

const SOURCE_TEXT: Record<string, string> = {
  free: "Using a free analysis.",
  credit: "Using a credit.",
  reused: "Nothing spent: the server already had this analysis.",
};

/** A run on the server: its steps, upload and queue, and the cancel with what it would cost. */
export function HostedProgress({ lab }: { lab: LabState }) {
  /** null: no question. The phase is read from the server when Cancel is pressed, not from the last poll. */
  const [asking, setAsking] = useState<RemoteRunning["phase"] | "checking" | null>(null);
  const p = lab.progress;
  const askCancel = async () => {
    setAsking("checking");
    const phase = await lab.cancelPhase();
    if (phase === "before_charge") {
      setAsking(null);
      lab.cancelRemote();
    } else {
      setAsking(phase);
    }
  };
  const steps = hostedSteps(lab.service?.capabilities);
  const shown = [...steps, ...lab.stages.filter((s) => !steps.includes(s) && s !== stageLabel("start"))];
  const current = lab.stages[lab.stages.length - 1];
  const showBar = (p?.stage === "transcode" || p?.stage === "upload") && p.percent != null;

  return (
    <div className="flex flex-col gap-2 rounded-lg bg-surface-2 border border-surface-4 p-3" data-testid="lab-progress">
      <div className="flex items-center gap-2">
        <Spinner size={14} className="text-accent-purple" />
        <span className="text-xs font-medium">{lab.cancelling ? "Cancelling…" : "Analysing on the server…"}</span>
        {!lab.cancelling && !asking && (
          <Button size="sm" variant="ghost" icon={Square} className="ml-auto" onClick={askCancel}>
            Cancel
          </Button>
        )}
      </div>

      {asking === "checking" && !lab.cancelling && <div className="text-[11px] text-text-secondary" role="status">Checking the analysis…</div>}
      {asking && asking !== "checking" && !lab.cancelling && (
        <div className={`flex flex-col gap-2 rounded-lg px-2.5 py-2 text-[11px] border ${asking === "queued" ? "bg-surface-3 border-surface-4" : "bg-accent-gold/10 border-accent-gold/30"}`} data-testid="lab-cancel-confirm">
          <span>{cancelWarning(asking)}</span>
          <div className="flex items-center gap-2">
            <Button size="sm" variant={asking === "queued" ? "primary" : "danger"} onClick={() => { setAsking(null); lab.cancelRemote(); }}>Cancel the analysis</Button>
            <Button size="sm" variant="ghost" onClick={() => setAsking(null)}>Keep it going</Button>
          </div>
        </div>
      )}

      <ol className="flex flex-col gap-1">
        {shown.map((s) => {
          const seen = lab.stages.includes(s);
          const isCurrent = current === s;
          return (
            <li key={s} className={`flex items-center gap-2 text-[11px] ${isCurrent ? "text-text-primary" : seen ? "text-text-secondary" : "text-text-muted"}`}>
              {isCurrent ? <Spinner size={10} className="text-accent-purple" /> : seen ? <Check size={11} className="text-accent-green" /> : <i className="w-2.5 h-2.5 rounded-full border border-surface-4" />}
              {s}
            </li>
          );
        })}
      </ol>

      {showBar && (
        <div className="flex items-center gap-2" data-testid="lab-percent">
          <div className="h-1.5 flex-1 rounded-full bg-surface-4 overflow-hidden">
            <div className="h-full bg-accent-purple" style={{ width: `${Math.min(100, p!.percent!)}%` }} />
          </div>
          <span className="text-[10px] font-mono text-text-muted tabular-nums">{Math.round(p!.percent!)}%</span>
        </div>
      )}
      {p?.stage === "upload" && p.percent === 100 && <div className="text-[11px] text-text-secondary">Finishing the upload: the server is storing the file.</div>}
      {p?.stage === "queued" && p.position != null && (
        <div className="text-[11px] text-text-secondary" data-testid="lab-queue-position">Position {p.position} in the queue</div>
      )}
      {lab.source && SOURCE_TEXT[lab.source] && <div className="text-[11px] text-text-secondary" data-testid="lab-source">{SOURCE_TEXT[lab.source]}</div>}
      <div className="text-[10px] text-text-muted">
        {closeIsSafe(p?.stage)
          ? "A run can take a while on the shared server. You can leave this screen or close the app."
          : "You can leave this screen, but keep the app open until the video is on the server."}
      </div>
    </div>
  );
}

const TITLE: Record<string, string> = {
  quota_exhausted: "No analyses left this week",
  unavailable: "Not available right now",
  queue_full: "The analysis queue is full",
  one_at_a_time: "One analysis at a time",
  too_long: "This video can't be analysed",
  too_small: "This video can't be analysed",
  too_big: "This video can't be analysed",
  no_video: "This video can't be analysed",
  storage_full: "The upload was refused",
  banned: "The upload was refused",
  upload_failed: "The upload failed",
  start_failed: "The analysis didn't start",
  job_failed: "The analysis failed",
  expired: "The analysis is gone",
  bad_result: "The result couldn't be read",
  result_pending: "The result is still to come",
  result_write: "The result couldn't be saved",
  start_unconfirmed: "The start wasn't confirmed",
  cancel_unconfirmed: "The cancel wasn't confirmed",
  scope: "Please report this",
  network: "No connection",
  rate_limited: "Too many requests",
  credential_store: "The sign-in couldn't be read",
  busy: "Already running",
};

/** How a hosted run ended, when it didn't end well. A sign-out is shown by the sign-in panel. */
export function RemoteErrorNote({ e, lab }: { e: RemoteError; lab: LabState }) {
  if (e.code === "signed_out") return null;
  const reset = resetDate(e.quota?.week_resets_at);
  return (
    <div className="flex items-start gap-2 rounded-lg px-2.5 py-2 text-[11px] border bg-p1/10 border-p1/30 text-p1-light" role="alert" data-testid="lab-remote-error" data-code={e.code}>
      <AlertTriangle size={13} className="mt-0.5 shrink-0" />
      <div className="min-w-0 flex flex-col gap-1.5">
        <div className="font-semibold">{TITLE[e.code] ?? "The analysis didn't finish"}</div>
        <div className="whitespace-pre-wrap break-words">{e.message}</div>
        {e.code === "quota_exhausted" && (
          <>
            {reset && <div>Free analyses come back on {reset}.</div>}
            <div><SupportButton lab={lab} /></div>
          </>
        )}
        {/* true only for the call that gave it back; false says nothing either way, the quota line does */}
        {e.refunded === true && <div data-testid="lab-refunded">Your analysis was given back.</div>}
      </div>
    </div>
  );
}
