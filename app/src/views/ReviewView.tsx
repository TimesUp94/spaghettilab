// The review screen: the video fitted to the space on the left with the timeline under it,
// the results in a tabbed panel on the right (below the video on narrow windows).
import { useEffect, useMemo, useState, type ReactNode } from "react";
import {
  ArrowLeft, RefreshCw, Share2, Save, LayoutDashboard, ListOrdered, Sparkles, Activity,
  StickyNote, FolderOpen, ChevronDown, ScanLine,
} from "lucide-react";
import type {
  DamageEvent, Drawing, FrameDataPoint, Highlight, Match, MatchStats, Note, Replay, RoundResult, VideoMode,
} from "../types";
import { VideoPlayer } from "../components/VideoPlayer";
import { ReviewTimeline } from "../components/ReviewTimeline";
import { MatchOverview } from "../components/MatchOverview";
import { MatchList } from "../components/MatchList";
import { RoundBreakdown } from "../components/RoundBreakdown";
import { HighlightsPanel } from "../components/Highlights";
import { DamageLog } from "../components/DamageLog";
import { NotesPanel } from "../components/NotesPanel";
import { Button, IconButton, Spinner, Tabs, type TabDef } from "../components/ui";
import { formatClock, replayTitle, setInfo } from "../lib/matches";
import { useMediaQuery } from "../lib/useMediaQuery";

export type ReviewTab = "overview" | "rounds" | "highlights" | "damage" | "notes";

export interface ReviewData {
  replay: Replay;
  frameData: FrameDataPoint[];
  damageEvents: DamageEvent[];
  rounds: RoundResult[];
  matches: Match[];
  stats: MatchStats;
  highlights: Highlight[];
  notes: Note[];
  drawings: Drawing[];
}

interface Props {
  data: ReviewData;
  displayStats: MatchStats;
  selectedMatch: Match | null;
  onSelectMatch: (index: number | null) => void;
  videoSrc: string | null;
  videoMode: VideoMode;
  seekToMs: number | null;
  onSeek: (ms: number) => void;
  onSeeked: () => void;
  currentTimeMs: number;
  onTimeUpdate: (ms: number) => void;
  onRoundSeek: (ms: number) => void;
  onExportClip: (startMs: number, endMs: number) => void;
  onOverrideWinner: (roundIndex: number, winner: string) => void;
  onLocateVideo: () => void;
  onAddNote: (ms: number, text: string) => void;
  onUpdateNote: (id: number, text: string) => void;
  onDeleteNote: (id: number) => void;
  onSaveDrawing: (ms: number, strokes: string) => void;
  onDeleteDrawing: (id: number) => void;
  onBack: () => void;
  onReload: () => void;
  onReanalyze?: () => void;
  busy: { loading: boolean; reanalyzing: boolean; reanalyzeText?: ReactNode };
  share: {
    session: "spag" | "spagz" | null;
    onSave: () => void;
    saving: boolean;
    onExportSpag: () => void;
    onExportSpagz: () => void;
    exporting: boolean;
  };
}

function ShareMenu({ share }: { share: Props["share"] }) {
  const [open, setOpen] = useState(false);
  if (share.session) {
    return (
      <Button size="sm" variant="primary" icon={Save} loading={share.saving} onClick={share.onSave}>
        Save .{share.session}
      </Button>
    );
  }
  return (
    <div className="relative">
      <Button size="sm" variant="secondary" icon={Share2} loading={share.exporting} onClick={() => setOpen((o) => !o)}>
        <span className="hidden sm:inline">Share</span>
        <ChevronDown size={13} />
      </Button>
      {open && (
        <>
          <div className="fixed inset-0 z-30" onClick={() => setOpen(false)} />
          <div className="absolute right-0 top-9 z-40 w-72 rounded-xl bg-surface-2 border border-surface-4 shadow-2xl p-1.5">
            {[
              { ext: "spag", text: "Analysis with the video inside: one file, plays anywhere.", run: share.onExportSpag },
              { ext: "spagz", text: "Analysis only, small: the other person needs the video (or its link).", run: share.onExportSpagz },
            ].map((o) => (
              <button
                key={o.ext}
                onClick={() => {
                  setOpen(false);
                  o.run();
                }}
                className="w-full text-left px-3 py-2 rounded-lg hover:bg-surface-3 cursor-pointer"
              >
                <div className="text-sm font-medium">Export .{o.ext}</div>
                <div className="text-xs text-text-secondary mt-0.5">{o.text}</div>
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

export function ReviewView(p: Props) {
  const { data, selectedMatch } = p;
  const wide = useMediaQuery("(min-width: 1024px)");
  const [tab, setTab] = useState<ReviewTab>("overview");
  useEffect(() => setTab("overview"), [data.replay.replay_id]);

  const info = setInfo(data.replay.replay_id);
  const title = info.p1 && info.p2 ? `${info.p1} vs ${info.p2}` : replayTitle(data.replay);
  const range = useMemo(
    () =>
      selectedMatch
        ? { start: selectedMatch.start_ms, end: selectedMatch.end_ms }
        : { start: 0, end: Math.max(data.replay.duration_ms, data.rounds[data.rounds.length - 1]?.round_end_ms ?? 0, 1) },
    [selectedMatch, data.replay.duration_ms, data.rounds]
  );

  // The lists follow the selected game
  const inGame = <T,>(xs: T[], at: (x: T) => number) =>
    selectedMatch ? xs.filter((x) => at(x) >= selectedMatch.start_ms && at(x) <= selectedMatch.end_ms) : xs;
  const rounds = selectedMatch ? selectedMatch.rounds : data.rounds;
  const highlights = useMemo(() => inGame(data.highlights, (h) => h.timestamp_ms), [data.highlights, selectedMatch]); // eslint-disable-line react-hooks/exhaustive-deps
  const damage = useMemo(() => inGame(data.damageEvents, (e) => e.timestamp_ms), [data.damageEvents, selectedMatch]); // eslint-disable-line react-hooks/exhaustive-deps

  const tabs: TabDef<ReviewTab>[] = [
    { key: "overview", label: "Overview", icon: LayoutDashboard },
    { key: "rounds", label: "Rounds", icon: ListOrdered, count: rounds.length },
    { key: "highlights", label: "Highlights", icon: Sparkles, count: highlights.length },
    { key: "damage", label: "Damage", icon: Activity, count: damage.length },
    { key: "notes", label: "Notes", icon: StickyNote, count: data.notes.length + data.drawings.length },
  ];

  const canReanalyze = p.videoMode === "local" && !!p.onReanalyze;

  const video = (
    <VideoPlayer
      fill={wide}
      showProgress={false}
      src={p.videoSrc}
      seekToMs={p.seekToMs}
      onSeeked={p.onSeeked}
      durationMs={data.replay.duration_ms}
      rounds={data.rounds}
      damageEvents={data.damageEvents}
      selectedMatch={selectedMatch}
      onClearSelection={() => p.onSelectMatch(null)}
      onLocateVideo={p.onLocateVideo}
      onTimeUpdate={p.onTimeUpdate}
      notes={data.notes}
      drawings={data.drawings}
      onSaveDrawing={p.onSaveDrawing}
      onDeleteDrawing={p.onDeleteDrawing}
    />
  );

  const timeline = (
    <div className="shrink-0 rounded-lg bg-surface-1 border border-surface-4/60 px-3 pt-2 pb-1.5">
      <div className="flex items-center gap-x-3 gap-y-1 flex-wrap text-[10px] text-text-muted mb-1.5">
        <span className="font-semibold uppercase tracking-wider text-text-secondary">
          {selectedMatch ? `Match ${selectedMatch.match_index + 1}` : "Timeline"}
        </span>
        <span className="inline-flex items-center gap-1"><i className="w-2.5 h-0.5 bg-p1 rounded" /> P1 health</span>
        <span className="inline-flex items-center gap-1"><i className="w-2.5 h-0.5 bg-p2 rounded" /> P2 health</span>
        <span className="hidden sm:inline">lower bars: tension</span>
        {data.notes.length > 0 && <span className="inline-flex items-center gap-1"><i className="w-1.5 h-1.5 bg-accent-green rounded-full" /> notes</span>}
        <span className="ml-auto font-mono text-text-secondary">{formatClock(p.currentTimeMs)}</span>
      </div>
      <ReviewTimeline
        frameData={data.frameData}
        rounds={data.rounds}
        matches={data.matches}
        damageEvents={data.damageEvents}
        notes={data.notes}
        drawings={data.drawings}
        durationMs={data.replay.duration_ms}
        range={range}
        currentMs={p.currentTimeMs}
        onSeek={p.onSeek}
      />
    </div>
  );

  const panel = (
    <>
      <div className="px-3 pt-3 pb-2 border-b border-surface-4/60 shrink-0">
        <Tabs tabs={tabs} active={tab} onChange={setTab} />
      </div>
      <div className={`p-3 ${wide ? "flex-1 overflow-y-auto min-h-0" : ""}`}>
        {tab === "overview" && (
          <div className="flex flex-col gap-3">
            <MatchOverview stats={p.displayStats} replay={data.replay} selectedMatch={selectedMatch} onClearSelection={() => p.onSelectMatch(null)} />
            {data.matches.length > 0 && (
              <div>
                <div className="text-[11px] font-semibold uppercase tracking-wider text-text-secondary mb-2">Matches</div>
                <MatchList
                  matches={data.matches}
                  onSeek={p.onRoundSeek}
                  onExport={p.onExportClip}
                  selectedMatchIndex={selectedMatch?.match_index ?? null}
                  onOverrideWinner={p.onOverrideWinner}
                />
              </div>
            )}
          </div>
        )}
        {selectedMatch && tab !== "overview" && tab !== "notes" && (
          <div className="mb-2 text-[11px] text-text-secondary">
            Match {selectedMatch.match_index + 1} only ·{" "}
            <button onClick={() => p.onSelectMatch(null)} className="text-accent-purple hover:underline cursor-pointer">show all matches</button>
          </div>
        )}
        {tab === "rounds" && <RoundBreakdown rounds={rounds} onSeek={p.onRoundSeek} onExport={p.onExportClip} />}
        {tab === "highlights" && <HighlightsPanel highlights={highlights} onSeek={p.onSeek} onExport={p.onExportClip} />}
        {tab === "damage" && <DamageLog events={damage} onSeek={p.onSeek} />}
        {tab === "notes" && (
          <NotesPanel
            notes={data.notes}
            drawings={data.drawings}
            onSeek={p.onSeek}
            onAdd={p.onAddNote}
            onUpdate={p.onUpdateNote}
            onDelete={p.onDeleteNote}
            onDeleteDrawing={p.onDeleteDrawing}
            currentTimeMs={p.currentTimeMs}
          />
        )}
      </div>
    </>
  );

  return (
    <div className="h-full flex flex-col min-h-0 relative">
      {/* Header */}
      <header className="h-14 shrink-0 flex items-center gap-2 px-3 border-b border-surface-4/60 bg-surface-1">
        <IconButton icon={ArrowLeft} label="Back to the library" onClick={p.onBack} />
        <div className="min-w-0 mr-2">
          <div className="text-sm font-semibold truncate">{title}</div>
          {info.bracket && <div className="text-[11px] text-text-muted truncate">{info.bracket}{info.number && ` · set ${info.number}`}</div>}
        </div>

        {/* game picker */}
        {data.matches.length > 1 && (
          <div className="flex items-center gap-1 overflow-x-auto no-scrollbar min-w-0">
            <button
              onClick={() => p.onSelectMatch(null)}
              className={`h-7 px-2.5 rounded-md text-xs font-medium whitespace-nowrap cursor-pointer transition-colors ${!selectedMatch ? "bg-accent-purple/15 text-accent-purple" : "text-text-secondary hover:bg-surface-3"}`}
            >
              All matches
            </button>
            {data.matches.map((m) => {
              const on = selectedMatch?.match_index === m.match_index;
              return (
                <button
                  key={m.match_index}
                  onClick={() => {
                    p.onSelectMatch(m.match_index);
                    p.onSeek(m.start_ms);
                  }}
                  title={`Match ${m.match_index + 1}: ${m.p1_rounds_won}–${m.p2_rounds_won}, won by ${m.winner}`}
                  className={`h-7 px-2 rounded-md text-xs font-medium whitespace-nowrap cursor-pointer transition-colors inline-flex items-center gap-1.5 ${on ? "bg-accent-purple/15 text-accent-purple" : "text-text-secondary hover:bg-surface-3"}`}
                >
                  M{m.match_index + 1}
                  <span className={`w-1.5 h-1.5 rounded-full ${m.winner === "P1" ? "bg-p1" : m.winner === "P2" ? "bg-p2" : "bg-text-muted"}`} />
                </button>
              );
            })}
          </div>
        )}

        <div className="ml-auto flex items-center gap-1 shrink-0">
          {!p.videoSrc && p.videoMode === "local" && (
            <Button size="sm" variant="ghost" icon={FolderOpen} onClick={p.onLocateVideo}>
              <span className="hidden md:inline">Locate video</span>
            </Button>
          )}
          <IconButton icon={RefreshCw} label="Reload the results" onClick={p.onReload} disabled={p.busy.loading} />
          {canReanalyze && (
            <Button size="sm" variant="ghost" icon={ScanLine} onClick={p.onReanalyze} disabled={p.busy.reanalyzing} title="Run the analysis again on this video (after an app update)">
              <span className="hidden xl:inline">Reanalyse</span>
            </Button>
          )}
          <ShareMenu share={p.share} />
        </div>
      </header>

      {/* Body */}
      {wide ? (
        <div className="flex-1 min-h-0 grid grid-cols-[minmax(0,1fr)_minmax(340px,30%)] 2xl:grid-cols-[minmax(0,1fr)_480px]">
          <section className="min-h-0 flex flex-col gap-3 p-3">
            <div className="flex-1 min-h-0">{video}</div>
            {timeline}
          </section>
          <aside className="min-h-0 flex flex-col border-l border-surface-4/60 bg-surface-1/50">{panel}</aside>
        </div>
      ) : (
        <div className="flex-1 min-h-0 overflow-y-auto">
          <section className="flex flex-col gap-3 p-3">
            {video}
            {timeline}
          </section>
          <section className="border-t border-surface-4/60">{panel}</section>
        </div>
      )}

      {(p.busy.loading || p.busy.reanalyzing) && (
        <div className="absolute inset-0 top-14 z-20 bg-surface-0/70 backdrop-blur-sm flex items-center justify-center">
          <div className="flex flex-col items-center gap-3 bg-surface-2 rounded-xl border border-surface-4 px-8 py-6 shadow-2xl min-w-[300px]">
            <Spinner size={28} className="text-accent-purple" />
            <div className="text-sm font-medium">{p.busy.reanalyzing ? "Running the analysis again…" : "Loading…"}</div>
            {p.busy.reanalyzeText && <div className="text-xs text-text-muted max-w-[280px] truncate">{p.busy.reanalyzeText}</div>}
          </div>
        </div>
      )}
    </div>
  );
}
