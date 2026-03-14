import { useState, useCallback, useMemo } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import type {
  Replay,
  FrameDataPoint,
  DamageEvent,
  RoundResult,
  Match,
  MatchStats,
  Highlight,
  ActiveTab,
} from "./types";
import {
  getReplays,
  getFrameData,
  getDamageEvents,
  getRounds,
  getMatchStats,
  getHighlights,
  resolveVideoPath,
} from "./api";
import { convertFileSrc } from "@tauri-apps/api/core";
import { Sidebar } from "./components/Sidebar";
import { VideoPlayer } from "./components/VideoPlayer";
import { HealthTimeline } from "./components/HealthTimeline";
import { MatchOverview } from "./components/MatchOverview";
import { RoundBreakdown } from "./components/RoundBreakdown";
import { MatchList } from "./components/MatchList";
import { HighlightsPanel } from "./components/Highlights";
import { DamageLog } from "./components/DamageLog";
import { WelcomeScreen } from "./components/WelcomeScreen";
import { ExportModal } from "./components/ExportModal";
import { AnalysisProgress } from "./components/AnalysisProgress";

/** Group rounds into matches (first-to-2 round wins). */
function groupRoundsIntoMatches(rounds: RoundResult[]): Match[] {
  const matches: Match[] = [];
  let currentRounds: RoundResult[] = [];
  let p1Wins = 0;
  let p2Wins = 0;

  for (const round of rounds) {
    currentRounds.push(round);
    if (round.winner === "P1") p1Wins++;
    else if (round.winner === "P2") p2Wins++;

    if (p1Wins >= 2 || p2Wins >= 2) {
      matches.push({
        match_index: matches.length,
        rounds: currentRounds,
        winner: p1Wins >= 2 ? "P1" : "P2",
        p1_rounds_won: p1Wins,
        p2_rounds_won: p2Wins,
        start_ms: currentRounds[0].round_start_ms,
        end_ms: currentRounds[currentRounds.length - 1].round_end_ms,
      });
      currentRounds = [];
      p1Wins = 0;
      p2Wins = 0;
    }
  }

  // Remaining rounds that didn't complete a match
  if (currentRounds.length > 0) {
    matches.push({
      match_index: matches.length,
      rounds: currentRounds,
      winner: p1Wins > p2Wins ? "P1" : p2Wins > p1Wins ? "P2" : "??",
      p1_rounds_won: p1Wins,
      p2_rounds_won: p2Wins,
      start_ms: currentRounds[0].round_start_ms,
      end_ms: currentRounds[currentRounds.length - 1].round_end_ms,
    });
  }

  return matches;
}

type AppView = "welcome" | "analyze" | "dashboard";

export default function App() {
  const [view, setView] = useState<AppView>("welcome");

  // DB + replay state
  const [dbPath, setDbPath] = useState<string | null>(null);
  const [replays, setReplays] = useState<Replay[]>([]);
  const [selectedReplay, setSelectedReplay] = useState<Replay | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Analysis data for selected replay
  const [frameData, setFrameData] = useState<FrameDataPoint[]>([]);
  const [damageEvents, setDamageEvents] = useState<DamageEvent[]>([]);
  const [rounds, setRounds] = useState<RoundResult[]>([]);
  const [stats, setStats] = useState<MatchStats | null>(null);
  const [highlights, setHighlights] = useState<Highlight[]>([]);

  // Video state
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [videoPath, setVideoPath] = useState<string | null>(null);
  const [seekToMs, setSeekToMs] = useState<number | null>(null);

  // UI state
  const [activeTab, setActiveTab] = useState<ActiveTab>("matches");
  const [exportTarget, setExportTarget] = useState<{
    startMs: number;
    endMs: number;
  } | null>(null);

  // Derived: group rounds into matches
  const matches = useMemo(() => groupRoundsIntoMatches(rounds), [rounds]);

  const openDatabase = useCallback(async (path: string) => {
    setDbPath(path);
    setLoading(true);
    setError(null);
    try {
      const reps = await getReplays(path);
      setReplays(reps);
      setView("dashboard");
      if (reps.length > 0) {
        await loadReplayData(path, reps[0]);
      }
    } catch (err) {
      console.error("Failed to load DB:", err);
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  const handleOpenDb = useCallback(async () => {
    const selected = await open({
      multiple: false,
      filters: [{ name: "SQLite Database", extensions: ["db"] }],
    });
    if (!selected) return;
    await openDatabase(selected as string);
  }, [openDatabase]);

  const loadReplayData = useCallback(
    async (db: string, replay: Replay) => {
      setSelectedReplay(replay);
      setLoading(true);
      try {
        const [fd, de, rn, st, hl, vp] = await Promise.all([
          getFrameData(db, replay.replay_id),
          getDamageEvents(db, replay.replay_id),
          getRounds(db, replay.replay_id),
          getMatchStats(db, replay.replay_id),
          getHighlights(db, replay.replay_id),
          resolveVideoPath(db, replay.replay_id).catch(() => ""),
        ]);
        setFrameData(fd);
        setDamageEvents(de);
        setRounds(rn);
        setStats(st);
        setHighlights(hl);
        if (vp) {
          setVideoPath(vp);
          try {
            setVideoSrc(convertFileSrc(vp));
          } catch {
            setVideoSrc(null);
          }
        } else {
          setVideoPath(null);
          setVideoSrc(null);
        }
        setActiveTab("matches");
      } catch (err) {
        console.error("Failed to load replay:", err);
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const handleSelectReplay = useCallback(
    (replay: Replay) => {
      if (dbPath) {
        loadReplayData(dbPath, replay);
      }
    },
    [dbPath, loadReplayData]
  );

  const handleSeek = useCallback((ms: number) => {
    setSeekToMs(ms);
  }, []);

  const handleExport = useCallback((startMs: number, endMs: number) => {
    setExportTarget({ startMs, endMs });
  }, []);

  const handleAnalysisComplete = useCallback(
    async (resultDbPath: string) => {
      await openDatabase(resultDbPath);
    },
    [openDatabase]
  );

  // Welcome screen
  if (view === "welcome") {
    return (
      <WelcomeScreen
        onOpenDb={handleOpenDb}
        onAnalyze={() => setView("analyze")}
      />
    );
  }

  // Analysis progress view
  if (view === "analyze") {
    return (
      <AnalysisProgress
        onComplete={handleAnalysisComplete}
        onCancel={() => setView("welcome")}
      />
    );
  }

  // Dashboard
  return (
    <div className="h-screen flex flex-col overflow-hidden">
      {/* Top bar */}
      <header className="h-12 bg-surface-1 border-b border-surface-4/50 flex items-center px-4 shrink-0">
        <div className="flex items-center gap-3">
          <button
            onClick={() => {
              setDbPath(null);
              setView("welcome");
            }}
            className="text-accent-purple font-semibold text-sm tracking-wide hover:text-accent-purple/80 transition-colors cursor-pointer"
          >
            REPLANAL
          </button>
          <span className="text-text-muted text-xs">|</span>
          <span className="text-text-secondary text-xs truncate max-w-[400px]">
            {dbPath?.split(/[/\\]/).pop()}
          </span>
          <span className="text-text-muted text-[10px]">
            {replays.length} sets
          </span>
        </div>
        <div className="ml-auto flex items-center gap-2">
          {error && (
            <span className="text-p1 text-xs mr-2">{error}</span>
          )}
          <button onClick={handleOpenDb} className="btn-ghost text-xs">
            Open Database
          </button>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <Sidebar
          replays={replays}
          selectedReplay={selectedReplay}
          onSelect={handleSelectReplay}
          onOpenDb={handleOpenDb}
        />

        {/* Main content */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {loading ? (
            <div className="flex-1 flex items-center justify-center">
              <div className="flex flex-col items-center gap-3">
                <div className="w-8 h-8 border-2 border-accent-purple/30 border-t-accent-purple rounded-full animate-spin" />
                <div className="text-text-secondary text-sm">
                  Loading analysis data...
                </div>
              </div>
            </div>
          ) : selectedReplay && stats ? (
            <>
              {/* Video + Overview row */}
              <div className="flex gap-4 p-4 pb-0 shrink-0">
                <div className="flex-1 min-w-0">
                  <VideoPlayer
                    src={videoSrc}
                    seekToMs={seekToMs}
                    onSeeked={() => setSeekToMs(null)}
                    durationMs={selectedReplay.duration_ms}
                    rounds={rounds}
                    damageEvents={damageEvents}
                  />
                </div>
                <div className="w-[320px] shrink-0">
                  <MatchOverview stats={stats} replay={selectedReplay} />
                </div>
              </div>

              {/* Health timeline */}
              <div className="px-4 pt-3 shrink-0">
                <HealthTimeline
                  frameData={frameData}
                  rounds={rounds}
                  damageEvents={damageEvents}
                  highlights={highlights}
                  onSeek={handleSeek}
                />
              </div>

              {/* Tab section */}
              <div className="flex-1 flex flex-col overflow-hidden px-4 pt-2 pb-4">
                <div className="flex border-b border-surface-4/50 shrink-0">
                  {(
                    [
                      ["matches", "Matches"],
                      ["rounds", "Rounds"],
                      ["highlights", "Highlights"],
                      ["damage", "Damage Log"],
                    ] as const
                  ).map(([key, label]) => (
                    <button
                      key={key}
                      className={`tab-btn ${activeTab === key ? "active" : ""}`}
                      onClick={() => setActiveTab(key)}
                    >
                      {label}
                      {key === "matches" && matches.length > 0 && (
                        <span className="ml-1.5 text-[10px] text-text-muted bg-surface-3 px-1 py-0.5 rounded">
                          {matches.length}
                        </span>
                      )}
                      {key === "rounds" && rounds.length > 0 && (
                        <span className="ml-1.5 text-[10px] text-text-muted bg-surface-3 px-1 py-0.5 rounded">
                          {rounds.length}
                        </span>
                      )}
                      {key === "highlights" && highlights.length > 0 && (
                        <span className="ml-1.5 text-[10px] text-accent-gold bg-accent-gold/15 px-1 py-0.5 rounded">
                          {highlights.length}
                        </span>
                      )}
                    </button>
                  ))}
                </div>
                <div className="flex-1 overflow-y-auto mt-3">
                  {activeTab === "matches" && (
                    <MatchList
                      matches={matches}
                      onSeek={handleSeek}
                      onExport={handleExport}
                    />
                  )}
                  {activeTab === "rounds" && (
                    <RoundBreakdown
                      rounds={rounds}
                      onSeek={handleSeek}
                      onExport={handleExport}
                    />
                  )}
                  {activeTab === "highlights" && (
                    <HighlightsPanel
                      highlights={highlights}
                      onSeek={handleSeek}
                      onExport={handleExport}
                    />
                  )}
                  {activeTab === "damage" && (
                    <DamageLog
                      events={damageEvents}
                      onSeek={handleSeek}
                    />
                  )}
                </div>
              </div>
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center text-text-muted">
              Select a replay from the sidebar
            </div>
          )}
        </main>
      </div>

      {/* Export modal */}
      {exportTarget && videoPath && (
        <ExportModal
          videoPath={videoPath}
          startMs={exportTarget.startMs}
          endMs={exportTarget.endMs}
          onClose={() => setExportTarget(null)}
        />
      )}
    </div>
  );
}
