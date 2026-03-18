import { useState, useCallback, useMemo, useEffect, useRef } from "react";
import { open, save } from "@tauri-apps/plugin-dialog";
import { listen } from "@tauri-apps/api/event";
import type {
  Replay,
  FrameDataPoint,
  DamageEvent,
  RoundResult,
  Match,
  MatchStats,
  Highlight,
  Note,
  SpagSession,
  ActiveTab,
} from "./types";
import {
  getReplays,
  getFrameData,
  getDamageEvents,
  getRounds,
  getMatchStats,
  getHighlights,
  getNotes,
  addNote,
  updateNote,
  deleteNote,
  getDefaultDbPath,
  resolveVideoPath,
  reanalyzeReplay,
  reanalyzeAll,
  setRoundWinner,
  exportSpag,
  openSpag,
  saveSpag,
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
import { SplitVodView } from "./components/SplitVodView";
import { NotesPanel } from "./components/NotesPanel";
import { UpdateBanner } from "./components/UpdateBanner";

/** Group rounds into matches (first-to-2 round wins). */
function groupRoundsIntoMatches(rounds: RoundResult[]): Match[] {
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

type AppView = "welcome" | "analyze" | "split" | "dashboard";

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
  const [notes, setNotes] = useState<Note[]>([]);

  // Video state
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [videoPath, setVideoPath] = useState<string | null>(null);
  const [seekToMs, setSeekToMs] = useState<number | null>(null);
  const [currentTimeMs, setCurrentTimeMs] = useState(0);

  // UI state
  const [activeTab, setActiveTab] = useState<ActiveTab>("matches");
  const [exportTarget, setExportTarget] = useState<{
    startMs: number;
    endMs: number;
  } | null>(null);
  const [selectedMatchIndex, setSelectedMatchIndex] = useState<number | null>(
    null
  );
  const [spagSession, setSpagSession] = useState<SpagSession | null>(null);
  const [savingSpag, setSavingSpag] = useState(false);
  const [exportingSpag, setExportingSpag] = useState(false);

  // Resizable split between top (video/timeline) and bottom (tabs)
  const [topHeight, setTopHeight] = useState<number | string>("55%");
  const resizeRef = useRef<{ startY: number; startH: number } | null>(null);

  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const container = (e.target as HTMLElement).parentElement;
    const topEl = container?.querySelector<HTMLElement>("[style*='height']");
    if (!topEl) return;
    const startH = topEl.getBoundingClientRect().height;
    resizeRef.current = { startY: e.clientY, startH };

    const onMove = (ev: MouseEvent) => {
      if (!resizeRef.current) return;
      const delta = ev.clientY - resizeRef.current.startY;
      const newH = Math.max(150, resizeRef.current.startH + delta);
      setTopHeight(newH);
    };
    const onUp = () => {
      resizeRef.current = null;
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
    document.body.style.cursor = "row-resize";
    document.body.style.userSelect = "none";
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }, []);

  // Derived: group rounds into matches
  const matches = useMemo(() => groupRoundsIntoMatches(rounds), [rounds]);

  const selectedMatch =
    selectedMatchIndex !== null ? matches[selectedMatchIndex] ?? null : null;

  // Compute match-scoped stats when a match is selected
  const displayStats = useMemo(() => {
    if (!selectedMatch || !stats) return stats;
    const matchDamage = damageEvents.filter(
      (e) =>
        e.timestamp_ms >= selectedMatch.start_ms &&
        e.timestamp_ms <= selectedMatch.end_ms
    );
    const p1Hits = matchDamage.filter((e) => e.target_side === 1);
    const p2Hits = matchDamage.filter((e) => e.target_side === 2);
    const durations = selectedMatch.rounds.map(
      (r) => (r.round_end_ms - r.round_start_ms) / 1000
    );
    const winnerHps = selectedMatch.rounds.map((r) => r.winner_final_hp);
    return {
      ...stats,
      total_rounds: selectedMatch.rounds.length,
      p1_round_wins: selectedMatch.p1_rounds_won,
      p2_round_wins: selectedMatch.p2_rounds_won,
      total_damage_events: matchDamage.length,
      p1_damage_taken: p1Hits.reduce((s, e) => s + e.damage_pct, 0),
      p2_damage_taken: p2Hits.reduce((s, e) => s + e.damage_pct, 0),
      p1_biggest_hit:
        p1Hits.length > 0 ? Math.max(...p1Hits.map((e) => e.damage_pct)) : 0,
      p2_biggest_hit:
        p2Hits.length > 0 ? Math.max(...p2Hits.map((e) => e.damage_pct)) : 0,
      avg_round_duration_s:
        durations.length > 0
          ? durations.reduce((a, b) => a + b, 0) / durations.length
          : 0,
      longest_round_s: durations.length > 0 ? Math.max(...durations) : 0,
      shortest_round_s: durations.length > 0 ? Math.min(...durations) : 0,
      comeback_count: selectedMatch.rounds.filter((r) => r.is_comeback).length,
      close_rounds: selectedMatch.rounds.filter(
        (r) => r.winner_final_hp < 0.2
      ).length,
      avg_winner_final_hp:
        winnerHps.length > 0
          ? winnerHps.reduce((a, b) => a + b, 0) / winnerHps.length
          : 0,
      duration_s: (selectedMatch.end_ms - selectedMatch.start_ms) / 1000,
    } as MatchStats;
  }, [selectedMatch, stats, damageEvents]);

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

  // Auto-open the default DB on startup if it exists
  useEffect(() => {
    getDefaultDbPath()
      .then((path) => openDatabase(path))
      .catch(() => {
        // No default DB yet — stay on welcome screen
      });
  }, [openDatabase]);

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
      setSelectedMatchIndex(null);
      setLoading(true);
      try {
        const [fd, de, rn, st, hl, nt, vp] = await Promise.all([
          getFrameData(db, replay.replay_id),
          getDamageEvents(db, replay.replay_id),
          getRounds(db, replay.replay_id),
          getMatchStats(db, replay.replay_id),
          getHighlights(db, replay.replay_id),
          getNotes(db, replay.replay_id).catch(() => [] as Note[]),
          resolveVideoPath(db, replay.replay_id).catch(() => ""),
        ]);
        setFrameData(fd);
        setDamageEvents(de);
        setRounds(rn);
        setNotes(nt);
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

  const handleRoundSeek = useCallback(
    (ms: number) => {
      setSeekToMs(ms);
      // Find which match contains this timestamp
      const idx = matches.findIndex((m) =>
        m.rounds.some((r) => r.round_start_ms === ms)
      );
      setSelectedMatchIndex(idx >= 0 ? idx : null);
    },
    [matches]
  );

  const handleExport = useCallback((startMs: number, endMs: number) => {
    setExportTarget({ startMs, endMs });
  }, []);

  const handleLocateVideo = useCallback(async () => {
    const selected = await open({
      multiple: false,
      filters: [{ name: "Video Files", extensions: ["mp4", "mkv", "avi"] }],
    });
    if (selected) {
      const path = selected as string;
      setVideoPath(path);
      try {
        setVideoSrc(convertFileSrc(path));
      } catch {
        setVideoSrc(null);
      }
    }
  }, []);

  const handleAnalysisComplete = useCallback(
    async (resultDbPath: string) => {
      await openDatabase(resultDbPath);
    },
    [openDatabase]
  );

  const handleAddNote = useCallback(
    async (timestampMs: number, text: string) => {
      if (!dbPath || !selectedReplay) return;
      const note = await addNote(dbPath, selectedReplay.replay_id, timestampMs, text);
      setNotes((prev) => [...prev, note].sort((a, b) => a.timestamp_ms - b.timestamp_ms));
    },
    [dbPath, selectedReplay]
  );

  const handleUpdateNote = useCallback(
    async (noteId: number, text: string) => {
      if (!dbPath) return;
      await updateNote(dbPath, noteId, text);
      setNotes((prev) => prev.map((n) => (n.note_id === noteId ? { ...n, text } : n)));
    },
    [dbPath]
  );

  const handleDeleteNote = useCallback(
    async (noteId: number) => {
      if (!dbPath) return;
      await deleteNote(dbPath, noteId);
      setNotes((prev) => prev.filter((n) => n.note_id !== noteId));
    },
    [dbPath]
  );

  // Reanalyze: re-run Python CV pipeline then reload
  const [reanalyzing, setReanalyzing] = useState(false);
  const handleReanalyze = useCallback(async () => {
    if (!dbPath || !selectedReplay) return;
    setReanalyzing(true);
    setError(null);
    try {
      await reanalyzeReplay(dbPath, selectedReplay.replay_id);
      await loadReplayData(dbPath, selectedReplay);
    } catch (err) {
      console.error("Reanalysis failed:", err);
      setError(String(err));
    } finally {
      setReanalyzing(false);
    }
  }, [dbPath, selectedReplay, loadReplayData]);

  // Reanalyze all: re-run Python CV pipeline on every replay then reload
  const [reanalyzingAll, setReanalyzingAll] = useState(false);
  const handleReanalyzeAll = useCallback(async () => {
    if (!dbPath) return;
    setReanalyzingAll(true);
    setError(null);
    try {
      await reanalyzeAll(dbPath);
      // Refresh replay list (metadata may have changed)
      const reps = await getReplays(dbPath);
      setReplays(reps);
      // Reload the currently selected replay if any
      const current = selectedReplay
        ? reps.find((r) => r.replay_id === selectedReplay.replay_id)
        : null;
      if (current) {
        setSelectedReplay(current);
        await loadReplayData(dbPath, current);
      }
    } catch (err) {
      console.error("Reanalyze all failed:", err);
      setError(String(err));
    } finally {
      setReanalyzingAll(false);
    }
  }, [dbPath, selectedReplay, loadReplayData]);

  // Override uncertain round winner
  const handleOverrideWinner = useCallback(
    async (roundIndex: number, winner: string) => {
      if (!dbPath || !selectedReplay) return;
      try {
        await setRoundWinner(dbPath, selectedReplay.replay_id, roundIndex, winner);
        const rn = await getRounds(dbPath, selectedReplay.replay_id);
        setRounds(rn);
      } catch (err) {
        console.error("Failed to override winner:", err);
        setError(String(err));
      }
    },
    [dbPath, selectedReplay]
  );

  // .spag file support
  const openSpagFile = useCallback(async (path: string) => {
    setLoading(true);
    setError(null);
    try {
      const session = await openSpag(path);
      setSpagSession(session);
      setDbPath(session.db_path);

      const reps = await getReplays(session.db_path);
      setReplays(reps);
      setView("dashboard");

      // Load the replay (there's only one in a .spag)
      const replay = reps.find(r => r.replay_id === session.replay_id) || reps[0];
      if (replay) {
        setSelectedReplay(replay);
        setSelectedMatchIndex(null);
        const [fd, de, rn, st, hl, nt] = await Promise.all([
          getFrameData(session.db_path, replay.replay_id),
          getDamageEvents(session.db_path, replay.replay_id),
          getRounds(session.db_path, replay.replay_id),
          getMatchStats(session.db_path, replay.replay_id),
          getHighlights(session.db_path, replay.replay_id),
          getNotes(session.db_path, replay.replay_id).catch(() => [] as Note[]),
        ]);
        setFrameData(fd);
        setDamageEvents(de);
        setRounds(rn);
        setStats(st);
        setHighlights(hl);
        setNotes(nt);

        // Use the extracted video from the .spag session
        setVideoPath(session.video_path);
        try {
          setVideoSrc(convertFileSrc(session.video_path));
        } catch {
          setVideoSrc(null);
        }
        setActiveTab("matches");
      }
    } catch (err) {
      console.error("Failed to open .spag:", err);
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  const handleOpenSpag = useCallback(async () => {
    const selected = await open({
      multiple: false,
      filters: [{ name: "Spaghetti Lab Analysis", extensions: ["spag"] }],
    });
    if (!selected) return;
    await openSpagFile(selected as string);
  }, [openSpagFile]);

  const handleExportSpag = useCallback(async () => {
    if (!dbPath || !selectedReplay) return;
    const dest = await save({
      filters: [{ name: "Spaghetti Lab Analysis", extensions: ["spag"] }],
      defaultPath: `${selectedReplay.replay_id}.spag`,
    });
    if (!dest) return;
    setExportingSpag(true);
    try {
      await exportSpag(dbPath, selectedReplay.replay_id, dest);
    } catch (err) {
      console.error("Failed to export .spag:", err);
      setError(String(err));
    } finally {
      setExportingSpag(false);
    }
  }, [dbPath, selectedReplay]);

  const handleSaveSpag = useCallback(async () => {
    if (!spagSession) return;
    setSavingSpag(true);
    try {
      await saveSpag(spagSession.spag_path, spagSession.db_path);
    } catch (err) {
      console.error("Failed to save .spag:", err);
      setError(String(err));
    } finally {
      setSavingSpag(false);
    }
  }, [spagSession]);

  // Listen for .spag file association (app launched with .spag argument)
  useEffect(() => {
    const unlisten = listen<string>("open-spag-file", (event) => {
      openSpagFile(event.payload);
    });
    return () => { unlisten.then(fn => fn()); };
  }, [openSpagFile]);

  const handleOpenDefaultDb = useCallback(async () => {
    try {
      const path = await getDefaultDbPath();
      await openDatabase(path);
    } catch {
      // No default DB — fall back to file picker
      await handleOpenDb();
    }
  }, [openDatabase, handleOpenDb]);

  // Welcome screen
  if (view === "welcome") {
    return (
      <WelcomeScreen
        onAnalyze={() => setView("analyze")}
        onSplitVod={() => setView("split")}
        onOpenSpag={handleOpenSpag}
        onOpenDatabase={handleOpenDefaultDb}
      />
    );
  }

  // Analysis progress view
  if (view === "analyze") {
    return (
      <AnalysisProgress
        onComplete={handleAnalysisComplete}
        onCancel={() => setView(dbPath ? "dashboard" : "welcome")}
      />
    );
  }

  // VOD splitter view
  if (view === "split") {
    return (
      <SplitVodView
        onComplete={(outputDir) => {
          console.log("VOD split complete:", outputDir);
          setView("welcome");
        }}
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
              setSpagSession(null);
              setView("welcome");
            }}
            className="flex items-center gap-2 hover:opacity-80 transition-opacity cursor-pointer"
          >
            <img
              src="/spaghetti-showdown-logo.png"
              alt="Spaghetti Showdown"
              className="h-7"
            />
            <span className="text-accent-purple font-semibold text-sm tracking-wide">
              SPAGHETTI LAB
            </span>
          </button>
          <span className="text-text-muted text-xs">|</span>
          <span className="text-text-secondary text-xs truncate max-w-[400px]">
            {spagSession
              ? spagSession.spag_path.split(/[/\\]/).pop()
              : dbPath?.split(/[/\\]/).pop()}
          </span>
          {spagSession ? (
            <span className="text-p2 text-[10px] font-medium">.spag</span>
          ) : (
            <span className="text-text-muted text-[10px]">
              {replays.length} sets
            </span>
          )}
        </div>
        <div className="ml-auto flex items-center gap-2">
          {error && (
            <span className="text-p1 text-xs mr-2">{error}</span>
          )}
          {spagSession && (
            <button
              onClick={handleSaveSpag}
              disabled={savingSpag}
              className="btn-ghost text-xs text-accent-green border-accent-green/30 hover:bg-accent-green/10"
            >
              {savingSpag ? "Saving..." : "Save .spag"}
            </button>
          )}
          {selectedReplay && !spagSession && (
            <button
              onClick={handleExportSpag}
              disabled={exportingSpag}
              className="btn-ghost text-xs text-p2 border-p2/30 hover:bg-p2/10"
            >
              {exportingSpag ? "Exporting..." : "Export .spag"}
            </button>
          )}
          <button onClick={handleOpenDb} className="btn-ghost text-xs">
            Open Database
          </button>
        </div>
      </header>

      <UpdateBanner />

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <Sidebar
          replays={replays}
          selectedReplay={selectedReplay}
          onSelect={handleSelectReplay}
          onOpenDb={handleOpenDb}
          onReload={selectedReplay && dbPath ? () => loadReplayData(dbPath, selectedReplay) : undefined}
          reloading={loading}
          onReanalyze={selectedReplay && dbPath ? handleReanalyze : undefined}
          reanalyzing={reanalyzing}
          onReanalyzeAll={dbPath ? handleReanalyzeAll : undefined}
          reanalyzingAll={reanalyzingAll}
          onAnalyzeNew={() => setView("analyze")}
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
              {/* Top section: video + overview + timeline (resizable) */}
              <div
                className="shrink-0 overflow-y-auto"
                style={{ height: topHeight }}
              >
                {/* Match zoom banner */}
                {selectedMatch && (
                  <div className="mx-4 mt-3 mb-0 flex items-center gap-2 bg-accent-purple/10 border border-accent-purple/25 rounded-lg px-3 py-1.5">
                    <span className="text-accent-purple font-semibold text-sm">
                      Viewing Game {selectedMatch.match_index + 1}
                    </span>
                    <span className="text-text-muted text-xs">
                      {Math.floor(selectedMatch.start_ms / 60000)}:{String(Math.floor((selectedMatch.start_ms / 1000) % 60)).padStart(2, "0")} – {Math.floor(selectedMatch.end_ms / 60000)}:{String(Math.floor((selectedMatch.end_ms / 1000) % 60)).padStart(2, "0")}
                    </span>
                    <button
                      onClick={() => setSelectedMatchIndex(null)}
                      className="ml-auto text-xs bg-accent-purple/20 hover:bg-accent-purple/30 text-accent-purple px-2.5 py-1 rounded transition-colors cursor-pointer"
                    >
                      Show all games
                    </button>
                  </div>
                )}

                {/* Video + Overview row */}
                <div className="flex gap-4 p-4 pb-0">
                  <div className="flex-1 min-w-0">
                    <VideoPlayer
                      src={videoSrc}
                      seekToMs={seekToMs}
                      onSeeked={() => setSeekToMs(null)}
                      durationMs={selectedReplay.duration_ms}
                      rounds={rounds}
                      damageEvents={damageEvents}
                      selectedMatch={selectedMatch}
                      onClearSelection={() => setSelectedMatchIndex(null)}
                      onLocateVideo={handleLocateVideo}
                      onTimeUpdate={setCurrentTimeMs}
                      notes={notes}
                    />
                  </div>
                  <div className="w-[320px] shrink-0">
                    <MatchOverview
                      stats={displayStats!}
                      replay={selectedReplay}
                      selectedMatch={selectedMatch}
                      onClearSelection={() => setSelectedMatchIndex(null)}
                    />
                  </div>
                </div>

                {/* Health timeline */}
                <div className="px-4 pt-3">
                  <HealthTimeline
                    frameData={frameData}
                    rounds={rounds}
                    damageEvents={damageEvents}
                    highlights={highlights}
                    notes={notes}
                    onSeek={handleSeek}
                    selectedMatch={selectedMatch}
                    onClearSelection={() => setSelectedMatchIndex(null)}
                  />
                </div>
              </div>

              {/* Resize handle */}
              <div
                className="h-1.5 shrink-0 cursor-row-resize group flex items-center justify-center hover:bg-accent-purple/10 transition-colors"
                onMouseDown={handleResizeStart}
              >
                <div className="w-10 h-0.5 rounded-full bg-surface-4 group-hover:bg-accent-purple/50 transition-colors" />
              </div>

              {/* Tab section */}
              <div className="flex-1 flex flex-col overflow-hidden px-4 pt-1 pb-4">
                <div className="flex border-b border-surface-4/50 shrink-0">
                  {(
                    [
                      ["matches", "Matches"],
                      ["rounds", "Rounds"],
                      ["highlights", "Highlights"],
                      ["damage", "Damage Log"],
                      ["notes", "Notes"],
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
                      {key === "notes" && notes.length > 0 && (
                        <span className="ml-1.5 text-[10px] text-accent-green bg-accent-green/15 px-1 py-0.5 rounded">
                          {notes.length}
                        </span>
                      )}
                    </button>
                  ))}
                </div>
                <div className="flex-1 overflow-y-auto mt-3">
                  {activeTab === "matches" && (
                    <MatchList
                      matches={matches}
                      onSeek={handleRoundSeek}
                      onExport={handleExport}
                      selectedMatchIndex={selectedMatchIndex}
                      onOverrideWinner={handleOverrideWinner}
                    />
                  )}
                  {activeTab === "rounds" && (
                    <RoundBreakdown
                      rounds={rounds}
                      onSeek={handleRoundSeek}
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
                  {activeTab === "notes" && (
                    <NotesPanel
                      notes={notes}
                      onSeek={handleSeek}
                      onAdd={handleAddNote}
                      onUpdate={handleUpdateNote}
                      onDelete={handleDeleteNote}
                      currentTimeMs={currentTimeMs}
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
