import { useState, useCallback, useMemo, useEffect } from "react";
import { open, save } from "@tauri-apps/plugin-dialog";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { convertFileSrc } from "@tauri-apps/api/core";
import { AlertTriangle, X } from "lucide-react";
import type {
  Replay,
  FrameDataPoint,
  DamageEvent,
  RoundResult,
  MatchStats,
  Highlight,
  Note,
  Drawing,
  SpagSession,
  SpagzSession,
  VideoMode,
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
  getDrawings,
  saveDrawing,
  deleteDrawing,
  getDefaultDbPath,
  resolveVideoPath,
  reanalyzeReplay,
  reanalyzeAll,
  setRoundWinner,
  exportSpag,
  openSpag,
  saveSpag,
  exportSpagz,
  openSpagz,
  saveSpagz,
} from "./api";
import { ExportModal } from "./components/ExportModal";
import { AnalysisProgress } from "./components/AnalysisProgress";
import { SplitVodView } from "./components/SplitVodView";
import { StriveExportView } from "./components/StriveExportView";
import { UpdateBanner } from "./components/UpdateBanner";
import { SpagzVideoModal } from "./components/SpagzVideoModal";
import { AnnotateSetup } from "./components/AnnotateSetup";
import { AnnotateView } from "./components/AnnotateView";
import { AppRail, type RailTarget } from "./components/shell/AppRail";
import { Spinner } from "./components/ui";
import { LibraryView } from "./views/LibraryView";
import { ReviewView } from "./views/ReviewView";
import { detectVideoSource, type VideoSourceType } from "./lib/videoSourceDetect";
import { groupRoundsIntoMatches, statsForMatch } from "./lib/matches";

type AppView = "library" | "review" | "analyze" | "split" | "strive" | "annotate";

const VIDEO_EXT = /\.(mp4|mkv|avi|mov|webm)$/i;

export default function App() {
  const [view, setView] = useState<AppView>("library");

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
  const [drawings, setDrawings] = useState<Drawing[]>([]);

  // Video state
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [videoPath, setVideoPath] = useState<string | null>(null);
  const [seekToMs, setSeekToMs] = useState<number | null>(null);
  const [currentTimeMs, setCurrentTimeMs] = useState(0);

  // UI state
  const [exportTarget, setExportTarget] = useState<{ startMs: number; endMs: number } | null>(null);
  const [selectedMatchIndex, setSelectedMatchIndex] = useState<number | null>(null);
  const [spagSession, setSpagSession] = useState<SpagSession | null>(null);
  const [spagzSession, setSpagzSession] = useState<SpagzSession | null>(null);
  const [videoMode, setVideoMode] = useState<VideoMode>("local");
  const [savingSpag, setSavingSpag] = useState(false);
  const [exportingSpag, setExportingSpag] = useState(false);
  const [showSpagzVideoModal, setShowSpagzVideoModal] = useState(false);
  const [pendingSpagzSession, setPendingSpagzSession] = useState<SpagzSession | null>(null);
  const [showAnnotateSetup, setShowAnnotateSetup] = useState(false);
  const [annotateSrcType, setAnnotateSrcType] = useState<VideoSourceType>("file");
  const [annotateEmbedId, setAnnotateEmbedId] = useState<string | undefined>();
  const [pendingVideo, setPendingVideo] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [reanalyzing, setReanalyzing] = useState(false);
  const [reanalyzingAll, setReanalyzingAll] = useState(false);
  const [reanalyzeText, setReanalyzeText] = useState("");

  // Derived: group rounds into matches
  const matches = useMemo(() => groupRoundsIntoMatches(rounds), [rounds]);
  const selectedMatch = selectedMatchIndex !== null ? matches[selectedMatchIndex] ?? null : null;
  const displayStats = useMemo(
    () => (selectedMatch && stats ? statsForMatch(stats, selectedMatch, damageEvents) : stats),
    [selectedMatch, stats, damageEvents]
  );

  useEffect(() => {
    const un = listen<string>("reanalyze-progress", (e) => setReanalyzeText(e.payload));
    return () => {
      un.then((fn) => fn());
    };
  }, []);

  const setLocalVideo = useCallback((path: string | null) => {
    setVideoPath(path);
    if (!path) {
      setVideoSrc(null);
      return;
    }
    try {
      setVideoSrc(convertFileSrc(path));
    } catch {
      setVideoSrc(null);
    }
  }, []);

  const loadReplayData = useCallback(
    async (db: string, replay: Replay) => {
      setSelectedReplay(replay);
      setSelectedMatchIndex(null);
      setCurrentTimeMs(0);
      setLoading(true);
      try {
        const [fd, de, rn, st, hl, nt, dr, vp] = await Promise.all([
          getFrameData(db, replay.replay_id),
          getDamageEvents(db, replay.replay_id),
          getRounds(db, replay.replay_id),
          getMatchStats(db, replay.replay_id),
          getHighlights(db, replay.replay_id),
          getNotes(db, replay.replay_id).catch(() => [] as Note[]),
          getDrawings(db, replay.replay_id).catch(() => [] as Drawing[]),
          resolveVideoPath(db, replay.replay_id).catch(() => ""),
        ]);
        setFrameData(fd);
        setDamageEvents(de);
        setRounds(rn);
        setNotes(nt);
        setDrawings(dr);
        setStats(st);
        setHighlights(hl);
        setLocalVideo(vp || null);
      } catch (err) {
        console.error("Failed to load replay:", err);
        setError(`Could not load this video's results: ${err}`);
      } finally {
        setLoading(false);
      }
    },
    [setLocalVideo]
  );

  /** Open an analysis database and show its library. */
  const openDatabase = useCallback(async (path: string, openReplayId?: string) => {
    setDbPath(path);
    setSpagSession(null);
    setSpagzSession(null);
    setVideoMode("local");
    setLoading(true);
    setError(null);
    try {
      const reps = await getReplays(path);
      setReplays(reps);
      const target = openReplayId ? reps.find((r) => r.replay_id === openReplayId) : undefined;
      if (target) {
        setView("review");
        await loadReplayData(path, target);
      } else {
        setView("library");
      }
    } catch (err) {
      console.error("Failed to load DB:", err);
      setError(`Could not open the database: ${err}`);
    } finally {
      setLoading(false);
    }
  }, [loadReplayData]);

  // Open the default DB on startup if it exists
  useEffect(() => {
    getDefaultDbPath()
      .then((path) => openDatabase(path))
      .catch(() => {
        // No default DB yet: the library shows the getting-started page
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

  const handleOpenReplay = useCallback(
    (replay: Replay) => {
      if (!dbPath) return;
      if (replay.frame_count === 0) {
        // notes-only session: the annotate view
        setSelectedReplay(replay);
        setAnnotateSrcType("file");
        setAnnotateEmbedId(undefined);
        resolveVideoPath(dbPath, replay.replay_id)
          .then((vp) => setLocalVideo(vp || null))
          .catch(() => setLocalVideo(null));
        Promise.all([
          getNotes(dbPath, replay.replay_id).catch(() => [] as Note[]),
          getDrawings(dbPath, replay.replay_id).catch(() => [] as Drawing[]),
        ]).then(([nt, dr]) => {
          setNotes(nt);
          setDrawings(dr);
          setView("annotate");
        });
        return;
      }
      setView("review");
      loadReplayData(dbPath, replay);
    },
    [dbPath, loadReplayData, setLocalVideo]
  );

  const handleSeek = useCallback((ms: number) => setSeekToMs(ms), []);

  const handleRoundSeek = useCallback(
    (ms: number) => {
      setSeekToMs(ms);
      // Zoom to the game that contains this round
      const idx = matches.findIndex((m) => m.rounds.some((r) => r.round_start_ms === ms));
      setSelectedMatchIndex(idx >= 0 ? idx : null);
    },
    [matches]
  );

  const handleLocateVideo = useCallback(async () => {
    const selected = await open({
      multiple: false,
      filters: [{ name: "Video Files", extensions: ["mp4", "mkv", "avi"] }],
    });
    if (selected) {
      setLocalVideo(selected as string);
      setVideoMode("local");
    }
  }, [setLocalVideo]);

  const handleAnalysisComplete = useCallback(
    async (resultDbPath: string) => {
      setPendingVideo(null);
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

  const handleSaveDrawing = useCallback(
    async (timestampMs: number, strokesJson: string) => {
      if (!dbPath || !selectedReplay) return;
      const result = await saveDrawing(dbPath, selectedReplay.replay_id, timestampMs, strokesJson);
      setDrawings((prev) => {
        // Remove existing drawing at this timestamp
        const filtered = prev.filter((d) => Math.abs(d.timestamp_ms - timestampMs) >= 1);
        if (result) {
          return [...filtered, result].sort((a, b) => a.timestamp_ms - b.timestamp_ms);
        }
        return filtered;
      });
    },
    [dbPath, selectedReplay]
  );

  const handleDeleteDrawing = useCallback(
    async (drawingId: number) => {
      if (!dbPath) return;
      await deleteDrawing(dbPath, drawingId);
      setDrawings((prev) => prev.filter((d) => d.drawing_id !== drawingId));
    },
    [dbPath]
  );

  // Reanalyze: re-run Python CV pipeline then reload
  const handleReanalyze = useCallback(async () => {
    if (!dbPath || !selectedReplay) return;
    setReanalyzing(true);
    setReanalyzeText("");
    setError(null);
    try {
      await reanalyzeReplay(dbPath, selectedReplay.replay_id);
      await loadReplayData(dbPath, selectedReplay);
    } catch (err) {
      console.error("Reanalysis failed:", err);
      setError(`The analysis failed: ${err}`);
    } finally {
      setReanalyzing(false);
    }
  }, [dbPath, selectedReplay, loadReplayData]);

  // Reanalyze all: re-run Python CV pipeline on every replay then reload
  const handleReanalyzeAll = useCallback(async () => {
    if (!dbPath) return;
    setReanalyzingAll(true);
    setReanalyzeText("");
    setError(null);
    try {
      await reanalyzeAll(dbPath);
      setReplays(await getReplays(dbPath));
    } catch (err) {
      console.error("Reanalyze all failed:", err);
      setError(`The analysis failed: ${err}`);
    } finally {
      setReanalyzingAll(false);
    }
  }, [dbPath]);

  // Override uncertain round winner
  const handleOverrideWinner = useCallback(
    async (roundIndex: number, winner: string) => {
      if (!dbPath || !selectedReplay) return;
      try {
        await setRoundWinner(dbPath, selectedReplay.replay_id, roundIndex, winner);
        setRounds(await getRounds(dbPath, selectedReplay.replay_id));
      } catch (err) {
        console.error("Failed to override winner:", err);
        setError(String(err));
      }
    },
    [dbPath, selectedReplay]
  );

  /** Notes and drawings of a session, then the annotate view. */
  const enterAnnotate = useCallback(async (db: string, replay: Replay) => {
    setSelectedReplay(replay);
    const [nt, dr] = await Promise.all([
      getNotes(db, replay.replay_id).catch(() => [] as Note[]),
      getDrawings(db, replay.replay_id).catch(() => [] as Drawing[]),
    ]);
    setNotes(nt);
    setDrawings(dr);
    setView("annotate");
  }, []);

  /** All analysis data of a session's replay, then the review screen. */
  const enterReview = useCallback(async (db: string, replay: Replay) => {
    setView("review");
    setSelectedReplay(replay);
    setSelectedMatchIndex(null);
    setCurrentTimeMs(0);
    const [fd, de, rn, st, hl, nt, dr] = await Promise.all([
      getFrameData(db, replay.replay_id),
      getDamageEvents(db, replay.replay_id),
      getRounds(db, replay.replay_id),
      getMatchStats(db, replay.replay_id),
      getHighlights(db, replay.replay_id),
      getNotes(db, replay.replay_id).catch(() => [] as Note[]),
      getDrawings(db, replay.replay_id).catch(() => [] as Drawing[]),
    ]);
    setFrameData(fd);
    setDamageEvents(de);
    setRounds(rn);
    setStats(st);
    setHighlights(hl);
    setNotes(nt);
    setDrawings(dr);
  }, []);

  // .spag file support
  const openSpagFile = useCallback(async (path: string) => {
    setLoading(true);
    setError(null);
    try {
      const session = await openSpag(path);
      setSpagSession(session);
      setSpagzSession(null);
      setDbPath(session.db_path);
      setVideoMode("local");
      const reps = await getReplays(session.db_path);
      setReplays(reps);
      // There's only one replay in a .spag
      const replay = reps.find((r) => r.replay_id === session.replay_id) || reps[0];
      if (replay) {
        setLocalVideo(session.video_path);
        if (replay.frame_count === 0) {
          setAnnotateSrcType("file");
          setAnnotateEmbedId(undefined);
          await enterAnnotate(session.db_path, replay);
        } else {
          await enterReview(session.db_path, replay);
        }
      }
    } catch (err) {
      console.error("Failed to open .spag:", err);
      setError(`Could not open the file: ${err}`);
    } finally {
      setLoading(false);
    }
  }, [enterAnnotate, enterReview, setLocalVideo]);

  /** A .spagz session: annotate sessions open at once, full analyses ask for the video. */
  const loadSpagzReplayData = useCallback(async (session: SpagzSession) => {
    setSpagzSession(session);
    setSpagSession(null);
    setDbPath(session.db_path);
    const reps = await getReplays(session.db_path);
    setReplays(reps);
    const replay = reps.find((r) => r.replay_id === session.replay_id) || reps[0];
    if (!replay) return;
    if (replay.frame_count === 0) {
      const detected = session.video_url ? detectVideoSource(session.video_url) : null;
      if (detected && detected.type !== "file") {
        setAnnotateSrcType(detected.type);
        setAnnotateEmbedId(detected.id);
        setVideoSrc(null);
      } else {
        setAnnotateSrcType("file");
        setAnnotateEmbedId(undefined);
      }
      await enterAnnotate(session.db_path, replay);
      return;
    }
    await enterReview(session.db_path, replay);
  }, [enterAnnotate, enterReview]);

  const openSpagzFile = useCallback(async (path: string) => {
    setLoading(true);
    setError(null);
    try {
      const session = await openSpagz(path);
      const reps = await getReplays(session.db_path);
      const replay = reps.find((r) => r.replay_id === session.replay_id) || reps[0];
      if (replay && replay.frame_count === 0) {
        setVideoSrc(null);
        await loadSpagzReplayData(session);
        return;
      }
      // Full analysis: ask the user for the video
      setPendingSpagzSession(session);
      setShowSpagzVideoModal(true);
    } catch (err) {
      console.error("Failed to open .spagz:", err);
      setError(`Could not open the file: ${err}`);
    } finally {
      setLoading(false);
    }
  }, [loadSpagzReplayData]);

  const openAnyFile = useCallback(
    async (path: string) => {
      if (path.toLowerCase().endsWith(".spagz")) await openSpagzFile(path);
      else await openSpagFile(path);
    },
    [openSpagFile, openSpagzFile]
  );

  const handleOpenSpag = useCallback(async () => {
    const selected = await open({
      multiple: false,
      filters: [{ name: "Spaghetti Lab Analysis", extensions: ["spag", "spagz"] }],
    });
    if (selected) await openAnyFile(selected as string);
  }, [openAnyFile]);

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
      setError(`Export failed: ${err}`);
    } finally {
      setExportingSpag(false);
    }
  }, [dbPath, selectedReplay]);

  const handleExportSpagz = useCallback(async () => {
    if (!dbPath || !selectedReplay) return;
    const dest = await save({
      filters: [{ name: "Spaghetti Lab Analysis (no video)", extensions: ["spagz"] }],
      defaultPath: `${selectedReplay.replay_id}.spagz`,
    });
    if (!dest) return;
    setExportingSpag(true);
    try {
      await exportSpagz(dbPath, selectedReplay.replay_id, dest);
    } catch (err) {
      console.error("Failed to export .spagz:", err);
      setError(`Export failed: ${err}`);
    } finally {
      setExportingSpag(false);
    }
  }, [dbPath, selectedReplay]);

  const handleSaveSession = useCallback(async () => {
    setSavingSpag(true);
    try {
      if (spagzSession) await saveSpagz(spagzSession.spagz_path, spagzSession.db_path);
      else if (spagSession) await saveSpag(spagSession.spag_path, spagSession.db_path);
    } catch (err) {
      console.error("Failed to save:", err);
      setError(`Saving failed: ${err}`);
    } finally {
      setSavingSpag(false);
    }
  }, [spagSession, spagzSession]);

  const finishSpagzModal = useCallback(
    async (mode: VideoMode, setup: () => void) => {
      setShowSpagzVideoModal(false);
      const session = pendingSpagzSession;
      if (!session) return;
      setLoading(true);
      try {
        setup();
        setVideoMode(mode);
        await loadSpagzReplayData(session);
      } catch (err) {
        console.error("Failed to load .spagz:", err);
        setError(`Could not open the file: ${err}`);
      } finally {
        setLoading(false);
        setPendingSpagzSession(null);
      }
    },
    [pendingSpagzSession, loadSpagzReplayData]
  );

  // Annotate session handler
  const handleAnnotateSession = useCallback(
    async (session: SpagzSession, srcType: VideoSourceType, embedId?: string) => {
      setShowAnnotateSetup(false);
      setDbPath(session.db_path);
      setSpagSession(null);
      setSpagzSession(session.spagz_path ? session : null);
      setAnnotateSrcType(srcType);
      setAnnotateEmbedId(embedId);
      if (srcType === "file" && session.video_hint) setLocalVideo(session.video_hint);
      else setVideoSrc(null);
      await enterAnnotate(session.db_path, {
        replay_id: session.replay_id,
        video_path: session.video_hint,
        duration_ms: 0,
        frame_count: 0,
      });
    },
    [enterAnnotate, setLocalVideo]
  );

  // .spag/.spagz file association (app launched with a file argument)
  useEffect(() => {
    const unlisten1 = listen<string>("open-spag-file", (event) => openSpagFile(event.payload));
    const unlisten2 = listen<string>("open-spagz-file", (event) => openSpagzFile(event.payload));
    return () => {
      unlisten1.then((fn) => fn());
      unlisten2.then((fn) => fn());
    };
  }, [openSpagFile, openSpagzFile]);

  // Drag and drop: a video starts an analysis, a .spag / .spagz opens
  useEffect(() => {
    const un = getCurrentWebview().onDragDropEvent((e) => {
      const t = e.payload.type;
      if (t === "enter" || t === "over") setDragActive(true);
      else if (t === "leave") setDragActive(false);
      else if (t === "drop") {
        setDragActive(false);
        const path = e.payload.paths[0];
        if (!path) return;
        if (/\.spagz?$/i.test(path)) openAnyFile(path);
        else if (VIDEO_EXT.test(path)) {
          setPendingVideo(path);
          setView("analyze");
        } else setError("Drop a match video (MP4, MKV, AVI) or a .spag / .spagz file.");
      }
    });
    return () => {
      un.then((fn) => fn());
    };
  }, [openAnyFile]);

  const handleOpenDefaultDb = useCallback(async () => {
    try {
      await openDatabase(await getDefaultDbPath());
    } catch {
      await handleOpenDb();
    }
  }, [openDatabase, handleOpenDb]);

  const navigate = useCallback((t: RailTarget) => {
    if (t === "annotate") setShowAnnotateSetup(true);
    else if (t === "analyze") {
      setPendingVideo(null);
      setView("analyze");
    } else setView(t);
  }, []);

  const railActive: RailTarget | null =
    view === "review" || view === "library" ? "library" : view;

  let content: React.ReactNode;
  if (view === "review" && selectedReplay && stats && displayStats) {
    content = (
      <ReviewView
        data={{ replay: selectedReplay, frameData, damageEvents, rounds, matches, stats, highlights, notes, drawings }}
        displayStats={displayStats}
        selectedMatch={selectedMatch}
        onSelectMatch={setSelectedMatchIndex}
        videoSrc={videoSrc}
        videoMode={videoMode}
        seekToMs={seekToMs}
        onSeek={handleSeek}
        onSeeked={() => setSeekToMs(null)}
        currentTimeMs={currentTimeMs}
        onTimeUpdate={setCurrentTimeMs}
        onRoundSeek={handleRoundSeek}
        onExportClip={(startMs, endMs) => setExportTarget({ startMs, endMs })}
        onOverrideWinner={handleOverrideWinner}
        onLocateVideo={handleLocateVideo}
        onAddNote={handleAddNote}
        onUpdateNote={handleUpdateNote}
        onDeleteNote={handleDeleteNote}
        onSaveDrawing={handleSaveDrawing}
        onDeleteDrawing={handleDeleteDrawing}
        onBack={() => setView("library")}
        onReload={() => dbPath && loadReplayData(dbPath, selectedReplay)}
        onReanalyze={dbPath ? handleReanalyze : undefined}
        busy={{ loading, reanalyzing, reanalyzeText }}
        share={{
          session: spagzSession ? "spagz" : spagSession ? "spag" : null,
          onSave: handleSaveSession,
          saving: savingSpag,
          onExportSpag: handleExportSpag,
          onExportSpagz: handleExportSpagz,
          exporting: exportingSpag,
        }}
      />
    );
  } else if (view === "review") {
    content = (
      <div className="h-full flex items-center justify-center gap-3 text-text-secondary">
        <Spinner className="text-accent-purple" /> Loading the results…
      </div>
    );
  } else if (view === "annotate" && selectedReplay && dbPath) {
    content = (
      <AnnotateView
        videoSrc={videoSrc}
        srcType={annotateSrcType}
        embedId={annotateEmbedId}
        replay={selectedReplay}
        dbPath={dbPath}
        spagzSession={spagzSession}
        notes={notes}
        drawings={drawings}
        onNotesChange={setNotes}
        onDrawingsChange={setDrawings}
        onBack={() => setView("library")}
      />
    );
  } else if (view === "analyze") {
    content = (
      <AnalysisProgress
        key={pendingVideo ?? "pick"}
        initialPath={pendingVideo}
        onComplete={handleAnalysisComplete}
        onCancel={() => setView("library")}
      />
    );
  } else if (view === "strive") {
    content = <StriveExportView onComplete={handleAnalysisComplete} onCancel={() => setView("library")} />;
  } else if (view === "split") {
    content = (
      <SplitVodView
        onComplete={(outputDir) => {
          console.log("VOD split complete:", outputDir);
          setView("library");
        }}
        onCancel={() => setView("library")}
      />
    );
  } else {
    content = (
      <LibraryView
        replays={replays}
        dbPath={dbPath}
        loading={loading}
        dragActive={dragActive}
        onOpenReplay={handleOpenReplay}
        onAddVideo={() => navigate("analyze")}
        onOpenFile={handleOpenSpag}
        onOpenDatabase={handleOpenDefaultDb}
        onAnnotate={() => setShowAnnotateSetup(true)}
        onSplit={() => setView("split")}
        onStrive={() => setView("strive")}
        onReanalyzeAll={dbPath && !spagSession && !spagzSession ? handleReanalyzeAll : undefined}
        reanalyzingAll={reanalyzingAll}
      />
    );
  }

  return (
    <div className="h-screen flex overflow-hidden bg-surface-0">
      <AppRail active={railActive} onNavigate={navigate} onOpenFile={handleOpenSpag} />
      <main className="flex-1 min-w-0 flex flex-col overflow-hidden relative">
        <UpdateBanner />
        {error && (
          <div className="mx-3 mt-3 flex items-start gap-2 rounded-lg border border-p1/40 bg-p1/10 px-3 py-2 text-sm text-p1-light shrink-0">
            <AlertTriangle size={16} className="mt-0.5 shrink-0" />
            <span className="flex-1 min-w-0 break-words">{error}</span>
            <button onClick={() => setError(null)} className="shrink-0 opacity-70 hover:opacity-100 cursor-pointer" aria-label="Dismiss">
              <X size={16} />
            </button>
          </div>
        )}
        <div className="flex-1 min-h-0">{content}</div>

        {reanalyzingAll && (
          <div className="absolute inset-0 z-40 bg-surface-0/80 backdrop-blur-sm flex items-center justify-center">
            <div className="flex flex-col items-center gap-3 bg-surface-2 rounded-xl border border-surface-4 px-8 py-6 shadow-2xl min-w-[320px]">
              <Spinner size={28} className="text-accent-purple" />
              <div className="text-sm font-medium">Running the analysis again on every video…</div>
              {reanalyzeText && <div className="text-xs text-text-muted max-w-[280px] truncate">{reanalyzeText}</div>}
            </div>
          </div>
        )}
      </main>

      {/* Clip export: only for a local video */}
      {exportTarget && videoPath && videoMode === "local" && (
        <ExportModal
          videoPath={videoPath}
          startMs={exportTarget.startMs}
          endMs={exportTarget.endMs}
          onClose={() => setExportTarget(null)}
        />
      )}

      {showSpagzVideoModal && pendingSpagzSession && (
        <SpagzVideoModal
          videoHint={pendingSpagzSession.video_hint}
          videoUrl={pendingSpagzSession.video_url}
          onLocalFile={(filePath) => finishSpagzModal("local", () => setLocalVideo(filePath))}
          onStreamUrl={(url) =>
            finishSpagzModal("url", () => {
              setVideoSrc(url);
              setVideoPath(null);
            })
          }
          onSkip={() => finishSpagzModal("none", () => setLocalVideo(null))}
          onCancel={() => {
            setShowSpagzVideoModal(false);
            setPendingSpagzSession(null);
          }}
        />
      )}
      {showAnnotateSetup && (
        <AnnotateSetup onSession={handleAnnotateSession} onCancel={() => setShowAnnotateSetup(false)} />
      )}
    </div>
  );
}
