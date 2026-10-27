// Home: the analysed matches of the open database, or a getting-started page when there are none.
import { useMemo, useState } from "react";
import {
  Upload, FolderOpen, Database, RefreshCw, Search, Film, Clock, MessageSquarePlus, Scissors,
  Gamepad2, FileArchive, ArrowRight, Swords, BarChart3, StickyNote,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { Replay } from "../types";
import { Badge, Button, EmptyState, PageHeader, Spinner } from "../components/ui";
import { fileName, formatLength, replayTitle, setInfo } from "../lib/matches";

interface Props {
  replays: Replay[];
  dbPath: string | null;
  loading: boolean;
  dragActive: boolean;
  onOpenReplay: (r: Replay) => void;
  onAddVideo: () => void;
  onOpenFile: () => void;
  onOpenDatabase: () => void;
  onAnnotate: () => void;
  onSplit: () => void;
  onStrive: () => void;
  onReanalyzeAll?: () => void;
  reanalyzingAll: boolean;
}

function DropZone({ dragActive, onBrowse, big }: { dragActive: boolean; onBrowse: () => void; big?: boolean }) {
  return (
    <button
      onClick={onBrowse}
      className={`w-full rounded-xl border-2 border-dashed transition-colors cursor-pointer flex flex-col items-center justify-center text-center gap-2
        ${big ? "py-12 px-6" : "py-6 px-4"}
        ${dragActive ? "border-accent-purple bg-accent-purple/10" : "border-surface-4 hover:border-accent-purple/50 hover:bg-surface-2"}`}
    >
      <div className={`rounded-full flex items-center justify-center ${big ? "w-14 h-14" : "w-10 h-10"} ${dragActive ? "bg-accent-purple text-surface-0" : "bg-accent-purple/15 text-accent-purple"}`}>
        <Upload size={big ? 26 : 18} />
      </div>
      <div className={`font-semibold text-text-primary ${big ? "text-base" : "text-sm"}`}>
        {dragActive ? "Drop it to start the analysis" : "Drop a match video here"}
      </div>
      <div className="text-xs text-text-secondary">
        or <span className="text-accent-purple underline underline-offset-2">browse your files</span> · MP4, MKV or AVI, 1080p works best
      </div>
    </button>
  );
}

function Tool({ icon: Icon, title, text, onClick }: { icon: LucideIcon; title: string; text: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="group text-left flex gap-3 p-4 rounded-xl bg-surface-2 border border-surface-4/70 hover:border-accent-purple/40 hover:bg-surface-3/60 transition-colors cursor-pointer"
    >
      <div className="w-9 h-9 rounded-lg bg-surface-3 group-hover:bg-accent-purple/15 text-text-secondary group-hover:text-accent-purple flex items-center justify-center shrink-0 transition-colors">
        <Icon size={18} />
      </div>
      <div className="min-w-0">
        <div className="text-sm font-medium text-text-primary flex items-center gap-1">
          {title}
          <ArrowRight size={13} className="opacity-0 -translate-x-1 group-hover:opacity-100 group-hover:translate-x-0 transition-all text-accent-purple" />
        </div>
        <div className="text-xs text-text-secondary mt-0.5 leading-relaxed">{text}</div>
      </div>
    </button>
  );
}

function Welcome(props: Props) {
  const steps: { icon: LucideIcon; title: string; text: string }[] = [
    { icon: Film, title: "Add a match", text: "A recording of a Guilty Gear Strive match or a whole set." },
    { icon: BarChart3, title: "Let it read the game", text: "Health, tension, timer and rounds are read from the HUD, frame by frame." },
    { icon: Swords, title: "Review", text: "Jump to any round, see who won it and how, and keep notes and drawings." },
  ];
  return (
    <div className="max-w-4xl mx-auto">
      <div className="flex items-center gap-4 mb-8">
        <img src="/spaghetti-showdown-logo.png" alt="" className="w-16 h-16 object-contain" />
        <div>
          <h1 className="text-3xl font-bold tracking-tight">
            Welcome to <span className="text-accent-purple">Spaghetti Lab</span>
          </h1>
          <p className="text-text-secondary mt-1">Turn your Guilty Gear Strive matches into something you can study.</p>
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-3 mb-6">
        {steps.map((s, i) => (
          <div key={s.title} className="p-4 rounded-xl bg-surface-2 border border-surface-4/70">
            <div className="flex items-center gap-2 mb-2">
              <span className="w-6 h-6 rounded-full bg-accent-purple/15 text-accent-purple text-xs font-bold flex items-center justify-center">{i + 1}</span>
              <s.icon size={16} className="text-text-secondary" />
            </div>
            <div className="text-sm font-medium">{s.title}</div>
            <div className="text-xs text-text-secondary mt-1 leading-relaxed">{s.text}</div>
          </div>
        ))}
      </div>

      <DropZone dragActive={props.dragActive} onBrowse={props.onAddVideo} big />

      <h2 className="text-sm font-semibold text-text-secondary mt-10 mb-3">Other things you can do</h2>
      <div className="grid gap-3 sm:grid-cols-2">
        <Tool icon={MessageSquarePlus} title="Annotate a VOD" text="Notes, drawings and a review checklist on a video file or a YouTube / Twitch link. No analysis needed." onClick={props.onAnnotate} />
        <Tool icon={Scissors} title="Split a tournament VOD" text="Find every set in a long stream by the player names and cut it into one file per set." onClick={props.onSplit} />
        <Tool icon={Gamepad2} title="Export Strive replays" text="Let the app play your saved replays in the game and record them, ready to analyse." onClick={props.onStrive} />
        <Tool icon={FileArchive} title="Open a shared analysis" text="Open a .spag or .spagz file someone sent you." onClick={props.onOpenFile} />
      </div>
      <div className="mt-6 text-xs text-text-muted">
        Have an analysis database already?{" "}
        <button onClick={props.onOpenDatabase} className="text-text-secondary hover:text-accent-purple underline underline-offset-2 cursor-pointer">
          Open a database
        </button>
      </div>
    </div>
  );
}

export function LibraryView(props: Props) {
  const { replays, dbPath, loading, onOpenReplay } = props;
  const [query, setQuery] = useState("");

  const items = useMemo(() => {
    const q = query.trim().toLowerCase();
    return replays
      .map((r) => ({ r, info: setInfo(r.replay_id), title: replayTitle(r) }))
      .filter(({ r, title }) => !q || title.toLowerCase().includes(q) || r.replay_id.toLowerCase().includes(q));
  }, [replays, query]);

  if (loading && replays.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-text-secondary gap-3">
        <Spinner /> Loading your library…
      </div>
    );
  }

  if (replays.length === 0) {
    return (
      <div className="h-full overflow-y-auto px-4 sm:px-8 py-8">
        <Welcome {...props} />
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto px-4 sm:px-8 py-6">
      <div className="max-w-6xl mx-auto">
        <PageHeader
          title="Library"
          subtitle={
            <span className="inline-flex items-center gap-1.5">
              <Database size={13} /> {replays.length} analysed {replays.length === 1 ? "video" : "videos"} · {fileName(dbPath)}
            </span>
          }
          actions={
            <>
              {props.onReanalyzeAll && (
                <Button variant="ghost" size="md" icon={RefreshCw} loading={props.reanalyzingAll} onClick={props.onReanalyzeAll} title="Run the analysis again on every video (after an update)">
                  <span className="hidden md:inline">Reanalyse all</span>
                </Button>
              )}
              <Button variant="ghost" icon={Database} onClick={props.onOpenDatabase} title="Open another analysis database">
                <span className="hidden md:inline">Open database</span>
              </Button>
              <Button variant="secondary" icon={FolderOpen} onClick={props.onOpenFile}>
                <span className="hidden sm:inline">Open file</span>
              </Button>
              <Button variant="primary" icon={Upload} onClick={props.onAddVideo}>
                Add video
              </Button>
            </>
          }
        />

        {props.dragActive && (
          <div className="mb-6">
            <DropZone dragActive onBrowse={props.onAddVideo} />
          </div>
        )}

        {replays.length > 6 && (
          <div className="relative mb-4 max-w-sm">
            <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search players or file names"
              className="w-full h-9 pl-9 pr-3 rounded-lg bg-surface-2 border border-surface-4 text-sm placeholder:text-text-muted focus:outline-none focus:border-accent-purple/60"
            />
          </div>
        )}

        {items.length === 0 ? (
          <EmptyState icon={Search} title="No match for your search">Try a player name or part of the file name.</EmptyState>
        ) : (
          <div className="grid gap-3 grid-cols-[repeat(auto-fill,minmax(250px,1fr))]">
            {items.map(({ r, info, title }) => (
              <button
                key={r.replay_id}
                onClick={() => onOpenReplay(r)}
                className="group text-left rounded-xl bg-surface-2 border border-surface-4/70 hover:border-accent-purple/50 transition-colors cursor-pointer overflow-hidden"
              >
                <div className="h-20 relative bg-gradient-to-br from-p1/25 via-surface-3 to-p2/25 flex items-center justify-center">
                  {info.p1 && info.p2 ? (
                    <div className="flex items-center gap-2 px-3 w-full text-sm font-semibold">
                      <span className="flex-1 truncate text-right text-p1-light">{info.p1}</span>
                      <span className="text-[10px] text-text-muted font-bold">VS</span>
                      <span className="flex-1 truncate text-p2-light">{info.p2}</span>
                    </div>
                  ) : (
                    <Film size={26} className="text-text-muted group-hover:text-accent-purple transition-colors" />
                  )}
                  <div className="absolute inset-0 bg-accent-purple/0 group-hover:bg-accent-purple/5 transition-colors" />
                </div>
                <div className="p-3">
                  <div className="text-sm font-medium truncate" title={title}>
                    {info.p1 && info.bracket ? `${info.bracket}${info.number ? ` · Set ${Number(info.number)}` : ""}` : title}
                  </div>
                  <div className="flex items-center gap-2 mt-1.5 text-[11px] text-text-secondary">
                    {r.duration_ms > 0 && (
                      <span className="inline-flex items-center gap-1"><Clock size={11} /> {formatLength(r.duration_ms)}</span>
                    )}
                    {r.frame_count === 0 && <Badge tone="green"><StickyNote size={10} /> Notes only</Badge>}
                  </div>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
