// The app's left navigation rail: the library and the tools, plus open-file and support.
import type { LucideIcon } from "lucide-react";
import { LayoutGrid, Plus, MessageSquarePlus, Scissors, Gamepad2, FolderOpen, Heart } from "lucide-react";
import { open as shellOpen } from "@tauri-apps/plugin-shell";

export type RailTarget = "library" | "analyze" | "annotate" | "split" | "strive";

const DONATE_URL =
  "https://www.paypal.com/donate/?business=79WVBGLTEYV8N&no_recurring=0&item_name=Supports+the+Spaghetti+Showdown+community%2C+further+tool+development+and+TimesUp+himself&currency_code=EUR";

const ITEMS: { key: RailTarget; label: string; hint: string; icon: LucideIcon }[] = [
  { key: "library", label: "Library", hint: "Your analysed matches", icon: LayoutGrid },
  { key: "analyze", label: "Analyse", hint: "Analyse a match video", icon: Plus },
  { key: "annotate", label: "Annotate", hint: "Take notes on a video or a YouTube / Twitch VOD", icon: MessageSquarePlus },
  { key: "split", label: "Split VOD", hint: "Cut a tournament VOD into sets", icon: Scissors },
  { key: "strive", label: "Strive", hint: "Record your saved Strive replays", icon: Gamepad2 },
];

function RailButton({
  icon: Icon,
  label,
  hint,
  active,
  onClick,
}: {
  icon: LucideIcon;
  label: string;
  hint: string;
  active?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      title={hint}
      aria-current={active ? "page" : undefined}
      className={`relative w-full flex flex-col items-center gap-1 py-2.5 rounded-lg transition-colors cursor-pointer
        focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-purple/60
        ${active ? "text-accent-purple bg-accent-purple/10" : "text-text-secondary hover:text-text-primary hover:bg-surface-3"}`}
    >
      {active && <span className="absolute -left-2 top-2 bottom-2 w-[3px] rounded-r bg-accent-purple" />}
      <Icon size={20} strokeWidth={active ? 2.25 : 1.75} />
      <span className="text-[10px] font-medium leading-none">{label}</span>
    </button>
  );
}

export function AppRail({
  active,
  onNavigate,
  onOpenFile,
}: {
  active: RailTarget | null;
  onNavigate: (t: RailTarget) => void;
  onOpenFile: () => void;
}) {
  return (
    <nav className="w-[72px] shrink-0 bg-surface-1 border-r border-surface-4/60 flex flex-col items-center px-2 py-3 gap-1">
      <button
        onClick={() => onNavigate("library")}
        className="mb-3 p-1 rounded-lg hover:bg-surface-3 transition-colors cursor-pointer"
        title="Spaghetti Lab"
      >
        <img src="/spaghetti-showdown-logo.png" alt="Spaghetti Lab" className="w-10 h-10 object-contain" />
      </button>
      {ITEMS.map((it) => (
        <RailButton key={it.key} icon={it.icon} label={it.label} hint={it.hint} active={active === it.key} onClick={() => onNavigate(it.key)} />
      ))}
      <div className="mt-auto w-full flex flex-col gap-1">
        <RailButton icon={FolderOpen} label="Open" hint="Open a .spag / .spagz file" onClick={onOpenFile} />
        <RailButton icon={Heart} label="Support" hint="Support the project" onClick={() => shellOpen(DONATE_URL)} />
      </div>
    </nav>
  );
}
