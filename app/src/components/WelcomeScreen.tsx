import { open as shellOpen } from "@tauri-apps/plugin-shell";

const DONATE_URL =
  "https://www.paypal.com/donate/?business=79WVBGLTEYV8N&no_recurring=0&item_name=Supports+the+Spaghetti+Showdown+community%2C+further+tool+development+and+TimesUp+himself&currency_code=EUR";

interface Props {
  onAnalyze: () => void;
  onSplitVod: () => void;
  onOpenSpag: () => void;
}

export function WelcomeScreen({ onAnalyze, onSplitVod, onOpenSpag }: Props) {
  return (
    <div className="h-screen flex flex-col items-center justify-center bg-surface-0">
      <div className="text-center max-w-lg">
        {/* Logo / Title */}
        <div className="mb-8">
          <img
            src="/spaghetti-showdown-logo.png"
            alt="Spaghetti Showdown"
            className="w-48 mx-auto mb-4 opacity-90"
          />
          <h1 className="text-5xl font-bold tracking-tight mb-3">
            <span className="text-accent-purple">SPAGHETTI</span>{" "}
            <span className="text-text-primary">LAB</span>
          </h1>
          <p className="text-text-secondary text-sm">
            Guilty Gear Strive Replay Analyzer
          </p>
        </div>

        {/* Decorative line */}
        <div className="w-48 h-px bg-gradient-to-r from-transparent via-accent-purple/40 to-transparent mx-auto mb-10" />

        {/* Actions */}
        <div className="grid grid-cols-3 gap-3 max-w-md mx-auto">
          <button
            onClick={onAnalyze}
            className="px-5 py-4 bg-accent-green/12 text-accent-green border border-accent-green/20
                       rounded-xl hover:bg-accent-green/20 hover:border-accent-green/35
                       transition-all duration-300 cursor-pointer group"
          >
            <div className="text-lg mb-1 opacity-70 group-hover:opacity-100 transition-opacity">
              &#127916;
            </div>
            <div className="font-medium text-sm">Analyze Video</div>
            <div className="text-[10px] text-text-muted mt-1">
              Process a new replay
            </div>
          </button>
          <button
            onClick={onSplitVod}
            className="px-5 py-4 bg-accent-gold/12 text-accent-gold border border-accent-gold/20
                       rounded-xl hover:bg-accent-gold/20 hover:border-accent-gold/35
                       transition-all duration-300 cursor-pointer group"
          >
            <div className="text-lg mb-1 opacity-70 group-hover:opacity-100 transition-opacity">
              &#9986;
            </div>
            <div className="font-medium text-sm">Split VOD</div>
            <div className="text-[10px] text-text-muted mt-1">
              Cut a VOD into sets
            </div>
          </button>
          <button
            onClick={onOpenSpag}
            className="px-5 py-4 bg-p2/12 text-p2 border border-p2/20
                       rounded-xl hover:bg-p2/20 hover:border-p2/35
                       transition-all duration-300 cursor-pointer group"
          >
            <div className="text-lg mb-1 opacity-70 group-hover:opacity-100 transition-opacity">
              &#128230;
            </div>
            <div className="font-medium text-sm">Open .spag</div>
            <div className="text-[10px] text-text-muted mt-1">
              Open exported analysis
            </div>
          </button>
        </div>

        {/* Feature cards */}
        <div className="mt-12 grid grid-cols-4 gap-3 text-center">
          {[
            ["Health Tracking", "Per-frame HP"],
            ["Round Detection", "Win/loss analysis"],
            ["Comebacks", "Deficit detection"],
            ["Clip Export", "ffmpeg extraction"],
          ].map(([title, desc]) => (
            <div key={title} className="stat-card !p-3">
              <div className="text-text-primary text-[11px] font-medium mb-0.5">
                {title}
              </div>
              <div className="text-text-muted text-[10px]">{desc}</div>
            </div>
          ))}
        </div>
      </div>
      <button
        onClick={() => shellOpen(DONATE_URL)}
        className="mt-8 text-[11px] text-text-muted hover:text-accent-purple transition-colors cursor-pointer"
      >
        Support the Spaghetti Showdown community
      </button>
    </div>
  );
}
