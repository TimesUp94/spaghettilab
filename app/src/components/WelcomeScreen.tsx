interface Props {
  onOpenDb: () => void;
  onAnalyze: () => void;
}

export function WelcomeScreen({ onOpenDb, onAnalyze }: Props) {
  return (
    <div className="h-screen flex items-center justify-center bg-surface-0">
      <div className="text-center max-w-lg">
        {/* Logo / Title */}
        <div className="mb-8">
          <h1 className="text-5xl font-bold tracking-tight mb-3">
            <span className="text-accent-purple">REPL</span>
            <span className="text-text-primary">ANAL</span>
          </h1>
          <p className="text-text-secondary text-sm">
            Guilty Gear Strive Replay Analyzer
          </p>
        </div>

        {/* Decorative line */}
        <div className="w-48 h-px bg-gradient-to-r from-transparent via-accent-purple/40 to-transparent mx-auto mb-10" />

        {/* Actions */}
        <div className="grid grid-cols-2 gap-3 max-w-sm mx-auto">
          <button
            onClick={onOpenDb}
            className="px-5 py-4 bg-accent-purple/12 text-accent-purple border border-accent-purple/20
                       rounded-xl hover:bg-accent-purple/20 hover:border-accent-purple/35
                       transition-all duration-300 cursor-pointer group"
          >
            <div className="text-lg mb-1 opacity-70 group-hover:opacity-100 transition-opacity">
              &#128194;
            </div>
            <div className="font-medium text-sm">Open Database</div>
            <div className="text-[10px] text-text-muted mt-1">
              Browse existing analysis
            </div>
          </button>
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
    </div>
  );
}
