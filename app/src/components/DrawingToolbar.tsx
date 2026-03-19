interface Props {
  activeTool: "pen" | "eraser";
  penColor: string;
  penSize: number;
  onToolChange: (tool: "pen" | "eraser") => void;
  onColorChange: (color: string) => void;
  onSizeChange: (size: number) => void;
  onClear: () => void;
  onDone: () => void;
}

const COLORS = [
  { value: "#ff3333", label: "Red" },
  { value: "#3388ff", label: "Blue" },
  { value: "#ffc833", label: "Gold" },
  { value: "#ffffff", label: "White" },
  { value: "#33cc66", label: "Green" },
  { value: "#000000", label: "Black" },
];

export function DrawingToolbar({
  activeTool,
  penColor,
  penSize,
  onToolChange,
  onColorChange,
  onSizeChange,
  onClear,
  onDone,
}: Props) {
  return (
    <div className="absolute top-2 right-2 z-20 flex items-center gap-2 bg-black/70 backdrop-blur-sm rounded-lg px-3 py-2">
      {/* Pen / Eraser */}
      <button
        onClick={() => onToolChange("pen")}
        className={`px-2 py-1 rounded text-xs font-medium transition-colors cursor-pointer ${
          activeTool === "pen"
            ? "bg-accent-purple/30 text-accent-purple"
            : "text-text-muted hover:text-text-secondary"
        }`}
      >
        Pen
      </button>
      <button
        onClick={() => onToolChange("eraser")}
        className={`px-2 py-1 rounded text-xs font-medium transition-colors cursor-pointer ${
          activeTool === "eraser"
            ? "bg-accent-purple/30 text-accent-purple"
            : "text-text-muted hover:text-text-secondary"
        }`}
      >
        Eraser
      </button>

      <div className="w-px h-4 bg-text-muted/30" />

      {/* Color swatches */}
      {COLORS.map((c) => (
        <button
          key={c.value}
          onClick={() => onColorChange(c.value)}
          className={`w-5 h-5 rounded-full border-2 transition-all cursor-pointer ${
            penColor === c.value && activeTool === "pen"
              ? "border-white scale-110"
              : c.value === "#000000"
                ? "border-white/20 hover:border-white/40"
                : "border-transparent hover:border-white/40"
          }`}
          style={{ backgroundColor: c.value }}
          title={c.label}
        />
      ))}

      <div className="w-px h-4 bg-text-muted/30" />

      {/* Size slider */}
      <input
        type="range"
        min="2"
        max="24"
        step="1"
        value={penSize}
        onChange={(e) => onSizeChange(parseInt(e.target.value))}
        className="w-16 h-1 accent-accent-purple"
        title={`Size: ${penSize}`}
      />

      <div className="w-px h-4 bg-text-muted/30" />

      {/* Clear & Done */}
      <button
        onClick={onClear}
        className="px-2 py-1 rounded text-xs text-p1 hover:bg-p1/20 transition-colors cursor-pointer"
      >
        Clear
      </button>
      <button
        onClick={onDone}
        className="px-2 py-1 rounded text-xs font-medium bg-accent-purple/30 text-accent-purple hover:bg-accent-purple/40 transition-colors cursor-pointer"
      >
        Done
      </button>
    </div>
  );
}
