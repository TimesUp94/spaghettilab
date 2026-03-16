import { useState, useRef, useCallback, useEffect } from "react";
import type { RoiRect, VodRoiConfig } from "../types";

// Default ROIs for 1920x1080 GGS with "SHOW DOWN" overlay
const DEFAULT_ROIS: VodRoiConfig = {
  p1_tension: { y1: 1040, y2: 1058, x1: 50, x2: 440 },
  p2_tension: { y1: 1040, y2: 1058, x1: 1480, x2: 1870 },
  timer: { y1: 30, y2: 100, x1: 910, x2: 1010 },
  banner: { y1: 2, y2: 20, x1: 100, x2: 1820 },
};

const ROI_COLORS: Record<string, string> = {
  p1_tension: "rgba(59, 130, 246, 0.4)",
  p2_tension: "rgba(239, 68, 68, 0.4)",
  timer: "rgba(250, 204, 21, 0.4)",
  banner: "rgba(168, 85, 247, 0.4)",
};

const ROI_BORDER_COLORS: Record<string, string> = {
  p1_tension: "rgb(59, 130, 246)",
  p2_tension: "rgb(239, 68, 68)",
  timer: "rgb(250, 204, 21)",
  banner: "rgb(168, 85, 247)",
};

const ROI_LABELS: Record<string, string> = {
  p1_tension: "P1 Tension",
  p2_tension: "P2 Tension",
  timer: "Timer",
  banner: "Banner",
};

interface Props {
  imageSrc: string;
  videoWidth: number;
  videoHeight: number;
  onConfirm: (config: VodRoiConfig) => void;
  onBack: () => void;
}

type DragState = {
  key: string;
  type: "move" | "resize";
  corner?: "tl" | "tr" | "bl" | "br";
  startX: number;
  startY: number;
  origRoi: RoiRect;
};

export function RoiPicker({ imageSrc, videoWidth, videoHeight, onConfirm, onBack }: Props) {
  const [rois, setRois] = useState<VodRoiConfig>({ ...DEFAULT_ROIS });
  const [drag, setDrag] = useState<DragState | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 });

  const scale = imgSize.w > 0 ? imgSize.w / videoWidth : 1;

  const onImgLoad = useCallback((e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    setImgSize({ w: img.clientWidth, h: img.clientHeight });
  }, []);

  // Resize observer for container size changes
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const img = container.querySelector("img");
    if (!img) return;
    const observer = new ResizeObserver(() => {
      setImgSize({ w: img.clientWidth, h: img.clientHeight });
    });
    observer.observe(img);
    return () => observer.disconnect();
  }, [imageSrc]);

  const getMousePos = useCallback(
    (e: MouseEvent | React.MouseEvent) => {
      const container = containerRef.current;
      if (!container) return { x: 0, y: 0 };
      const img = container.querySelector("img");
      if (!img) return { x: 0, y: 0 };
      const rect = img.getBoundingClientRect();
      return {
        x: (e.clientX - rect.left) / scale,
        y: (e.clientY - rect.top) / scale,
      };
    },
    [scale]
  );

  const handleMouseDown = useCallback(
    (key: string, type: "move" | "resize", corner?: "tl" | "tr" | "bl" | "br") =>
      (e: React.MouseEvent) => {
        e.stopPropagation();
        e.preventDefault();
        const pos = getMousePos(e);
        const roi = rois[key as keyof VodRoiConfig];
        setDrag({ key, type, corner, startX: pos.x, startY: pos.y, origRoi: { ...roi } });
      },
    [getMousePos, rois]
  );

  useEffect(() => {
    if (!drag) return;

    const onMove = (e: MouseEvent) => {
      const pos = getMousePos(e);
      const dx = pos.x - drag.startX;
      const dy = pos.y - drag.startY;
      const orig = drag.origRoi;

      setRois((prev) => {
        const updated = { ...prev };
        if (drag.type === "move") {
          const w = orig.x2 - orig.x1;
          const h = orig.y2 - orig.y1;
          let nx1 = Math.round(orig.x1 + dx);
          let ny1 = Math.round(orig.y1 + dy);
          nx1 = Math.max(0, Math.min(videoWidth - w, nx1));
          ny1 = Math.max(0, Math.min(videoHeight - h, ny1));
          updated[drag.key as keyof VodRoiConfig] = {
            y1: ny1, y2: ny1 + h, x1: nx1, x2: nx1 + w,
          };
        } else if (drag.type === "resize" && drag.corner) {
          let { y1, y2, x1, x2 } = orig;
          if (drag.corner.includes("l")) x1 = Math.max(0, Math.round(orig.x1 + dx));
          if (drag.corner.includes("r")) x2 = Math.min(videoWidth, Math.round(orig.x2 + dx));
          if (drag.corner.includes("t")) y1 = Math.max(0, Math.round(orig.y1 + dy));
          if (drag.corner.includes("b")) y2 = Math.min(videoHeight, Math.round(orig.y2 + dy));
          if (x2 - x1 >= 10 && y2 - y1 >= 4) {
            updated[drag.key as keyof VodRoiConfig] = { y1, y2, x1, x2 };
          }
        }
        return updated;
      });
    };

    const onUp = () => setDrag(null);

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [drag, getMousePos, videoWidth, videoHeight]);

  const renderRoi = (key: string) => {
    const roi = rois[key as keyof VodRoiConfig];
    const left = roi.x1 * scale;
    const top = roi.y1 * scale;
    const width = (roi.x2 - roi.x1) * scale;
    const height = (roi.y2 - roi.y1) * scale;
    const handleSize = 6;

    return (
      <div key={key}>
        {/* ROI rectangle */}
        <div
          style={{
            position: "absolute",
            left, top, width, height,
            backgroundColor: ROI_COLORS[key],
            border: `2px solid ${ROI_BORDER_COLORS[key]}`,
            cursor: "move",
          }}
          onMouseDown={handleMouseDown(key, "move")}
        >
          <span
            style={{
              position: "absolute",
              top: -16,
              left: 0,
              fontSize: 10,
              color: ROI_BORDER_COLORS[key],
              whiteSpace: "nowrap",
              fontWeight: 600,
              textShadow: "0 0 3px rgba(0,0,0,0.8)",
            }}
          >
            {ROI_LABELS[key]}
          </span>
        </div>
        {/* Corner handles */}
        {(["tl", "tr", "bl", "br"] as const).map((corner) => {
          const cx = corner.includes("l") ? left - handleSize / 2 : left + width - handleSize / 2;
          const cy = corner.includes("t") ? top - handleSize / 2 : top + height - handleSize / 2;
          const cursor = corner === "tl" || corner === "br" ? "nwse-resize" : "nesw-resize";
          return (
            <div
              key={corner}
              style={{
                position: "absolute",
                left: cx, top: cy,
                width: handleSize, height: handleSize,
                backgroundColor: ROI_BORDER_COLORS[key],
                cursor,
              }}
              onMouseDown={handleMouseDown(key, "resize", corner)}
            />
          );
        })}
      </div>
    );
  };

  return (
    <div className="h-screen flex flex-col bg-surface-0">
      <div className="flex items-center justify-between p-4 border-b border-surface-4/30">
        <h3 className="text-sm font-semibold text-text-primary">
          Adjust Detection Regions
        </h3>
        <div className="flex gap-2">
          <button onClick={onBack} className="btn-ghost text-xs">Back</button>
          <button onClick={() => onConfirm(rois)} className="btn-primary text-xs">
            Start Scan
          </button>
        </div>
      </div>

      <div className="flex-1 flex items-center justify-center p-4 overflow-hidden">
        <div
          ref={containerRef}
          className="relative inline-block max-w-full max-h-full"
          style={{ userSelect: "none" }}
        >
          <img
            src={imageSrc}
            onLoad={onImgLoad}
            className="max-w-full max-h-[calc(100vh-160px)] object-contain"
            draggable={false}
          />
          {imgSize.w > 0 && (
            <>
              {renderRoi("p1_tension")}
              {renderRoi("p2_tension")}
              {renderRoi("timer")}
              {renderRoi("banner")}
            </>
          )}
        </div>
      </div>

      <div className="px-4 py-2 border-t border-surface-4/30 text-[10px] text-text-muted flex gap-6">
        <span>Drag rectangles to move. Drag corners to resize.</span>
        <span className="text-blue-400">P1 Tension</span>
        <span className="text-red-400">P2 Tension</span>
        <span className="text-yellow-400">Timer</span>
        <span className="text-purple-400">Banner</span>
      </div>
    </div>
  );
}
