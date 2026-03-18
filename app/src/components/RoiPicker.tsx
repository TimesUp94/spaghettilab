import { useState, useRef, useCallback, useEffect } from "react";
import type { RoiRect, RoiQuad, VodRoiConfig } from "../types";

// Default ROIs for 1920x1080 GGS with "SHOW DOWN" overlay
const DEFAULT_ROIS: VodRoiConfig = {
  p1_tension: { y1: 1040, y2: 1058, x1: 50, x2: 440 },
  p2_tension: { y1: 1040, y2: 1058, x1: 1480, x2: 1870 },
  timer: { y1: 30, y2: 100, x1: 910, x2: 1010 },
  p1_name: { tl: [75, 150], tr: [340, 145], br: [370, 178], bl: [45, 178] },
  p2_name: { tl: [1555, 145], tr: [1850, 150], br: [1880, 178], bl: [1580, 178] },
};

const ROI_COLORS: Record<string, string> = {
  p1_tension: "rgba(59, 130, 246, 0.4)",
  p2_tension: "rgba(239, 68, 68, 0.4)",
  timer: "rgba(250, 204, 21, 0.4)",
  p1_name: "rgba(59, 130, 246, 0.3)",
  p2_name: "rgba(239, 68, 68, 0.3)",
};

const ROI_BORDER_COLORS: Record<string, string> = {
  p1_tension: "rgb(59, 130, 246)",
  p2_tension: "rgb(239, 68, 68)",
  timer: "rgb(250, 204, 21)",
  p1_name: "rgb(96, 165, 250)",
  p2_name: "rgb(248, 113, 113)",
};

const ROI_LABELS: Record<string, string> = {
  p1_tension: "P1 Tension",
  p2_tension: "P2 Tension",
  timer: "Timer",
  p1_name: "P1 Name",
  p2_name: "P2 Name",
};

const QUAD_KEYS = ["p1_name", "p2_name"] as const;
const RECT_KEYS = ["p1_tension", "p2_tension", "timer"] as const;

type QuadCorner = "tl" | "tr" | "br" | "bl";

interface Props {
  imageSrc: string;
  videoWidth: number;
  videoHeight: number;
  videoDurationSecs?: number;
  onSeekPreview?: (timestampSecs: number) => void;
  onConfirm: (config: VodRoiConfig) => void;
  onBack: () => void;
}

type RectDragState = {
  kind: "rect";
  key: string;
  type: "move" | "resize";
  corner?: "tl" | "tr" | "bl" | "br";
  startX: number;
  startY: number;
  origRoi: RoiRect;
};

type QuadDragState = {
  kind: "quad";
  key: string;
  type: "corner" | "move";
  corner?: QuadCorner;
  startX: number;
  startY: number;
  origQuad: RoiQuad;
};

type DragState = RectDragState | QuadDragState;

export function RoiPicker({ imageSrc, videoWidth, videoHeight, videoDurationSecs, onSeekPreview, onConfirm, onBack }: Props) {
  const [rois, setRois] = useState<VodRoiConfig>({ ...DEFAULT_ROIS });
  const [drag, setDrag] = useState<DragState | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 });
  const [previewTime, setPreviewTime] = useState(30);

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

  // Mouse down for rect ROIs (tension, timer)
  const handleRectMouseDown = useCallback(
    (key: string, type: "move" | "resize", corner?: "tl" | "tr" | "bl" | "br") =>
      (e: React.MouseEvent) => {
        e.stopPropagation();
        e.preventDefault();
        const pos = getMousePos(e);
        const roi = rois[key as keyof VodRoiConfig] as RoiRect;
        setDrag({ kind: "rect", key, type, corner, startX: pos.x, startY: pos.y, origRoi: { ...roi } });
      },
    [getMousePos, rois]
  );

  // Mouse down for quad ROIs (name regions)
  const handleQuadMouseDown = useCallback(
    (key: string, type: "corner" | "move", corner?: QuadCorner) =>
      (e: React.MouseEvent) => {
        e.stopPropagation();
        e.preventDefault();
        const pos = getMousePos(e);
        const quad = rois[key as keyof VodRoiConfig] as RoiQuad;
        setDrag({
          kind: "quad",
          key,
          type,
          corner,
          startX: pos.x,
          startY: pos.y,
          origQuad: { tl: [...quad.tl], tr: [...quad.tr], br: [...quad.br], bl: [...quad.bl] },
        });
      },
    [getMousePos, rois]
  );

  useEffect(() => {
    if (!drag) return;

    const onMove = (e: MouseEvent) => {
      const pos = getMousePos(e);
      const dx = pos.x - drag.startX;
      const dy = pos.y - drag.startY;

      setRois((prev) => {
        const updated = { ...prev };

        if (drag.kind === "rect") {
          const orig = drag.origRoi;
          if (drag.type === "move") {
            const w = orig.x2 - orig.x1;
            const h = orig.y2 - orig.y1;
            let nx1 = Math.round(orig.x1 + dx);
            let ny1 = Math.round(orig.y1 + dy);
            nx1 = Math.max(0, Math.min(videoWidth - w, nx1));
            ny1 = Math.max(0, Math.min(videoHeight - h, ny1));
            (updated as any)[drag.key] = { y1: ny1, y2: ny1 + h, x1: nx1, x2: nx1 + w };
          } else if (drag.type === "resize" && drag.corner) {
            let { y1, y2, x1, x2 } = orig;
            if (drag.corner.includes("l")) x1 = Math.max(0, Math.round(orig.x1 + dx));
            if (drag.corner.includes("r")) x2 = Math.min(videoWidth, Math.round(orig.x2 + dx));
            if (drag.corner.includes("t")) y1 = Math.max(0, Math.round(orig.y1 + dy));
            if (drag.corner.includes("b")) y2 = Math.min(videoHeight, Math.round(orig.y2 + dy));
            if (x2 - x1 >= 10 && y2 - y1 >= 4) {
              (updated as any)[drag.key] = { y1, y2, x1, x2 };
            }
          }
        } else if (drag.kind === "quad") {
          const orig = drag.origQuad;
          if (drag.type === "move") {
            // Move all 4 points by the same delta
            const clampX = (v: number) => Math.max(0, Math.min(videoWidth, Math.round(v)));
            const clampY = (v: number) => Math.max(0, Math.min(videoHeight, Math.round(v)));
            (updated as any)[drag.key] = {
              tl: [clampX(orig.tl[0] + dx), clampY(orig.tl[1] + dy)] as [number, number],
              tr: [clampX(orig.tr[0] + dx), clampY(orig.tr[1] + dy)] as [number, number],
              br: [clampX(orig.br[0] + dx), clampY(orig.br[1] + dy)] as [number, number],
              bl: [clampX(orig.bl[0] + dx), clampY(orig.bl[1] + dy)] as [number, number],
            };
          } else if (drag.type === "corner" && drag.corner) {
            // Move just one corner
            const newQuad = {
              tl: [...orig.tl] as [number, number],
              tr: [...orig.tr] as [number, number],
              br: [...orig.br] as [number, number],
              bl: [...orig.bl] as [number, number],
            };
            const c = drag.corner;
            newQuad[c] = [
              Math.max(0, Math.min(videoWidth, Math.round(orig[c][0] + dx))),
              Math.max(0, Math.min(videoHeight, Math.round(orig[c][1] + dy))),
            ];
            (updated as any)[drag.key] = newQuad;
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
    const roi = rois[key as keyof VodRoiConfig] as RoiRect;
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
          onMouseDown={handleRectMouseDown(key, "move")}
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
              onMouseDown={handleRectMouseDown(key, "resize", corner)}
            />
          );
        })}
      </div>
    );
  };

  const renderQuad = (key: string) => {
    const quad = rois[key as keyof VodRoiConfig] as RoiQuad;
    const corners: QuadCorner[] = ["tl", "tr", "br", "bl"];
    const handleSize = 8;

    // Scale all points for display
    const pts = corners.map((c) => [quad[c][0] * scale, quad[c][1] * scale] as [number, number]);
    const polyPoints = pts.map((p) => `${p[0]},${p[1]}`).join(" ");

    // Compute bounding box for SVG viewBox and label placement
    const minX = Math.min(...pts.map((p) => p[0]));
    const minY = Math.min(...pts.map((p) => p[1]));

    return (
      <g key={key}>
        {/* Filled polygon */}
        <polygon
          points={polyPoints}
          fill={ROI_COLORS[key]}
          stroke={ROI_BORDER_COLORS[key]}
          strokeWidth={2}
          style={{ cursor: "move" }}
          onMouseDown={handleQuadMouseDown(key, "move")}
        />
        {/* Label */}
        <text
          x={minX}
          y={minY - 4}
          fill={ROI_BORDER_COLORS[key]}
          fontSize={10}
          fontWeight={600}
          style={{ textShadow: "0 0 3px rgba(0,0,0,0.8)", pointerEvents: "none" }}
        >
          {ROI_LABELS[key]}
        </text>
        {/* Corner handles */}
        {corners.map((c, i) => (
          <rect
            key={c}
            x={pts[i][0] - handleSize / 2}
            y={pts[i][1] - handleSize / 2}
            width={handleSize}
            height={handleSize}
            fill={ROI_BORDER_COLORS[key]}
            style={{ cursor: "crosshair" }}
            onMouseDown={handleQuadMouseDown(key, "corner", c)}
          />
        ))}
      </g>
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
              {/* SVG overlay for quad ROIs */}
              <svg
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: imgSize.w,
                  height: imgSize.h,
                  pointerEvents: "none",
                  overflow: "visible",
                }}
              >
                <g style={{ pointerEvents: "auto" }}>
                  {renderQuad("p1_name")}
                  {renderQuad("p2_name")}
                </g>
              </svg>
            </>
          )}
        </div>
      </div>

      <div className="px-4 py-2 border-t border-surface-4/30 flex flex-col gap-1.5">
        {videoDurationSecs && onSeekPreview && (
          <div className="flex items-center gap-3">
            <span className="text-[10px] text-text-muted shrink-0">Preview at:</span>
            <input
              type="range"
              min={0}
              max={Math.floor(videoDurationSecs)}
              value={previewTime}
              onChange={(e) => setPreviewTime(Number(e.target.value))}
              onMouseUp={() => onSeekPreview(previewTime)}
              onKeyUp={() => onSeekPreview(previewTime)}
              className="flex-1 h-1 accent-accent-purple cursor-pointer"
            />
            <span className="text-[11px] text-text-secondary font-mono w-16 text-right">
              {Math.floor(previewTime / 60)}:{String(Math.floor(previewTime % 60)).padStart(2, "0")}
              <span className="text-text-muted"> / {Math.floor(videoDurationSecs / 60)}:{String(Math.floor(videoDurationSecs % 60)).padStart(2, "0")}</span>
            </span>
          </div>
        )}
        <div className="text-[10px] text-text-muted flex gap-6">
          <span>Drag regions to move. Drag corners to resize/reshape. Use slider to find gameplay.</span>
          <span className="text-blue-400">P1 Name / Tension</span>
          <span className="text-red-400">P2 Name / Tension</span>
          <span className="text-yellow-400">Timer</span>
        </div>
      </div>
    </div>
  );
}
