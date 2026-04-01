import { useRef, useEffect, useCallback } from "react";
import type { Drawing, DrawingStroke } from "../types";

interface Props {
  currentTimeMs: number;
  drawings: Drawing[];
  drawingMode: boolean;
  activeTool: "pen" | "eraser";
  penColor: string;
  penSize: number;
  onSaveDrawing: (timestampMs: number, strokesJson: string) => void;
  containerRef: React.RefObject<HTMLDivElement | null>;
}

const FADE_DURATION_MS = 4000;
const FADE_START_MS = 3000;
const MIN_POINT_DIST = 0.002; // normalized distance threshold for downsampling

export function DrawingOverlay({
  currentTimeMs,
  drawings,
  drawingMode,
  activeTool,
  penColor,
  penSize,
  onSaveDrawing,
  containerRef,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);

  // Active drawing state (when in drawing mode)
  const activeStrokesRef = useRef<DrawingStroke[]>([]);
  const currentStrokeRef = useRef<DrawingStroke | null>(null);
  const drawingTimestampRef = useRef<number>(0);
  const isDrawingRef = useRef(false);

  // Sync canvas size with container
  const syncSize = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;
    const rect = container.getBoundingClientRect();
    if (canvas.width !== rect.width || canvas.height !== rect.height) {
      canvas.width = rect.width;
      canvas.height = rect.height;
    }
  }, [containerRef]);

  // Render a set of strokes to an offscreen canvas
  const renderStrokes = useCallback(
    (strokes: DrawingStroke[], width: number, height: number): HTMLCanvasElement => {
      const offscreen = document.createElement("canvas");
      offscreen.width = width;
      offscreen.height = height;
      const ctx = offscreen.getContext("2d")!;
      const scale = width / 1920; // reference width

      for (const stroke of strokes) {
        if (stroke.points.length < 2) continue;
        ctx.lineWidth = stroke.size * scale;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";

        if (stroke.isEraser) {
          ctx.globalCompositeOperation = "destination-out";
          ctx.strokeStyle = "rgba(0,0,0,1)";
        } else {
          ctx.globalCompositeOperation = "source-over";
          ctx.strokeStyle = stroke.color;
        }

        ctx.beginPath();
        ctx.moveTo(stroke.points[0][0] * width, stroke.points[0][1] * height);
        for (let i = 1; i < stroke.points.length; i++) {
          ctx.lineTo(stroke.points[i][0] * width, stroke.points[i][1] * height);
        }
        ctx.stroke();
      }

      return offscreen;
    },
    []
  );

  // Main render loop
  useEffect(() => {
    const render = () => {
      const canvas = canvasRef.current;
      if (!canvas) {
        rafRef.current = requestAnimationFrame(render);
        return;
      }

      syncSize();
      const ctx = canvas.getContext("2d")!;
      const { width, height } = canvas;
      ctx.clearRect(0, 0, width, height);

      const currentMs = currentTimeMs;

      // Render saved drawings with fade
      for (const drawing of drawings) {
        const elapsed = currentMs - drawing.timestamp_ms;
        if (elapsed < 0 || elapsed > FADE_DURATION_MS) continue;

        const opacity =
          elapsed < FADE_START_MS
            ? 1
            : 1 - (elapsed - FADE_START_MS) / (FADE_DURATION_MS - FADE_START_MS);

        let strokes: DrawingStroke[];
        try {
          strokes = JSON.parse(drawing.strokes_json);
        } catch {
          continue;
        }

        const offscreen = renderStrokes(strokes, width, height);
        ctx.globalAlpha = opacity;
        ctx.drawImage(offscreen, 0, 0);
        ctx.globalAlpha = 1;
      }

      // Render active strokes (in drawing mode) at full opacity
      if (drawingMode) {
        const allActive = [
          ...activeStrokesRef.current,
          ...(currentStrokeRef.current ? [currentStrokeRef.current] : []),
        ];
        if (allActive.length > 0) {
          const offscreen = renderStrokes(allActive, width, height);
          ctx.drawImage(offscreen, 0, 0);
        }
      }

      rafRef.current = requestAnimationFrame(render);
    };

    rafRef.current = requestAnimationFrame(render);
    return () => cancelAnimationFrame(rafRef.current);
  }, [drawings, drawingMode, currentTimeMs, syncSize, renderStrokes]);

  // When entering drawing mode: snapshot timestamp, load existing strokes
  useEffect(() => {
    if (drawingMode) {
      const ts = currentTimeMs;
      drawingTimestampRef.current = ts;

      // Load existing strokes at this timestamp (find closest within 50ms)
      const existing = drawings.find(
        (d) => Math.abs(d.timestamp_ms - ts) < 50
      );
      if (existing) {
        try {
          activeStrokesRef.current = JSON.parse(existing.strokes_json);
          drawingTimestampRef.current = existing.timestamp_ms;
        } catch {
          activeStrokesRef.current = [];
        }
      } else {
        activeStrokesRef.current = [];
      }
    } else {
      // Exiting drawing mode — save
      if (activeStrokesRef.current.length > 0 || drawingTimestampRef.current > 0) {
        const strokes = activeStrokesRef.current;
        const json = strokes.length > 0 ? JSON.stringify(strokes) : "[]";
        onSaveDrawing(drawingTimestampRef.current, json);
      }
      activeStrokesRef.current = [];
      currentStrokeRef.current = null;
    }
  }, [drawingMode]); // eslint-disable-line react-hooks/exhaustive-deps

  const getNormalizedPoint = useCallback(
    (e: React.PointerEvent): [number, number] => {
      const canvas = canvasRef.current!;
      const rect = canvas.getBoundingClientRect();
      return [
        (e.clientX - rect.left) / rect.width,
        (e.clientY - rect.top) / rect.height,
      ];
    },
    []
  );

  const handlePointerDown = useCallback(
    (e: React.PointerEvent) => {
      if (!drawingMode) return;
      e.preventDefault();
      e.stopPropagation();
      isDrawingRef.current = true;
      (e.target as HTMLElement).setPointerCapture(e.pointerId);

      const pt = getNormalizedPoint(e);
      currentStrokeRef.current = {
        points: [pt],
        color: activeTool === "eraser" ? "#000000" : penColor,
        size: activeTool === "eraser" ? penSize * 3 : penSize,
        isEraser: activeTool === "eraser",
      };
    },
    [drawingMode, activeTool, penColor, penSize, getNormalizedPoint]
  );

  const handlePointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!isDrawingRef.current || !currentStrokeRef.current) return;
      e.preventDefault();
      e.stopPropagation();

      const pt = getNormalizedPoint(e);
      const pts = currentStrokeRef.current.points;
      const last = pts[pts.length - 1];
      const dx = pt[0] - last[0];
      const dy = pt[1] - last[1];
      if (dx * dx + dy * dy < MIN_POINT_DIST * MIN_POINT_DIST) return;

      currentStrokeRef.current.points.push(pt);
    },
    [getNormalizedPoint]
  );

  const handlePointerUp = useCallback(
    (e: React.PointerEvent) => {
      if (!isDrawingRef.current) return;
      e.preventDefault();
      e.stopPropagation();
      isDrawingRef.current = false;

      if (currentStrokeRef.current && currentStrokeRef.current.points.length >= 2) {
        activeStrokesRef.current = [
          ...activeStrokesRef.current,
          currentStrokeRef.current,
        ];
      }
      currentStrokeRef.current = null;
    },
    []
  );

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full"
      style={{
        pointerEvents: drawingMode ? "auto" : "none",
        cursor: drawingMode ? "crosshair" : "default",
        touchAction: "none",
      }}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
    />
  );
}
