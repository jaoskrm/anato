import React, { useRef, useEffect, useState, useCallback } from "react";
import { SEG_CLASSES, type SegmentationState } from "@/hooks/useSegmentationState";

interface Props {
  state: SegmentationState;
  canvasRef: React.RefObject<HTMLCanvasElement>;
  maskCanvasRef: React.RefObject<HTMLCanvasElement>;
  imageRef: React.RefObject<HTMLImageElement | null>;
}

/**
 * Find approximate centroids for each class color present on the mask canvas.
 * We sample every Nth pixel for performance.
 */
function computeClassCentroids(maskCanvas: HTMLCanvasElement): { classIdx: number; cx: number; cy: number }[] {
  const ctx = maskCanvas.getContext("2d");
  if (!ctx) return [];
  const { width, height } = maskCanvas;
  const imageData = ctx.getImageData(0, 0, width, height);
  const data = imageData.data;

  // Map each class to its accumulated x, y, count
  const accum: { sx: number; sy: number; count: number }[] = SEG_CLASSES.map(() => ({ sx: 0, sy: 0, count: 0 }));

  // Parse class colors to [r, g, b]
  const classRgb = SEG_CLASSES.map((c) => {
    const m = c.color.match(/\d+/g);
    return m ? [Number(m[0]), Number(m[1]), Number(m[2])] : [0, 0, 0];
  });

  const step = Math.max(1, Math.floor(Math.min(width, height) / 200)); // sample every few pixels

  for (let y = 0; y < height; y += step) {
    for (let x = 0; x < width; x += step) {
      const idx = (y * width + x) * 4;
      const r = data[idx], g = data[idx + 1], b = data[idx + 2], a = data[idx + 3];
      if (a < 50) continue; // skip transparent

      // Match to closest class
      for (let ci = 0; ci < classRgb.length; ci++) {
        const [cr, cg, cb] = classRgb[ci];
        if (Math.abs(r - cr) < 30 && Math.abs(g - cg) < 30 && Math.abs(b - cb) < 30) {
          accum[ci].sx += x;
          accum[ci].sy += y;
          accum[ci].count++;
          break;
        }
      }
    }
  }

  return accum
    .map((a, i) => ({ classIdx: i, cx: a.count > 0 ? a.sx / a.count : 0, cy: a.count > 0 ? a.sy / a.count : 0, count: a.count }))
    .filter((a) => a.count > 0);
}

const SegmentationCanvas = ({ state, canvasRef, maskCanvasRef, imageRef }: Props) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [drawing, setDrawing] = useState(false);
  const [boxStart, setBoxStart] = useState<{ x: number; y: number } | null>(null);
  const [boxEnd, setBoxEnd] = useState<{ x: number; y: number } | null>(null);
  const [outlinePoints, setOutlinePoints] = useState<{ x: number; y: number }[]>([]);

  const getCanvasPoint = useCallback(
    (e: React.MouseEvent) => {
      const canvas = canvasRef.current;
      if (!canvas) return { x: 0, y: 0 };
      const rect = canvas.getBoundingClientRect();
      return {
        x: (e.clientX - rect.left) / state.zoom,
        y: (e.clientY - rect.top) / state.zoom,
      };
    },
    [state.zoom, canvasRef]
  );

  // Redraw main canvas
  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    const maskCanvas = maskCanvasRef.current;
    const img = imageRef.current;
    if (!canvas || !maskCanvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = maskCanvas.width;
    canvas.height = maskCanvas.height;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw image
    if (img && img.complete && img.naturalWidth > 0) {
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    } else {
      // Placeholder grid
      ctx.fillStyle = "#111";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = "#222";
      for (let x = 0; x < canvas.width; x += 40) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
      }
      for (let y = 0; y < canvas.height; y += 40) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
      }
      ctx.fillStyle = "#444";
      ctx.font = "16px JetBrains Mono";
      ctx.textAlign = "center";
      ctx.fillText("Import an image to begin", canvas.width / 2, canvas.height / 2);
    }

    // Draw mask overlay
    if (state.showOverlay) {
      ctx.globalAlpha = 0.4;
      ctx.drawImage(maskCanvas, 0, 0);
      ctx.globalAlpha = 1;
    }

    // Class labels at region centroids
    if (state.showLabels && state.showOverlay) {
      const centroids = computeClassCentroids(maskCanvas);
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      centroids.forEach(({ classIdx, cx, cy }) => {
        const cls = SEG_CLASSES[classIdx];
        const label = cls.name;
        ctx.font = "bold 14px JetBrains Mono";
        const metrics = ctx.measureText(label);
        const pad = 4;
        // Background pill
        ctx.fillStyle = "rgba(0, 0, 0, 0.65)";
        ctx.beginPath();
        ctx.roundRect(
          cx - metrics.width / 2 - pad,
          cy - 8 - pad,
          metrics.width + pad * 2,
          16 + pad * 2,
          4
        );
        ctx.fill();
        // Text
        ctx.fillStyle = cls.color;
        ctx.fillText(label, cx, cy);
      });
    }

    // Box selection preview
    if (boxStart && boxEnd && state.toolMode === "box") {
      ctx.strokeStyle = "red";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 3]);
      ctx.strokeRect(
        boxStart.x, boxStart.y,
        boxEnd.x - boxStart.x, boxEnd.y - boxStart.y
      );
      ctx.setLineDash([]);
    }

    // Outline preview
    if (outlinePoints.length > 0 && state.toolMode === "outline") {
      ctx.strokeStyle = SEG_CLASSES[state.activeClass].color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(outlinePoints[0].x, outlinePoints[0].y);
      outlinePoints.forEach((p) => ctx.lineTo(p.x, p.y));
      ctx.stroke();
    }
  }, [state.showOverlay, state.showLabels, state.toolMode, state.activeClass, boxStart, boxEnd, outlinePoints, canvasRef, maskCanvasRef, imageRef]);

  useEffect(() => {
    redraw();
  }, [redraw, state.currentFrame, state.zoom, state.showOverlay, state.showLabels]);

  const paintOnMask = useCallback(
    (x: number, y: number) => {
      const maskCanvas = maskCanvasRef.current;
      if (!maskCanvas) return;
      const ctx = maskCanvas.getContext("2d");
      if (!ctx) return;

      if (state.toolMode === "eraser") {
        ctx.globalCompositeOperation = "destination-out";
        ctx.beginPath();
        ctx.arc(x, y, state.brushSize, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalCompositeOperation = "source-over";
      } else if (state.toolMode === "paint") {
        ctx.fillStyle = SEG_CLASSES[state.activeClass].color;
        ctx.beginPath();
        ctx.arc(x, y, state.brushSize, 0, Math.PI * 2);
        ctx.fill();
      }
    },
    [state.toolMode, state.brushSize, state.activeClass, maskCanvasRef]
  );

  const handleMouseDown = (e: React.MouseEvent) => {
    const pt = getCanvasPoint(e);
    setDrawing(true);

    if (state.toolMode === "box") {
      setBoxStart(pt);
      setBoxEnd(pt);
    } else if (state.toolMode === "outline") {
      setOutlinePoints((prev) => [...prev, pt]);
    } else {
      // Push undo before starting to paint/erase
      state.pushUndo(maskCanvasRef.current);
      paintOnMask(pt.x, pt.y);
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!drawing && state.toolMode !== "outline") return;
    const pt = getCanvasPoint(e);

    if (state.toolMode === "box" && drawing) {
      setBoxEnd(pt);
      redraw();
    } else if (drawing && (state.toolMode === "paint" || state.toolMode === "eraser")) {
      paintOnMask(pt.x, pt.y);
      redraw();
    }
  };

  const handleMouseUp = () => {
    if (state.toolMode === "box" && boxStart && boxEnd) {
      // Box prompt complete — API call placeholder
      console.log("Box prompt:", boxStart, boxEnd, SEG_CLASSES[state.activeClass].name);
      state.pushUndo(maskCanvasRef.current);
    }
    // Paint/erase undo was pushed on mouseDown, no need to push again
    setDrawing(false);
    setBoxStart(null);
    setBoxEnd(null);
  };

  const handleDoubleClick = () => {
    if (state.toolMode === "outline" && outlinePoints.length >= 3) {
      // Fill polygon on mask
      const maskCanvas = maskCanvasRef.current;
      if (!maskCanvas) return;
      const ctx = maskCanvas.getContext("2d");
      if (!ctx) return;

      state.pushUndo(maskCanvas);

      ctx.fillStyle = SEG_CLASSES[state.activeClass].color;
      ctx.beginPath();
      ctx.moveTo(outlinePoints[0].x, outlinePoints[0].y);
      outlinePoints.forEach((p) => ctx.lineTo(p.x, p.y));
      ctx.closePath();
      ctx.fill();

      setOutlinePoints([]);
      redraw();
    }
  };

  const cursorStyle = (() => {
    if (state.toolMode === "paint" || state.toolMode === "eraser") return "crosshair";
    if (state.toolMode === "box") return "crosshair";
    if (state.toolMode === "outline") return "crosshair";
    return "default";
  })();

  return (
    <div
      ref={containerRef}
      className="flex-1 overflow-auto bg-background flex items-center justify-center p-4"
    >
      <div
        style={{
          transform: `scale(${state.zoom})`,
          transformOrigin: "center center",
          transition: "transform 0.15s ease",
        }}
      >
        <canvas
          ref={canvasRef}
          width={800}
          height={600}
          className="border border-border rounded-sm"
          style={{ cursor: cursorStyle }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={() => { if (drawing) handleMouseUp(); }}
          onDoubleClick={handleDoubleClick}
        />
        {/* Hidden mask canvas */}
        <canvas
          ref={maskCanvasRef}
          width={800}
          height={600}
          className="hidden"
        />
      </div>
    </div>
  );
};

export default SegmentationCanvas;
