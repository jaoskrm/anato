import { useState, useCallback, useRef } from "react";

export type SegClass = {
  name: string;
  color: string; // CSS color string
  key: string;
};

export const SEG_CLASSES: SegClass[] = [
  { name: "Uterus", color: "rgb(0, 255, 255)", key: "uterus" },
  { name: "Tools", color: "rgb(255, 0, 0)", key: "tools" },
  { name: "Ureter", color: "rgb(255, 255, 0)", key: "ureter" },
  { name: "Ovary", color: "rgb(0, 200, 0)", key: "ovary" },
  { name: "Vessel", color: "rgb(255, 0, 255)", key: "vessel" },
];

export type ToolMode = "box" | "paint" | "outline" | "eraser";

const MAX_HISTORY = 50;

export interface SegmentationState {
  activeClass: number;
  setActiveClass: (i: number) => void;
  toolMode: ToolMode;
  setToolMode: (m: ToolMode) => void;
  brushSize: number;
  setBrushSize: (s: number) => void;
  showOverlay: boolean;
  toggleOverlay: () => void;
  showLabels: boolean;
  setShowLabels: (v: boolean) => void;
  autoOutline: boolean;
  setAutoOutline: (v: boolean) => void;
  currentFrame: number;
  setCurrentFrame: (n: number) => void;
  totalFrames: number;
  filename: string;
  setFilename: (f: string) => void;
  setTotalFrames: (n: number) => void;
  undoCount: number;
  redoCount: number;
  pushUndo: (maskCanvas: HTMLCanvasElement | null) => void;
  undo: (maskCanvas: HTMLCanvasElement | null) => void;
  redo: (maskCanvas: HTMLCanvasElement | null) => void;
  zoom: number;
  setZoom: (z: number) => void;
  imageLoaded: boolean;
  setImageLoaded: (v: boolean) => void;
}

export function useSegmentationState(): SegmentationState {
  const [activeClass, setActiveClass] = useState(0);
  const [toolMode, setToolMode] = useState<ToolMode>("box");
  const [brushSize, setBrushSize] = useState(10);
  const [showOverlay, setShowOverlay] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [autoOutline, setAutoOutline] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [totalFrames, setTotalFrames] = useState(0);
  const [filename, setFilename] = useState("No file loaded");
  const [zoom, setZoom] = useState(1);
  const [imageLoaded, setImageLoaded] = useState(false);

  // True undo/redo with ImageData history
  const undoStackRef = useRef<ImageData[]>([]);
  const redoStackRef = useRef<ImageData[]>([]);
  const [undoCount, setUndoCount] = useState(0);
  const [redoCount, setRedoCount] = useState(0);

  const toggleOverlay = useCallback(() => setShowOverlay((v) => !v), []);

  const pushUndo = useCallback((maskCanvas: HTMLCanvasElement | null) => {
    if (!maskCanvas) return;
    const ctx = maskCanvas.getContext("2d");
    if (!ctx) return;
    const imageData = ctx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    undoStackRef.current.push(imageData);
    if (undoStackRef.current.length > MAX_HISTORY) {
      undoStackRef.current.shift();
    }
    redoStackRef.current = [];
    setUndoCount(undoStackRef.current.length);
    setRedoCount(0);
  }, []);

  const undo = useCallback((maskCanvas: HTMLCanvasElement | null) => {
    if (!maskCanvas || undoStackRef.current.length === 0) return;
    const ctx = maskCanvas.getContext("2d");
    if (!ctx) return;
    // Save current state to redo stack
    const currentData = ctx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    redoStackRef.current.push(currentData);
    // Restore previous state
    const prevData = undoStackRef.current.pop()!;
    ctx.putImageData(prevData, 0, 0);
    setUndoCount(undoStackRef.current.length);
    setRedoCount(redoStackRef.current.length);
  }, []);

  const redo = useCallback((maskCanvas: HTMLCanvasElement | null) => {
    if (!maskCanvas || redoStackRef.current.length === 0) return;
    const ctx = maskCanvas.getContext("2d");
    if (!ctx) return;
    // Save current state to undo stack
    const currentData = ctx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    undoStackRef.current.push(currentData);
    // Restore redo state
    const nextData = redoStackRef.current.pop()!;
    ctx.putImageData(nextData, 0, 0);
    setUndoCount(undoStackRef.current.length);
    setRedoCount(redoStackRef.current.length);
  }, []);

  return {
    activeClass, setActiveClass,
    toolMode, setToolMode,
    brushSize, setBrushSize,
    showOverlay, toggleOverlay,
    showLabels, setShowLabels,
    autoOutline, setAutoOutline,
    currentFrame, setCurrentFrame,
    totalFrames, setTotalFrames,
    filename, setFilename,
    undoCount, redoCount,
    pushUndo, undo, redo,
    zoom, setZoom,
    imageLoaded, setImageLoaded,
  };
}
