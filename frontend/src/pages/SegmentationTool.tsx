import React, { useRef, useEffect, useCallback } from "react";
import ToolSidebar from "@/components/ToolSidebar";
import SegmentationCanvas from "@/components/SegmentationCanvas";
import { useSegmentationState } from "@/hooks/useSegmentationState";
import { toast } from "sonner";

const SegmentationTool = () => {
  const state = useSegmentationState();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImportImage = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;

      const img = new Image();
      img.onload = () => {
        imageRef.current = img;

        // Resize canvases to image size
        if (canvasRef.current) {
          canvasRef.current.width = img.naturalWidth;
          canvasRef.current.height = img.naturalHeight;
        }
        if (maskCanvasRef.current) {
          maskCanvasRef.current.width = img.naturalWidth;
          maskCanvasRef.current.height = img.naturalHeight;
          // Clear mask
          const ctx = maskCanvasRef.current.getContext("2d");
          ctx?.clearRect(0, 0, img.naturalWidth, img.naturalHeight);
        }

        state.setFilename(file.name);
        state.setImageLoaded(true);
        state.setTotalFrames(1);
        state.setCurrentFrame(0);

        // Auto-fit zoom
        const containerW = window.innerWidth - 260;
        const containerH = window.innerHeight;
        const scale = Math.min(containerW / img.naturalWidth, containerH / img.naturalHeight, 1);
        state.setZoom(Math.round(scale * 4) / 4); // snap to 0.25 increments
      };
      img.src = URL.createObjectURL(file);
      e.target.value = "";
    },
    [state]
  );

  const handleClearMask = useCallback(() => {
    const maskCanvas = maskCanvasRef.current;
    if (!maskCanvas) return;
    state.pushUndo(maskCanvas);
    const ctx = maskCanvas.getContext("2d");
    ctx?.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    // Trigger redraw
    state.setCurrentFrame(state.currentFrame);
    toast.info("Mask cleared");
  }, [state]);

  const handleSave = useCallback(() => {
    // Placeholder — would send mask data to backend
    toast.success("Mask saved (placeholder)");
    console.log("Save triggered for frame", state.currentFrame);
  }, [state.currentFrame]);

  const handleUndo = useCallback(() => {
    state.undo(maskCanvasRef.current);
    state.setCurrentFrame(state.currentFrame); // trigger redraw
  }, [state]);

  const handleRedo = useCallback(() => {
    state.redo(maskCanvasRef.current);
    state.setCurrentFrame(state.currentFrame); // trigger redraw
  }, [state]);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        state.setCurrentFrame(Math.max(0, state.currentFrame - 1));
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        state.setCurrentFrame(state.currentFrame + 1);
      } else if (e.key === " ") {
        e.preventDefault();
        state.toggleOverlay();
      } else if ((e.ctrlKey || e.metaKey) && e.key === "z") {
        e.preventDefault();
        handleUndo();
      } else if ((e.ctrlKey || e.metaKey) && e.key === "y") {
        e.preventDefault();
        handleRedo();
      } else if ((e.ctrlKey || e.metaKey) && e.key === "s") {
        e.preventDefault();
        handleSave();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [state, handleSave, handleUndo, handleRedo]);

  return (
    <div className="dark h-screen w-screen flex overflow-hidden bg-background text-foreground">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleFileChange}
      />
      <ToolSidebar
        state={state}
        onClearMask={handleClearMask}
        onSave={handleSave}
        onImportImage={handleImportImage}
        onUndo={handleUndo}
        onRedo={handleRedo}
      />
      <SegmentationCanvas
        state={state}
        canvasRef={canvasRef}
        maskCanvasRef={maskCanvasRef}
        imageRef={imageRef}
      />
    </div>
  );
};

export default SegmentationTool;
