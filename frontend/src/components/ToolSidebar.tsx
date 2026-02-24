import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  SEG_CLASSES, type SegmentationState, type ToolMode,
} from "@/hooks/useSegmentationState";
import {
  Undo2, Redo2, Eye, EyeOff, Trash2, Save, Upload,
  ZoomIn, ZoomOut, ChevronLeft, ChevronRight, Paintbrush,
  PenTool, Eraser, Crosshair, Sparkles, Wand2,
} from "lucide-react";

interface Props {
  state: SegmentationState;
  onClearMask: () => void;
  onSave: () => void;
  onImportImage: () => void;
  onUndo: () => void;
  onRedo: () => void;
}

const ToolSidebar = ({ state, onClearMask, onSave, onImportImage, onUndo, onRedo }: Props) => {
  const s = state;

  const toolBtn = (mode: ToolMode, icon: React.ReactNode, label: string) => (
    <Button
      size="sm"
      variant={s.toolMode === mode ? "default" : "outline"}
      className="flex-1 gap-1.5 text-xs font-mono"
      onClick={() => s.setToolMode(mode)}
    >
      {icon} {label}
    </Button>
  );

  return (
    <aside className="w-[260px] min-w-[260px] h-full overflow-y-auto border-r border-border bg-card flex flex-col gap-1 p-3">
      {/* Info */}
      <div className="bg-secondary/50 rounded-md p-2 mb-1">
        <p className="font-mono text-[10px] text-muted-foreground truncate">
          {s.filename}
        </p>
        <p className="font-mono text-xs">
          Frame: <span className="text-primary font-bold">{s.currentFrame + 1}</span>
          {s.totalFrames > 0 && <span className="text-muted-foreground"> / {s.totalFrames}</span>}
        </p>
        <p className="font-mono text-[10px] text-muted-foreground">
          Mode: <span className="text-foreground">{s.toolMode.toUpperCase()}</span>
          {s.autoOutline && " + AUTO OUTLINE"}
        </p>
      </div>

      {/* Class Selector */}
      <div>
        <Label className="font-mono text-[10px] text-muted-foreground uppercase tracking-wider">Class</Label>
        <Select value={String(s.activeClass)} onValueChange={(v) => s.setActiveClass(Number(v))}>
          <SelectTrigger className="h-8 font-mono text-xs mt-1">
            <SelectValue />
          </SelectTrigger>
          <SelectContent className="dark bg-popover">
            {SEG_CLASSES.map((c, i) => (
              <SelectItem key={c.key} value={String(i)} className="font-mono text-xs">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: c.color }} />
                  {c.name}
                </div>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Auto Tools */}
      <div className="mt-2">
        <Label className="font-mono text-[10px] text-muted-foreground uppercase tracking-wider">Auto Tools</Label>
        <div className="flex gap-2 mt-1">
          <Button
            size="sm"
            variant={s.toolMode === "box" ? "default" : "outline"}
            className="flex-1 gap-1.5 text-xs font-mono"
            onClick={() => s.setToolMode("box")}
          >
            <Crosshair className="w-3.5 h-3.5" /> Box Prompt
          </Button>
          <Button
            size="sm"
            variant={s.autoOutline ? "default" : "outline"}
            className="flex-1 gap-1.5 text-xs font-mono"
            onClick={() => s.setAutoOutline(!s.autoOutline)}
          >
            <Sparkles className="w-3.5 h-3.5" /> Auto Outline
          </Button>
        </div>
      </div>

      {/* Manual Tools */}
      <div className="mt-2">
        <Label className="font-mono text-[10px] text-muted-foreground uppercase tracking-wider">Manual Tools</Label>
        <div className="grid grid-cols-3 gap-1.5 mt-1">
          {toolBtn("paint", <Paintbrush className="w-3.5 h-3.5" />, "Paint")}
          {toolBtn("outline", <PenTool className="w-3.5 h-3.5" />, "Outline")}
          {toolBtn("eraser", <Eraser className="w-3.5 h-3.5" />, "Eraser")}
        </div>
      </div>

      {/* Brush Size */}
      {(s.toolMode === "paint" || s.toolMode === "eraser") && (
        <div className="mt-2">
          <Label className="font-mono text-[10px] text-muted-foreground uppercase tracking-wider">
            Brush Size: {s.brushSize}
          </Label>
          <Slider
            value={[s.brushSize]}
            onValueChange={([v]) => s.setBrushSize(v)}
            min={1}
            max={50}
            step={1}
            className="mt-1"
          />
        </div>
      )}

      {/* Undo / Redo */}
      <div className="flex gap-2 mt-3">
        <Button size="sm" variant="outline" className="flex-1 gap-1 text-xs font-mono" onClick={onUndo}>
          <Undo2 className="w-3.5 h-3.5" /> Undo ({s.undoCount})
          <span className="text-[8px] text-muted-foreground ml-auto">⌘Z</span>
        </Button>
        <Button size="sm" variant="outline" className="flex-1 gap-1 text-xs font-mono" onClick={onRedo}>
          <Redo2 className="w-3.5 h-3.5" /> Redo ({s.redoCount})
          <span className="text-[8px] text-muted-foreground ml-auto">⌘Y</span>
        </Button>
      </div>

      {/* Overlay */}
      <div className="flex items-center gap-2 mt-3">
        <Button size="sm" variant="outline" className="flex-1 gap-1.5 text-xs font-mono" onClick={s.toggleOverlay}>
          {s.showOverlay ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
          Overlay ({s.showOverlay ? "ON" : "OFF"})
        </Button>
      </div>
      <div className="flex items-center gap-2 mt-1">
        <Checkbox
          id="labels"
          checked={s.showLabels}
          onCheckedChange={(v) => s.setShowLabels(!!v)}
        />
        <label htmlFor="labels" className="font-mono text-xs cursor-pointer">Show Class Labels</label>
      </div>

      {/* Clear Mask */}
      <Button size="sm" variant="destructive" className="mt-3 gap-1.5 text-xs font-mono" onClick={onClearMask}>
        <Trash2 className="w-3.5 h-3.5" /> Clear Mask
      </Button>

      {/* Save */}
      <Button
        size="sm"
        className="mt-2 gap-1.5 text-xs font-mono bg-green-600 hover:bg-green-700 text-white"
        onClick={onSave}
      >
        <Save className="w-3.5 h-3.5" /> 💾 SAVE (Ctrl+S)
      </Button>

      {/* Navigation */}
      <div className="mt-3">
        <Label className="font-mono text-[10px] text-muted-foreground uppercase tracking-wider">Navigation</Label>
        <div className="flex gap-1 mt-1">
          <Input
            type="number"
            min={1}
            max={s.totalFrames || 1}
            value={s.currentFrame + 1}
            onChange={(e) => s.setCurrentFrame(Math.max(0, Number(e.target.value) - 1))}
            className="h-7 w-14 text-xs font-mono text-center"
          />
          <Button size="sm" variant="outline" className="h-7 px-2 text-[10px] font-mono"
            onClick={() => s.setCurrentFrame(Math.max(0, s.currentFrame - 50))}>-50</Button>
          <Button size="sm" variant="outline" className="h-7 px-2 text-[10px] font-mono"
            onClick={() => s.setCurrentFrame(Math.max(0, s.currentFrame - 10))}>-10</Button>
          <Button size="sm" variant="outline" className="h-7 px-1.5" onClick={() => s.setCurrentFrame(Math.max(0, s.currentFrame - 1))}>
            <ChevronLeft className="w-3.5 h-3.5" />
          </Button>
          <Button size="sm" variant="outline" className="h-7 px-1.5" onClick={() => s.setCurrentFrame(s.currentFrame + 1)}>
            <ChevronRight className="w-3.5 h-3.5" />
          </Button>
          <Button size="sm" variant="outline" className="h-7 px-2 text-[10px] font-mono"
            onClick={() => s.setCurrentFrame(s.currentFrame + 10)}>+10</Button>
          <Button size="sm" variant="outline" className="h-7 px-2 text-[10px] font-mono"
            onClick={() => s.setCurrentFrame(s.currentFrame + 50)}>+50</Button>
        </div>
      </div>

      {/* Image Tools */}
      <div className="mt-3">
        <Label className="font-mono text-[10px] text-muted-foreground uppercase tracking-wider">Image Tools</Label>
        <div className="flex gap-1.5 mt-1">
          <Button size="sm" variant="outline" className="flex-1 gap-1 text-xs font-mono" onClick={onImportImage}>
            <Upload className="w-3.5 h-3.5" /> Import
          </Button>
          <Button size="sm" variant="outline" className="h-8 px-2" onClick={() => s.setZoom(Math.min(5, s.zoom + 0.25))}>
            <ZoomIn className="w-3.5 h-3.5" />
          </Button>
          <Button size="sm" variant="outline" className="h-8 px-2" onClick={() => s.setZoom(Math.max(0.25, s.zoom - 0.25))}>
            <ZoomOut className="w-3.5 h-3.5" />
          </Button>
        </div>
        <Button size="sm" variant="outline" className="w-full mt-1.5 gap-1.5 text-xs font-mono">
          <Wand2 className="w-3.5 h-3.5" /> Optimize for Tool
        </Button>
        <p className="font-mono text-[10px] text-muted-foreground mt-1">Zoom: {Math.round(s.zoom * 100)}%</p>
      </div>

      {/* Shortcuts legend */}
      <div className="mt-auto pt-4 border-t border-border">
        <p className="font-mono text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Shortcuts</p>
        {[
          ["←/→", "Nav frames"],
          ["Space", "Toggle overlay"],
          ["Ctrl+Z", "Undo"],
          ["Ctrl+Y", "Redo"],
          ["Ctrl+S", "Save"],
        ].map(([k, v]) => (
          <div key={k} className="flex justify-between font-mono text-[10px] text-muted-foreground">
            <span className="text-foreground">{k}</span>
            <span>{v}</span>
          </div>
        ))}
      </div>
    </aside>
  );
};

export default ToolSidebar;
