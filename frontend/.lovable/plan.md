

# MedSeg.io — Medical Image Segmentation Web App

## Page 1: Landing Page (inspired by screenshot)
- **Dark, cinematic hero section** with a large headline "PRECISION IN MEDICAL SEGMENTATION", subtitle about automated annotation for CTs/MRIs/Surgical Video, and an orange "Get Started" CTA button
- **Top navigation bar** with logo (MEDSEG.IO), links (Solutions, Vision, Docs), orange "Get Started" button, and "Log In" link
- **Background aesthetic**: dark gradient with subtle particle/bokeh effects and a wireframe medical illustration (lungs) using CSS/SVG
- Monospace/bold typography style matching the screenshot
- Optional additional sections: Features overview, How it works, Footer

## Page 2: Segmentation Tool (frontend replica of gui_main.py)

### Layout
- **Left sidebar** (control panel, ~250px) + **Main canvas area** (right side)
- Dark UI theme consistent with the landing page

### Left Sidebar — Control Panel
1. **Info Label** — Shows current frame number, filename, and active mode status
2. **Class Selector** — Dropdown with 5 classes: Uterus (Cyan), Tools (Red), Ureter (Yellow), Ovary (Green), Vessel (Magenta), each with their color indicator
3. **Auto Tools Group** — "Box Prompt" (default mode) label + toggleable "Auto Outline" button
4. **Manual Tools Group** — Toggle buttons for Paint Mode, Outline Mode, Eraser Mode (mutually exclusive with auto tools) + Brush Size slider (1–50)
5. **Undo / Redo buttons** — With step counters and keyboard shortcut labels (Ctrl+Z / Ctrl+Y)
6. **Toggle Overlay** button (Space) + "Show Class Labels" checkbox
7. **Clear Mask** button
8. **Save button** — Prominent green "💾 SAVE (Ctrl+S)" button
9. **Navigation Group** — Jump-to-frame input + Prev/Next/+10/+50 buttons
10. **AnyImage Tools Group** — Import Image button, Zoom In/Out buttons, Optimize for Tool button

### Main Canvas Area
- Large image display area where users can:
  - **Draw bounding boxes** (default SAM box prompt mode) by click-and-drag
  - **Paint** freehand masks with the selected class color
  - **Erase** parts of masks
  - **Draw outlines** (manual polygon fill or AI-assisted auto outline)
- Colored mask overlay blended on top of the image at ~40% opacity
- Class name labels displayed at the centroid of each segmented region
- Red rectangle visualization during box selection

### Keyboard Shortcuts
- Arrow Left/Right: navigate images
- Space: toggle overlay
- Ctrl+S: save
- Ctrl+Z: undo
- Ctrl+Y: redo

### Notes
- All AI inference (SAM model) will be handled by the user's backend — the frontend will make API calls to endpoints the user provides later
- Image upload, canvas drawing, mask overlay rendering, and UI state management will all be fully functional in the frontend
- The canvas interactions (painting, erasing, box drawing, polygon outline) will use HTML5 Canvas API

