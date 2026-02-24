# -*- coding: utf-8 -*-
import sys
import os
import glob
import time
import shutil
import threading
from PyQt5.QtGui import (
    QBrush, QPainter, QPen, QPixmap, QKeySequence, QColor, QImage
)
from PyQt5.QtWidgets import (
    QFileDialog, QApplication, QGraphicsScene, QGraphicsView,
    QHBoxLayout, QPushButton, QVBoxLayout, QWidget, QShortcut, QLabel,
    QComboBox, QSpinBox, QGroupBox, QGridLayout, QMessageBox,
    QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

import numpy as np
import cv2
from skimage import transform, io
from PIL import Image, ImageDraw, ImageFont

# Guarded import for torch so we can show a helpful message for Windows DLL errors
try:
    import torch
    import torch.nn.functional as F
except Exception as _torch_err:  # capture ImportError and OSError (DLL load failures)
    err_text = repr(_torch_err)
    print("\nERROR: failed to import 'torch' — the GUI requires PyTorch to run.")
    print("Exception:", err_text)
    print("\nQuick fixes:")
    print(" - Make sure you installed a PyTorch build that matches your OS/CUDA version. Visit: https://pytorch.org/get-started/locally/")
    print(" - For Windows 'WinError 1114' or DLL load failures, install the Microsoft Visual C++ Redistributable (2015-2022) and reboot:")
    print("     https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist")
    print(" - To try a CPU-only wheel (pip):")
    print("     pip uninstall -y torch torchvision && pip cache purge && pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision")
    print(" - If you used conda, prefer: conda install -c pytorch pytorch torchvision")
    print("\nAfter fixing, reactivate your venv and run `python gui_main.py` again.")
    # Provide extra hint for the common c10.dll issue
    if "c10.dll" in err_text or "WinError 1114" in err_text:
        print("\nDetected c10.dll/WinError 1114 — commonly caused by incompatible GPU/CUDA drivers or missing MSVC runtimes.")
    sys.exit(1)
from segment_anything import sam_model_registry

# --- CONFIG ---
BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(BASE_DIR, "dataset")
FRAMES_DIR = os.path.join(ROOT_DIR, "video_frames")
DATA_DIR = os.path.join(ROOT_DIR, "Data")
MASKS_DIR = os.path.join(ROOT_DIR, "Masks")
OVERLAYS_DIR = os.path.join(ROOT_DIR, "Overlays")
ANNOTATED_DIR = os.path.join(ROOT_DIR, "annotated")
OTHER_IMAGES_DIR = os.path.join(ROOT_DIR, "OtherImages")
os.makedirs(OTHER_IMAGES_DIR, exist_ok=True)
SAM_CKPT_PATH = os.path.join(BASE_DIR, "work_dir", "sam_model", "sam_vit_b.pth")
EMBEDDINGS_DIR = os.path.join(ROOT_DIR, "embeddings")
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)
MODEL_LOCK = threading.Lock()

CLASSES = {
    "Uterus":   {'color': (0, 255, 255)},   # Cyan
    "Tools":    {'color': (255, 0, 0)},     # Red
    "Ureter":   {'color': (255, 255, 0)},   # Yellow
    "Ovary":    {'color': (0, 255, 0)},     # Green
    "Vessel":   {'color': (255, 0, 255)},   # Magenta
}

# Maximum number of undo steps to keep in memory
MAX_UNDO_HISTORY = 20

for cls in CLASSES:
    os.makedirs(os.path.join(DATA_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(MASKS_DIR, f"{cls}_Masks"), exist_ok=True)
    os.makedirs(os.path.join(OVERLAYS_DIR, f"{cls}_Overlays"), exist_ok=True)
    os.makedirs(os.path.join(ANNOTATED_DIR, f"{cls}_Annotated"), exist_ok=True)

torch.manual_seed(2023)
torch.cuda.empty_cache()
torch.cuda.manual_seed(2023)
np.random.seed(2023)

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def sam_inference(sam_model, img_embed, box_1024, height, width):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]

    with MODEL_LOCK:
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=None, boxes=box_torch, masks=None,
        )
        low_res_logits, _ = sam_model.mask_decoder(
            image_embeddings=img_embed,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(
        low_res_pred, size=(height, width), mode="bilinear", align_corners=False,
    )
    low_res_pred = low_res_pred.squeeze().cpu().numpy()
    sam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return sam_seg

print("Loading SAM model...")
sam_model = sam_model_registry["vit_b"](checkpoint=SAM_CKPT_PATH).to(device)
sam_model.eval()
print("Done.")

def np2pixmap(np_img):
    height, width, channel = np_img.shape
    bytesPerLine = 3 * width
    # Keep a reference to prevent garbage collection
    np_img = np.ascontiguousarray(np_img)
    qImg = QImage(np_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
    return QPixmap.fromImage(qImg.copy())  # .copy() ensures data persists

class EmbeddingThread(QThread):
    def __init__(self, sam_model, image_files):
        super().__init__()
        self.sam_model = sam_model
        self.image_files = image_files
        self.queue = []
        self.active = True
        self.mutex = threading.Lock()

    def set_queue(self, idxs):
        with self.mutex:
            self.queue = list(idxs)
            
    def run(self):
        while self.active:
            target_idx = -1
            with self.mutex:
                if self.queue:
                    target_idx = self.queue.pop(0)
            
            if target_idx == -1:
                self.msleep(100)
                continue
                
            if target_idx < 0 or target_idx >= len(self.image_files):
                continue

            img_path = self.image_files[target_idx]
            filename = os.path.basename(img_path)
            emb_path = os.path.join(EMBEDDINGS_DIR, filename + ".pt")
            
            if os.path.exists(emb_path):
                continue
                
            try:
                # Load image
                img_np = io.imread(img_path)
                if len(img_np.shape) == 2:
                    img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
                else:
                    img_3c = img_np[:, :, :3]
                
                # Resize (cv2 is faster)
                img_1024 = cv2.resize(img_3c, (1024, 1024), interpolation=cv2.INTER_CUBIC)
                
                # Normalize
                img_1024 = (img_1024 - img_1024.min()) / np.clip(
                    img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
                )
                img_1024_tensor = (
                    torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
                )
                
                with MODEL_LOCK:
                    with torch.no_grad():
                        embedding = self.sam_model.image_encoder(img_1024_tensor)
                
                # Save (CPU)
                torch.save(embedding.cpu(), emb_path)
                
            except Exception as e:
                print(f"Error embedding {filename}: {e}")

    def stop(self):
        self.active = False
        self.wait()

class Window(QWidget):
    def __init__(self):
        super().__init__()
        
        self.image_files = []
        self.current_idx = 0
        self.image_path = None
        self.img_3c = None
        self.mask_c = None
        self.bg_img = None
        self.embedding = None
        self.show_overlay = True
        self.show_labels = True
        self.current_class = "Uterus"
        self.embedding_thread = None
        
        # Painting State
        self.paint_active = False
        self.eraser_active = False
        self.last_paint_pos = None

        # Manual Outline State
        self.outline_active = False # Manual Fill
        self.auto_outline_active = False # AI Inference within Outline
        self.outline_points = []


        # Outline State
        self.outline_active = False
        self.outline_points = []

        
        # Undo history stack - stores copies of mask states
        self.mask_history = []
        self.redo_history = []

        main_layout = QHBoxLayout(self)
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(250)
        
        self.lbl_info = QLabel("Loading...")
        self.lbl_info.setWordWrap(True)
        control_layout.addWidget(self.lbl_info)

        grp_class = QGroupBox("Select Class")
        v_class = QVBoxLayout()
        self.combo_class = QComboBox()
        self.combo_class.addItems(CLASSES.keys())
        self.combo_class.currentTextChanged.connect(self.change_class)
        v_class.addWidget(self.combo_class)
        grp_class.setLayout(v_class)
        control_layout.addWidget(grp_class)



        # --- Auto Tools ---
        grp_auto = QGroupBox("Auto Tools")
        v_auto = QVBoxLayout()
        self.btn_auto_outline = QPushButton("Auto Outline")
        self.btn_auto_outline.setCheckable(True)
        self.btn_auto_outline.toggled.connect(self.toggle_auto_outline)
        v_auto.addWidget(QLabel("Default: Box Prompt"))
        v_auto.addWidget(self.btn_auto_outline)
        grp_auto.setLayout(v_auto)
        control_layout.addWidget(grp_auto)

        # --- Manual Paint Tools ---
        grp_paint = QGroupBox("Manual Switch")
        v_paint = QVBoxLayout()
        
        h_paint_btns = QHBoxLayout()
        self.btn_paint_mode = QPushButton("Paint Mode")
        self.btn_paint_mode.setCheckable(True)
        self.btn_paint_mode.toggled.connect(self.toggle_paint_mode)
        h_paint_btns.addWidget(self.btn_paint_mode)

        self.btn_outline_mode = QPushButton("Outline Mode")
        self.btn_outline_mode.setCheckable(True)
        self.btn_outline_mode.toggled.connect(self.toggle_outline_mode)
        h_paint_btns.addWidget(self.btn_outline_mode)

        self.btn_eraser_mode = QPushButton("Eraser Mode")
        self.btn_eraser_mode.setCheckable(True)
        self.btn_eraser_mode.toggled.connect(self.toggle_eraser_mode)
        h_paint_btns.addWidget(self.btn_eraser_mode)
        
        h_brush = QHBoxLayout()
        h_brush.addWidget(QLabel("Brush Size:"))
        self.spin_brush = QSpinBox()
        self.spin_brush.setRange(1, 50)
        self.spin_brush.setValue(5)
        h_brush.addWidget(self.spin_brush)
        
        v_paint.addLayout(h_paint_btns)
        v_paint.addLayout(h_brush)
        grp_paint.setLayout(v_paint)
        control_layout.addWidget(grp_paint)
        # --------------------------

        # Undo Button
        self.btn_undo = QPushButton("↩ UNDO (Ctrl+Z)")
        self.btn_undo.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; padding: 8px;")
        self.btn_undo.clicked.connect(self.undo_mask)
        self.btn_undo.setEnabled(False)  # Disabled until there's history
        control_layout.addWidget(self.btn_undo)

        # Redo Button
        self.btn_redo = QPushButton("↪ REDO (Ctrl+Y)")
        self.btn_redo.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 8px;")
        self.btn_redo.clicked.connect(self.redo_mask)
        self.btn_redo.setEnabled(False)
        control_layout.addWidget(self.btn_redo)

        self.btn_toggle = QPushButton("Toggle Overlay (Space)")
        self.btn_toggle.clicked.connect(self.toggle_overlay)
        
        self.chk_labels = QCheckBox("Show Class Labels")
        self.chk_labels.setChecked(True)
        self.chk_labels.toggled.connect(self.toggle_labels)
        
        self.btn_clear = QPushButton("Clear Mask")
        self.btn_clear.clicked.connect(self.clear_mask)
        
        # Manual Save Button
        self.btn_save = QPushButton("💾 SAVE (Ctrl+S)")
        self.btn_save.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.btn_save.clicked.connect(self.save_data)
        
        control_layout.addWidget(self.btn_toggle)
        control_layout.addWidget(self.chk_labels)
        control_layout.addWidget(self.btn_clear)
        control_layout.addWidget(self.btn_save)

        grp_nav = QGroupBox("Navigation")
        grid_nav = QGridLayout()
        
        self.spin_jump = QSpinBox()
        self.spin_jump.setRange(1, 99999)
        self.btn_jump = QPushButton("Go")
        self.btn_jump.clicked.connect(self.jump_to_frame)
        
        grid_nav.addWidget(QLabel("Jump to:"), 0, 0)
        grid_nav.addWidget(self.spin_jump, 0, 1)
        grid_nav.addWidget(self.btn_jump, 0, 2)
        
        btn_prev = QPushButton("< Prev")
        btn_next = QPushButton("Next >")
        btn_skip10 = QPushButton("+10 >")
        btn_skip50 = QPushButton("+50 >")
        
        btn_prev.clicked.connect(self.prev_image)
        btn_next.clicked.connect(self.next_image)
        btn_skip10.clicked.connect(lambda: self.skip_image(10))
        btn_skip50.clicked.connect(lambda: self.skip_image(50))

        grid_nav.addWidget(btn_prev, 1, 0)
        grid_nav.addWidget(btn_next, 1, 2)
        grid_nav.addWidget(btn_skip10, 2, 0)
        grid_nav.addWidget(btn_skip50, 2, 2)
        
        grp_nav.setLayout(grid_nav)
        control_layout.addWidget(grp_nav)

        # --- AnyImage Tools ---
        grp_any = QGroupBox("AnyImage Tools")
        grp_any.setStyleSheet("QGroupBox { font-weight: bold; color: #9C27B0; border: 1px solid #9C27B0; margin-top: 10px; }")
        v_any = QVBoxLayout()
        
        self.btn_any_select = QPushButton("Import Image")
        self.btn_any_select.clicked.connect(self.select_any_image)
        
        h_zoom = QHBoxLayout()
        btn_zoom_in = QPushButton("Zoom (+)")
        btn_zoom_in.clicked.connect(lambda: self.view.scale(1.2, 1.2))
        btn_zoom_out = QPushButton("Zoom (-)")
        btn_zoom_out.clicked.connect(lambda: self.view.scale(0.8, 0.8))
        h_zoom.addWidget(btn_zoom_in)
        h_zoom.addWidget(btn_zoom_out)

        self.btn_optimize = QPushButton("Optimise for Tool")
        self.btn_optimize.setToolTip("Resize to match standard dataset resolution")
        self.btn_optimize.clicked.connect(self.optimize_current_image)
        
        v_any.addWidget(self.btn_any_select)
        v_any.addLayout(h_zoom)
        v_any.addWidget(self.btn_optimize)
        grp_any.setLayout(v_any)
        
        control_layout.addWidget(grp_any)
        control_layout.addStretch()

        self.view = QGraphicsView()
        self.view.setRenderHint(QPainter.Antialiasing)
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)

        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.view)

        self.resize(1400, 900)
        self.setWindowTitle("AnatoTrace Multi-Class Annotator")

        QShortcut(QKeySequence(Qt.Key_Left), self).activated.connect(self.prev_image)
        QShortcut(QKeySequence(Qt.Key_Right), self).activated.connect(self.next_image)
        QShortcut(QKeySequence(Qt.Key_Space), self).activated.connect(self.toggle_overlay)
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self.save_data)
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self.undo_mask)  # Undo shortcut
        QShortcut(QKeySequence("Ctrl+Y"), self).activated.connect(self.redo_mask)  # Redo shortcut

        self.is_mouse_down = False
        self.start_pos = (None, None)
        self.rect_item = None
        self.scene.mousePressEvent = self.mouse_press
        self.scene.mouseMoveEvent = self.mouse_move
        self.scene.mouseReleaseEvent = self.mouse_release

        self.load_directory(FRAMES_DIR)

    def load_directory(self, dir_path):
        if not os.path.exists(dir_path):
            self.lbl_info.setText(f"Dir not found: {dir_path}")
            return
            
        self.image_files = sorted(glob.glob(os.path.join(dir_path, "*.jpg")))
        if not self.image_files:
            self.lbl_info.setText("No .jpg images found!")
            return

        self.current_idx = 0
        
        # Restart worker thread with new files
        if self.embedding_thread is not None:
            self.embedding_thread.stop()
            
        self.embedding_thread = EmbeddingThread(sam_model, self.image_files)
        self.embedding_thread.start()
        
        self.load_image()

    def select_any_image(self):
        opts = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.jpg *.jpeg *.png *.bmp *.tiff);;All Files (*)", options=opts
        )
        if not file_path:
            return

        # 1. Ensure target dir
        if not os.path.exists(OTHER_IMAGES_DIR):
            os.makedirs(OTHER_IMAGES_DIR, exist_ok=True)
            
        # 2. Convert/Copy to JPG
        try:
            filename = os.path.basename(file_path)
            name, ext = os.path.splitext(filename)
            target_filename = name + ".jpg"
            target_path = os.path.join(OTHER_IMAGES_DIR, target_filename)
            
            # Use cv2 to read/write to ensure consistent JPG format
            # cv2.imread handles various formats
            img_in = cv2.imread(file_path)
            if img_in is None:
                # Try via io.imread if cv2 fails (e.g. some tiffs or paths)
                img_in_rgb = io.imread(file_path)
                if len(img_in_rgb.shape) == 3:
                     # RGB to BGR for cv2 save
                     img_in = cv2.cvtColor(img_in_rgb, cv2.COLOR_RGB2BGR)
                else:
                     img_in = img_in_rgb # Gray?

            if img_in is None:
                raise ValueError("Could not read image content")

            cv2.imwrite(target_path, img_in)
            print(f"Imported {filename} to {target_path}")

        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import image:\n{e}")
            return

        # 3. Load the folder
        self.load_directory(OTHER_IMAGES_DIR)
        
        # 4. Jump to the new image
        target_abs = os.path.abspath(target_path)
        for idx, path in enumerate(self.image_files):
            if os.path.abspath(path) == target_abs:
                self.current_idx = idx
                break
        
        self.load_image()

    def optimize_current_image(self):
        if not self.image_path or not os.path.exists(self.image_path):
            return
            
        # Determine target size from FRAMES_DIR or default to 1024x1024
        target_h, target_w = 1024, 1024
        ref_files = glob.glob(os.path.join(FRAMES_DIR, "*.jpg"))
        if ref_files:
            try:
                ref_img = cv2.imread(ref_files[0])
                if ref_img is not None:
                    target_h, target_w = ref_img.shape[:2]
            except:
                pass
                
        # Confirm action
        reply = QMessageBox.question(self, "Optimize Image", 
                                     f"Resize image to {target_w}x{target_h} to match dataset standards?\nThis will overwrite the file.",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.No:
            return
            
        try:
            img = cv2.imread(self.image_path)
            if img is None: return
            
            img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(self.image_path, img_resized)
            
            # Remove existing embedding cache as image changed
            filename = os.path.basename(self.image_path)
            emb_path = os.path.join(EMBEDDINGS_DIR, filename + ".pt")
            if os.path.exists(emb_path):
                os.remove(emb_path)
                self.embedding = None
            
            # Reload
            self.load_image()
            self.lbl_info.setText(f"Image Optimized ({target_w}x{target_h})")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def change_class(self, text):
        self.current_class = text

    def load_image(self):
        self.image_path = self.image_files[self.current_idx]
        self.embedding = None
        self.rect_item = None
        
        # 1. Try to load embedding from cache
        filename = os.path.basename(self.image_path)
        emb_path = os.path.join(EMBEDDINGS_DIR, filename + ".pt")
        if os.path.exists(emb_path):
            try:
                self.embedding = torch.load(emb_path, map_location=device)
            except Exception as e:
                print(f"Failed to load embedding: {e}")

        # 2. Update Background Worker Queue
        # Priorities: Current (if missing) -> Next 5 -> Prev 1
        queue = []
        if self.embedding is None:
            queue.append(self.current_idx)
            
        for i in range(1, 6):
            if self.current_idx + i < len(self.image_files):
                queue.append(self.current_idx + i)
                
        if self.embedding_thread:
            self.embedding_thread.set_queue(queue)
        
        # Clear undo history when loading new image
        self.mask_history = []
        self.redo_history = []
        self.update_history_buttons()
        
        self.lbl_info.setText(f"Frame: {self.current_idx + 1} / {len(self.image_files)}\nFile: {filename}")
        self.spin_jump.setValue(self.current_idx + 1)
        
        img_np = io.imread(self.image_path)
        if len(img_np.shape) == 2:
            self.img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        else:
            self.img_3c = img_np[:, :, :3]

        H, W, _ = self.img_3c.shape
        self.mask_c = np.zeros((H, W, 3), dtype="uint8")
        
        self.scene.clear()
        self.scene.setSceneRect(0, 0, W, H)
        self.update_display()

    def ensure_embedding(self):
        if self.embedding is None:
            # Check cache again (maybe worker finished)
            filename = os.path.basename(self.image_path)
            emb_path = os.path.join(EMBEDDINGS_DIR, filename + ".pt")
            if os.path.exists(emb_path):
                try:
                    self.embedding = torch.load(emb_path, map_location=device)
                    self.lbl_info.setText(f"Ready (Cached): {self.current_class}")
                    return
                except:
                    pass

            self.lbl_info.setText("Embedding... (Please Wait)")
            QApplication.processEvents()
            
            # Compute synchronously if not ready
            # Use cv2 for speed
            img_1024 = cv2.resize(self.img_3c, (1024, 1024), interpolation=cv2.INTER_CUBIC)
            
            img_1024 = (img_1024 - img_1024.min()) / np.clip(
                img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
            )
            img_1024_tensor = (
                torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
            )
            
            with MODEL_LOCK:
                with torch.no_grad():
                    self.embedding = sam_model.image_encoder(img_1024_tensor)
            
            # Save to disk
            try:
                torch.save(self.embedding.cpu(), emb_path)
            except Exception as e:
                print(f"Failed to save embedding: {e}")
                
            self.lbl_info.setText(f"Ready: {self.current_class}")

    def toggle_paint_mode(self, checked):
        self.paint_active = checked
        if checked:
            self.lbl_info.setText(f"Paint Mode ON\nClass: {self.current_class}")
            # Disable others
            if self.outline_active: self.btn_outline_mode.setChecked(False)
            if self.auto_outline_active: self.btn_auto_outline.setChecked(False)
            if self.eraser_active: self.btn_eraser_mode.setChecked(False)
            
            # Clear any existing selection rect
            if self.rect_item:
                if self.rect_item.scene() == self.scene:
                    self.scene.removeItem(self.rect_item)
                self.rect_item = None
        else:
            if not self.outline_active and not self.auto_outline_active and not self.eraser_active:
                self.lbl_info.setText("Paint Mode OFF\nBox Prompt Mode")

    def toggle_outline_mode(self, checked):
        self.outline_active = checked
        if checked:
            self.lbl_info.setText(f"Manual Outline ON\nClass: {self.current_class}")
            # Disable others
            if self.paint_active: self.btn_paint_mode.setChecked(False)
            if self.auto_outline_active: self.btn_auto_outline.setChecked(False)
            if self.eraser_active: self.btn_eraser_mode.setChecked(False)

            # Clear any existing selection rect
            if self.rect_item:
                if self.rect_item.scene() == self.scene:
                    self.scene.removeItem(self.rect_item)
                self.rect_item = None
        else:
            if not self.paint_active and not self.auto_outline_active and not self.eraser_active:
                self.lbl_info.setText("Manual Outline OFF\nBox Prompt Mode")

    def toggle_auto_outline(self, checked):
        self.auto_outline_active = checked
        if checked:
            self.lbl_info.setText(f"Auto Outline ON\nClass: {self.current_class}")
            # Disable others
            if self.paint_active: self.btn_paint_mode.setChecked(False)
            if self.outline_active: self.btn_outline_mode.setChecked(False)
            if self.eraser_active: self.btn_eraser_mode.setChecked(False)

            # Clear any existing selection rect
            if self.rect_item:
                if self.rect_item.scene() == self.scene:
                    self.scene.removeItem(self.rect_item)
                self.rect_item = None
        else:
            if not self.paint_active and not self.outline_active and not self.eraser_active:
                self.lbl_info.setText("Auto Outline OFF\nBox Prompt Mode")

    def toggle_eraser_mode(self, checked):
        self.eraser_active = checked
        if checked:
            self.lbl_info.setText(f"Eraser Mode ON")
            # Disable others
            if self.paint_active: self.btn_paint_mode.setChecked(False)
            if self.outline_active: self.btn_outline_mode.setChecked(False)
            if self.auto_outline_active: self.btn_auto_outline.setChecked(False)
            
            # Clear any existing selection rect
            if self.rect_item:
                if self.rect_item.scene() == self.scene:
                    self.scene.removeItem(self.rect_item)
                self.rect_item = None
        else:
            if not self.paint_active and not self.outline_active and not self.auto_outline_active:
                self.lbl_info.setText("Eraser Mode OFF\nBox Prompt Mode")

    def generate_overlay(self, force_labels=None):
        if self.img_3c is None: return None

        bg = Image.fromarray(self.img_3c.astype("uint8"), "RGB")
        
        show_ov = self.show_overlay
        show_lbl = self.show_labels
        
        if force_labels is not None:
             show_lbl = force_labels
             if force_labels: # If forcing labels, ensure overlay is visible too
                 show_ov = True
                 
        if show_ov and self.mask_c.max() > 0:
            mask = Image.fromarray(self.mask_c.astype("uint8"), "RGB")
            img = Image.blend(bg, mask, 0.4)
            
            # Add labels if enabled
            if show_lbl:
                draw = ImageDraw.Draw(img)
                # Try to load a font, or use default
                try:
                    # Use a larger font if available, or default
                    font = ImageFont.truetype("arial.ttf", 20)
                except IOError:
                    font = None # Use default

                for cls_name, cls_data in CLASSES.items():
                    color = cls_data['color']
                    # Create binary mask for this color
                    lower = np.array(color, dtype="uint8")
                    upper = np.array(color, dtype="uint8")
                    cls_mask = cv2.inRange(self.mask_c, lower, upper)
                    
                    cnts, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for c in cnts:
                        if cv2.contourArea(c) > 100: # Filter small noise
                            M = cv2.moments(c)
                            if M["m00"] != 0:
                                cX = int(M["m10"] / M["m00"])
                                cY = int(M["m01"] / M["m00"])
                                
                                text = cls_name
                                
                                # Draw text with outline/shadow for visibility
                                if font:
                                    if hasattr(draw, "textbbox"):
                                        bbox = draw.textbbox((0, 0), text, font=font)
                                        text_w = bbox[2] - bbox[0]
                                        text_h = bbox[3] - bbox[1]
                                    else:
                                        text_w, text_h = draw.textsize(text, font=font)
                                        
                                    pos_x = cX - text_w // 2
                                    pos_y = cY - text_h // 2
                                    draw.text((pos_x+1, pos_y+1), text, font=font, fill="black")
                                    draw.text((pos_x, pos_y), text, font=font, fill="white")
                                else:
                                    # Default font fallback
                                    draw.text((cX+1, cY+1), text, fill="black")
                                    draw.text((cX, cY), text, fill="white")
        else:
            img = bg
            
        return img

    def update_display(self):
        img = self.generate_overlay()
        if img is None: return
        pixmap = np2pixmap(np.array(img))
        self.scene.clear()
        self.scene.addPixmap(pixmap)
        return img 

    # --- UNDO FUNCTIONALITY ---
    def push_mask_state(self):
        """Save current mask state to history before making changes."""
        # Make a deep copy of the current mask
        self.mask_history.append(self.mask_c.copy())
        
        # Limit history size to prevent memory issues
        if len(self.mask_history) > MAX_UNDO_HISTORY:
            self.mask_history.pop(0)

        # Clear redo history because we branched off
        self.redo_history = []
        
        self.update_history_buttons()
    
    def undo_mask(self):
        """Restore the previous mask state."""
        if not self.mask_history:
            self.lbl_info.setText("Nothing to undo!")
            return
        
        # Push current state to redo stack
        self.redo_history.append(self.mask_c.copy())

        # Pop the last saved state and restore it
        self.mask_c = self.mask_history.pop()
        self.update_history_buttons()
        self.update_display()
        
        remaining = len(self.mask_history)
        self.lbl_info.setText(f"Undo! ({remaining} steps remaining)")

    def redo_mask(self):
        """Restore the state that was undone."""
        if not self.redo_history:
            self.lbl_info.setText("Nothing to redo!")
            return

        # Push current state to undo stack
        self.mask_history.append(self.mask_c.copy())

        # Pop from redo stack
        self.mask_c = self.redo_history.pop()
        self.update_history_buttons()
        self.update_display()
        
        self.lbl_info.setText(f"Redo! ({len(self.redo_history)} steps remaining)")
    
    def update_history_buttons(self):
        """Update undo/redo button states and text."""
        has_history = len(self.mask_history) > 0
        self.btn_undo.setEnabled(has_history)
        self.btn_undo.setText(f"↩ UNDO ({len(self.mask_history)}) Ctrl+Z")

        has_redo = len(self.redo_history) > 0
        self.btn_redo.setEnabled(has_redo)
        self.btn_redo.setText(f"↪ REDO ({len(self.redo_history)}) Ctrl+Y")

    def save_data(self):
        if self.mask_c.max() == 0:
            self.lbl_info.setText("No mask to save!")
            return
        
        filename = os.path.basename(self.image_path)
        cls_name = self.current_class
        mask_name = os.path.splitext(filename)[0] + ".png"
        
        # 1. Save Image
        save_img_path = os.path.join(DATA_DIR, cls_name, filename)
        if not os.path.exists(save_img_path):
            shutil.copy(self.image_path, save_img_path)
            
        # 2. Save Mask
        save_mask_path = os.path.join(MASKS_DIR, f"{cls_name}_Masks", mask_name)
        io.imsave(save_mask_path, self.mask_c, check_contrast=False)
        
        # 3. Save Overlay (Current View)
        overlay_img = self.update_display()
        save_overlay_path = os.path.join(OVERLAYS_DIR, f"{cls_name}_Overlays", mask_name)
        overlay_img.save(save_overlay_path)

        # 4. Save Annotated (Labeled Overlay - Always Labeled)
        annotated_img = self.generate_overlay(force_labels=True)
        save_annotated_path = os.path.join(ANNOTATED_DIR, f"{cls_name}_Annotated", mask_name)
        annotated_img.save(save_annotated_path)
        
        print(f"Saved {cls_name}: {mask_name}")
        self.lbl_info.setText(f"Saved: {cls_name} / {mask_name}")

    def prev_image(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_image()

    def next_image(self):
        if self.current_idx < len(self.image_files) - 1:
            self.current_idx += 1
            self.load_image()

    def skip_image(self, n):
        if self.current_idx + n < len(self.image_files):
            self.current_idx += n
            self.load_image()

    def jump_to_frame(self):
        val = self.spin_jump.value() - 1
        if 0 <= val < len(self.image_files):
            self.current_idx = val
            self.load_image()

    def toggle_overlay(self):
        self.show_overlay = not self.show_overlay
        self.update_display()

    def toggle_labels(self, checked):
        self.show_labels = checked
        self.update_display()

    def clear_mask(self):
        # Save state before clearing so we can undo
        if self.mask_c.max() > 0:
            self.push_mask_state()
        self.mask_c.fill(0)
        self.update_display()

    # --- Mouse Events ---
    def mouse_press(self, ev):
        if self.paint_active:
            # PAINT MODE
            if self.img_3c is None: return
            
            x, y = int(ev.scenePos().x()), int(ev.scenePos().y())
            H, W, _ = self.img_3c.shape
            
            # Bounds check
            if x < 0 or x >= W or y < 0 or y >= H: return

            self.push_mask_state() # Save state for Undo
            self.is_mouse_down = True
            
            color = CLASSES[self.current_class]['color']
            radius = self.spin_brush.value()
            
            # Draw initial circle
            cv2.circle(self.mask_c, (x, y), radius, color, -1)
            self.last_paint_pos = (x, y)
            self.update_display()

        elif self.eraser_active:
            # ERASER MODE
            if self.img_3c is None: return
            
            x, y = int(ev.scenePos().x()), int(ev.scenePos().y())
            H, W, _ = self.img_3c.shape
            
            # Bounds check
            if x < 0 or x >= W or y < 0 or y >= H: return

            self.push_mask_state() # Save state for Undo
            self.is_mouse_down = True
            
            color = (0, 0, 0) # Black/Transparent
            radius = self.spin_brush.value()
            
            # Draw initial circle
            cv2.circle(self.mask_c, (x, y), radius, color, -1)
            self.last_paint_pos = (x, y)
            self.update_display()
            
        elif self.outline_active or self.auto_outline_active:
            # OUTLINE MODE (Manual or Auto)
            if self.img_3c is None: return
            x, y = int(ev.scenePos().x()), int(ev.scenePos().y())
            
            # Note: We push state on release/apply to avoid phantom UNDO steps on invalid clicks
            
            self.is_mouse_down = True
            self.outline_points = [(x, y)]
            if self.auto_outline_active:
                self.ensure_embedding() # Prepare model
        
        else:
            # SAM BOX MODE
            self.ensure_embedding()
            self.is_mouse_down = True
            self.start_pos = ev.scenePos().x(), ev.scenePos().y()

    def mouse_move(self, ev):
        if not self.is_mouse_down: return
        x, y = ev.scenePos().x(), ev.scenePos().y()
        
        if self.paint_active:
            # DRAWING
            xi, yi = int(x), int(y)
            color = CLASSES[self.current_class]['color']
            radius = self.spin_brush.value()
            
            if self.last_paint_pos is not None:
                # Draw line to connect points for smooth stroke
                cv2.line(self.mask_c, self.last_paint_pos, (xi, yi), color, thickness=radius*2)
            
            # Draw circle at current to smooth edges
            cv2.circle(self.mask_c, (xi, yi), radius, color, -1)
            
            self.last_paint_pos = (xi, yi)
            self.update_display()

        elif self.eraser_active:
            # ERASING
            xi, yi = int(x), int(y)
            color = (0, 0, 0)
            radius = self.spin_brush.value()
            
            if self.last_paint_pos is not None:
                # Draw line to connect points for smooth stroke
                cv2.line(self.mask_c, self.last_paint_pos, (xi, yi), color, thickness=radius*2)
            
            # Draw circle at current to smooth edges
            cv2.circle(self.mask_c, (xi, yi), radius, color, -1)
            
            self.last_paint_pos = (xi, yi)
            self.update_display()

        elif self.outline_active or self.auto_outline_active:
            # OUTLINE DRAWING VISUALIZATION
            xi, yi = int(x), int(y)
            if self.outline_points:
                last_pt = self.outline_points[-1]
                # Draw visual line on scene (not on mask yet)
                # Draw visual line on scene using cosmetic pen (constant screen width)
                pen_color = CLASSES[self.current_class]['color']
                pen = QPen(QColor(*pen_color), 1)
                pen.setCosmetic(True)
                
                self.scene.addLine(last_pt[0], last_pt[1], xi, yi, pen)
                self.outline_points.append((xi, yi))
            
        else:
            # BOX SELECTION
            sx, sy = self.start_pos
            
            # Safe Removal
            # Safe Removal
            if self.rect_item:
                if self.rect_item.scene() == self.scene:
                    self.scene.removeItem(self.rect_item)
                self.rect_item = None
                
            # Use cosmetic pen for box selection too
            pen_box = QPen(QColor("red"), 1)
            pen_box.setCosmetic(True)
            
            self.rect_item = self.scene.addRect(
                min(sx, x), min(sy, y), abs(x - sx), abs(y - sy),
                pen=pen_box
            )

    def mouse_release(self, ev):
        self.is_mouse_down = False
        
        if self.paint_active or self.eraser_active:
            self.last_paint_pos = None
            self.update_display()
        elif self.outline_active:
            # FINISH MANUAL OUTLINE
            if len(self.outline_points) > 2:
                self.push_mask_state() # Save state before applying
                pts = np.array(self.outline_points, dtype=np.int32)
                color = CLASSES[self.current_class]['color']
                cv2.fillPoly(self.mask_c, [pts], color)
                self.update_display()
            self.outline_points = []
            
        elif self.auto_outline_active:
            # FINISH AUTO OUTLINE
            if len(self.outline_points) > 2:
                pts = np.array(self.outline_points, dtype=np.int32)
                
                # 1. Calc Bounding Box
                xs = pts[:, 0]
                ys = pts[:, 1]
                xmin, xmax = xs.min(), xs.max()
                ymin, ymax = ys.min(), ys.max()
                
                # Ignore tiny
                if (xmax - xmin) < 5 or (ymax - ymin) < 5:
                    self.outline_points = []
                    return

                self.push_mask_state()
                
                H, W, _ = self.img_3c.shape
                box_np = np.array([[xmin, ymin, xmax, ymax]])
                box_1024 = box_np / np.array([W, H, W, H]) * 1024
                
                # 2. Run Inference
                sam_mask = sam_inference(sam_model, self.embedding, box_1024, H, W)
                
                # 3. Mask with User Polygon (Strict "within it")
                poly_mask = np.zeros((H, W), dtype=np.uint8)
                cv2.fillPoly(poly_mask, [pts], 1)
                
                final_mask = sam_mask * poly_mask
                
                color = CLASSES[self.current_class]['color']
                self.mask_c[final_mask != 0] = color
                self.update_display()
                
            self.outline_points = []
            
        else:
            # SAM INFERENCE
            # Safe Removal
            if self.rect_item:
                if self.rect_item.scene() == self.scene:
                    self.scene.removeItem(self.rect_item)
                self.rect_item = None
            
            x, y = ev.scenePos().x(), ev.scenePos().y()
            sx, sy = self.start_pos
            xmin, xmax = min(x, sx), max(x, sx)
            ymin, ymax = min(y, sy), max(y, sy)
            
            # Ignore tiny clicks (<5px)
            if abs(xmax - xmin) < 5 or abs(ymax - ymin) < 5: return

            # Save current mask state BEFORE applying new segmentation
            self.push_mask_state()

            H, W, _ = self.img_3c.shape
            box_np = np.array([[xmin, ymin, xmax, ymax]])
            box_1024 = box_np / np.array([W, H, W, H]) * 1024

            sam_mask = sam_inference(sam_model, self.embedding, box_1024, H, W)
            
            color = CLASSES[self.current_class]['color']
            self.mask_c[sam_mask != 0] = color
            
            self.update_display()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    app.exec()