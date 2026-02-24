"""
Image processing utilities for the segmentation server.
"""
import cv2
import numpy as np
from skimage import io as skio


# Class color map (matching gui_main.py and frontend)
CLASSES = {
    "Uterus":  {"color": (0, 255, 255)},    # Cyan
    "Tools":   {"color": (255, 0, 0)},       # Red
    "Ureter":  {"color": (255, 255, 0)},     # Yellow
    "Ovary":   {"color": (0, 255, 0)},       # Green
    "Vessel":  {"color": (255, 0, 255)},     # Magenta
}


def load_image(path: str) -> np.ndarray:
    """Load image as RGB numpy array (H, W, 3)."""
    img = skio.imread(path)
    if len(img.shape) == 2:
        img = np.repeat(img[:, :, None], 3, axis=-1)
    else:
        img = img[:, :, :3]
    return img


def extract_polygon(binary_mask: np.ndarray) -> list[dict]:
    """
    Extract the largest contour polygon from a binary mask.
    Returns list of {x, y} points.
    """
    mask_u8 = (binary_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    # Take the largest contour
    largest = max(contours, key=cv2.contourArea)

    # Simplify polygon
    epsilon = 0.005 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)

    return [{"x": float(pt[0][0]), "y": float(pt[0][1])} for pt in approx]


def save_mask_png(mask_array: np.ndarray, path: str) -> None:
    """Save a colored mask (H, W, 3) as PNG."""
    # Convert RGB to BGR for OpenCV
    bgr = cv2.cvtColor(mask_array.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


def get_class_color(class_name: str) -> tuple[int, int, int]:
    """Get RGB color for a segmentation class."""
    cls = CLASSES.get(class_name)
    if cls is None:
        return (255, 255, 255)
    return cls["color"]
