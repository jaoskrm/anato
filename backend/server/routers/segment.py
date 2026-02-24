"""
Segmentation endpoints — box prompt and polygon prompt.
"""
import os
import asyncio
import numpy as np
import cv2
from fastapi import APIRouter, HTTPException

from ..config import settings
from ..models.schemas import BoxRequest, PolygonRequest, SegmentResponse, Point
from ..services.sam_service import sam_service
from ..services.embedding_cache import embedding_cache
from ..utils.image_utils import extract_polygon, get_class_color
from .images import get_image_store

router = APIRouter(prefix="/api/segment", tags=["segmentation"])


def _mask_path(image_id: str) -> str:
    return os.path.join(settings.MASK_DIR, f"{image_id}.png")


def _get_image_meta(image_id: str) -> dict:
    """Get image metadata or raise 404."""
    store = get_image_store()
    if image_id not in store:
        raise HTTPException(status_code=404, detail="Image not found. Upload first.")
    meta = store[image_id]
    if not meta.get("embedding_ready"):
        raise HTTPException(status_code=409, detail="Embedding not ready. Please wait.")
    return meta


def _load_existing_mask(image_id: str, height: int, width: int) -> np.ndarray:
    """Load existing mask or create blank one."""
    mask_path = _mask_path(image_id)
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path)
        if mask is not None:
            return cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    return np.zeros((height, width, 3), dtype=np.uint8)


def _apply_and_save_mask(
    image_id: str,
    binary_mask: np.ndarray,
    class_name: str,
    height: int,
    width: int,
) -> list[Point]:
    """Apply binary mask with class color to the stored colored mask, save, and return polygon."""
    color = get_class_color(class_name)
    colored_mask = _load_existing_mask(image_id, height, width)

    # Apply color where binary mask is nonzero
    colored_mask[binary_mask != 0] = color

    # Save mask
    mask_bgr = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)
    cv2.imwrite(_mask_path(image_id), mask_bgr)

    # Extract polygon
    polygon_points = extract_polygon(binary_mask)
    return [Point(x=p["x"], y=p["y"]) for p in polygon_points]


@router.post("/box", response_model=SegmentResponse)
async def segment_box(req: BoxRequest):
    """Run SAM segmentation with a bounding box prompt."""
    meta = _get_image_meta(req.image_id)
    width = meta["width"]
    height = meta["height"]

    # Get embedding
    embedding = embedding_cache.get(req.image_id)
    if embedding is None:
        raise HTTPException(status_code=500, detail="Embedding not found in cache")

    # Convert box from (x, y, w, h) to 1024-space (xmin, ymin, xmax, ymax)
    xmin = req.box.x
    ymin = req.box.y
    xmax = req.box.x + req.box.width
    ymax = req.box.y + req.box.height
    box_1024 = [
        xmin / width * 1024,
        ymin / height * 1024,
        xmax / width * 1024,
        ymax / height * 1024,
    ]

    # Run inference in thread pool (CPU-bound)
    binary_mask = await asyncio.to_thread(
        sam_service.segment_box, embedding, box_1024, height, width,
    )

    # Apply and save
    polygon = _apply_and_save_mask(req.image_id, binary_mask, req.class_name, height, width)

    color = get_class_color(req.class_name)
    return SegmentResponse(
        image_id=req.image_id,
        mask_url=f"/api/images/{req.image_id}/mask",
        polygon=polygon,
        class_name=req.class_name,
        color=list(color),
    )


@router.post("/polygon", response_model=SegmentResponse)
async def segment_polygon(req: PolygonRequest):
    """Run SAM segmentation with a polygon prompt."""
    meta = _get_image_meta(req.image_id)
    width = meta["width"]
    height = meta["height"]

    # Get embedding
    embedding = embedding_cache.get(req.image_id)
    if embedding is None:
        raise HTTPException(status_code=500, detail="Embedding not found in cache")

    # Convert points
    polygon_pts = [(p.x, p.y) for p in req.polygon]

    # Run inference in thread pool
    binary_mask = await asyncio.to_thread(
        sam_service.segment_polygon, embedding, polygon_pts, height, width,
    )

    # Apply and save
    polygon = _apply_and_save_mask(req.image_id, binary_mask, req.class_name, height, width)

    color = get_class_color(req.class_name)
    return SegmentResponse(
        image_id=req.image_id,
        mask_url=f"/api/images/{req.image_id}/mask",
        polygon=polygon,
        class_name=req.class_name,
        color=list(color),
    )
