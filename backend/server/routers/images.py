"""
Image upload, retrieval, and status endpoints.
"""
import os
import uuid
import asyncio
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from ..config import settings
from ..models.schemas import UploadResponse, EmbeddingStatusResponse
from ..services.embedding_cache import embedding_cache
from ..utils.image_utils import load_image

router = APIRouter(prefix="/api/images", tags=["images"])

# Map image_id → metadata
_image_store: dict[str, dict] = {}


def _image_path(image_id: str) -> str:
    return os.path.join(settings.UPLOAD_DIR, f"{image_id}.png")


def _mask_path(image_id: str) -> str:
    return os.path.join(settings.MASK_DIR, f"{image_id}.png")


@router.post("/upload", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """Upload an image and start embedding computation in the background."""
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Generate ID and save file
    image_id = str(uuid.uuid4())
    save_path = _image_path(image_id)

    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")

    with open(save_path, "wb") as f:
        f.write(content)

    # Get image dimensions
    img = load_image(save_path)
    height, width = img.shape[:2]

    # Store metadata
    _image_store[image_id] = {
        "path": save_path,
        "width": width,
        "height": height,
        "embedding_ready": False,
    }

    # Compute embedding in background
    embedding_status = "computing"
    if embedding_cache.has(image_id):
        _image_store[image_id]["embedding_ready"] = True
        embedding_status = "ready"
    else:
        # Start background task
        asyncio.get_event_loop().run_in_executor(
            None, _compute_embedding_sync, image_id, save_path,
        )

    return UploadResponse(
        image_id=image_id,
        width=width,
        height=height,
        embedding_status=embedding_status,
    )


def _compute_embedding_sync(image_id: str, image_path: str):
    """Synchronous wrapper for background embedding computation."""
    try:
        embedding_cache.compute_and_store(image_id, image_path)
        if image_id in _image_store:
            _image_store[image_id]["embedding_ready"] = True
        print(f"Embedding ready for {image_id}")
    except Exception as e:
        print(f"Error computing embedding for {image_id}: {e}")


@router.get("/{image_id}/status", response_model=EmbeddingStatusResponse)
async def get_embedding_status(image_id: str):
    """Check whether the image embedding is ready."""
    if image_id not in _image_store:
        raise HTTPException(status_code=404, detail="Image not found")

    ready = _image_store[image_id].get("embedding_ready", False)
    # Also check disk in case it was computed by another process
    if not ready and embedding_cache.has(image_id):
        _image_store[image_id]["embedding_ready"] = True
        ready = True

    return EmbeddingStatusResponse(
        image_id=image_id,
        embedding_ready=ready,
        message="Embedding computed successfully" if ready else "Embedding is being computed...",
    )


@router.get("/{image_id}")
async def get_image(image_id: str):
    """Serve the original uploaded image."""
    path = _image_path(image_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/png")


@router.get("/{image_id}/mask")
async def get_mask(image_id: str):
    """Serve the current mask for an image."""
    path = _mask_path(image_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No mask available for this image")
    return FileResponse(path, media_type="image/png")


def get_image_store():
    """Accessor for the image store (used by segment router)."""
    return _image_store
