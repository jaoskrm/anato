"""
Pydantic models for API request/response schemas.
"""
from pydantic import BaseModel


class Point(BaseModel):
    x: float
    y: float


class Box(BaseModel):
    x: float
    y: float
    width: float
    height: float


# --- Responses ---

class UploadResponse(BaseModel):
    image_id: str
    width: int
    height: int
    embedding_status: str  # "computing" | "ready"


class EmbeddingStatusResponse(BaseModel):
    image_id: str
    embedding_ready: bool
    message: str


class SegmentResponse(BaseModel):
    image_id: str
    mask_url: str
    polygon: list[Point]
    class_name: str
    color: list[int]  # [R, G, B]


class ErrorResponse(BaseModel):
    detail: str


# --- Requests ---

class BoxRequest(BaseModel):
    image_id: str
    box: Box
    class_name: str = "Uterus"


class PolygonRequest(BaseModel):
    image_id: str
    polygon: list[Point]
    class_name: str = "Uterus"
