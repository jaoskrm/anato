"""
FastAPI application entry point.

Run with:
    cd /path/to/medseg
    python -m uvicorn backend.server.main:app --port 8000 --reload
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .services.sam_service import sam_service
from .routers import images, segment


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load SAM model at startup."""
    sam_service.load_model()
    yield
    # Cleanup if needed


app = FastAPI(
    title="MedSeg.io API",
    description="Medical Image Segmentation API powered by SAM",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(images.router)
app.include_router(segment.router)


@app.get("/")
async def root():
    return {"status": "ok", "service": "MedSeg.io API"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
