"""
Application configuration via environment variables.
"""
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # SAM model
    SAM_CHECKPOINT_PATH: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "model", "medsam_vit_b.pth",
    )
    SAM_MODEL_TYPE: str = "vit_b"
    DEVICE: str = "auto"  # "auto", "cuda", "mps", or "cpu"

    # Storage
    UPLOAD_DIR: str = os.path.join(
        os.path.dirname(__file__), "uploads"
    )
    EMBEDDING_DIR: str = os.path.join(
        os.path.dirname(__file__), "embeddings"
    )
    MASK_DIR: str = os.path.join(
        os.path.dirname(__file__), "masks"
    )

    # Limits
    MAX_CACHE_SIZE: int = 100
    MAX_UPLOAD_SIZE_MB: int = 50

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.EMBEDDING_DIR, exist_ok=True)
os.makedirs(settings.MASK_DIR, exist_ok=True)
