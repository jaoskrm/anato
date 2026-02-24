"""
Embedding cache — file-backed with in-memory LRU eviction.
"""
import os
from collections import OrderedDict

import torch
import numpy as np

from ..config import settings
from .sam_service import sam_service
from ..utils.image_utils import load_image


class EmbeddingCache:
    """
    LRU cache for image embeddings.
    Stores tensors in memory and persists to disk as .pt files.
    """

    def __init__(self, max_size: int = settings.MAX_CACHE_SIZE):
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._max_size = max_size

    def _disk_path(self, image_id: str) -> str:
        return os.path.join(settings.EMBEDDING_DIR, f"{image_id}.pt")

    def has(self, image_id: str) -> bool:
        """Check if embedding exists in memory or on disk."""
        return image_id in self._cache or os.path.exists(self._disk_path(image_id))

    def get(self, image_id: str) -> torch.Tensor | None:
        """
        Retrieve embedding from memory cache or disk.
        Returns None if not found anywhere.
        """
        # Memory first
        if image_id in self._cache:
            self._cache.move_to_end(image_id)
            return self._cache[image_id]

        # Disk fallback
        disk_path = self._disk_path(image_id)
        if os.path.exists(disk_path):
            try:
                embedding = torch.load(disk_path, map_location=sam_service.device)
                self._put_memory(image_id, embedding)
                return embedding
            except Exception as e:
                print(f"Failed to load embedding from disk: {e}")
                return None

        return None

    def _put_memory(self, image_id: str, embedding: torch.Tensor) -> None:
        """Store in memory cache with LRU eviction."""
        if image_id in self._cache:
            self._cache.move_to_end(image_id)
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)  # evict oldest
            self._cache[image_id] = embedding

    def compute_and_store(self, image_id: str, image_path: str) -> torch.Tensor:
        """
        Compute embedding for an image, store in memory + disk.
        """
        img_rgb = load_image(image_path)
        embedding = sam_service.compute_embedding(img_rgb)

        # Save to disk (CPU tensor for portability)
        disk_path = self._disk_path(image_id)
        torch.save(embedding.cpu(), disk_path)

        # Store in memory
        self._put_memory(image_id, embedding)

        return embedding

    def get_or_compute(self, image_id: str, image_path: str) -> torch.Tensor:
        """
        Get from cache or compute fresh. One-stop method.
        """
        cached = self.get(image_id)
        if cached is not None:
            return cached
        return self.compute_and_store(image_id, image_path)


# Singleton
embedding_cache = EmbeddingCache()
