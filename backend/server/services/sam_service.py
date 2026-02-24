"""
SAM model service — singleton loader, embedding computation, and inference.
Extracted from gui_main.py.
"""
import threading
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from segment_anything import sam_model_registry

from ..config import settings

MODEL_LOCK = threading.Lock()


class SAMService:
    """Thread-safe wrapper around the SAM model."""

    def __init__(self):
        self._model = None
        self._device = None

    @property
    def device(self) -> torch.device:
        if self._device is None:
            if settings.DEVICE == "auto":
                if torch.backends.mps.is_available():
                    self._device = torch.device("mps")
                elif torch.cuda.is_available():
                    self._device = torch.device("cuda:0")
                else:
                    self._device = torch.device("cpu")
            else:
                self._device = torch.device(settings.DEVICE)
        return self._device

    @property
    def model(self):
        if self._model is None:
            self.load_model()
        return self._model

    def load_model(self) -> None:
        """Load SAM model from checkpoint."""
        print(f"Loading SAM model ({settings.SAM_MODEL_TYPE}) from {settings.SAM_CHECKPOINT_PATH}...")
        self._model = sam_model_registry[settings.SAM_MODEL_TYPE](
            checkpoint=settings.SAM_CHECKPOINT_PATH,
        ).to(self.device)
        self._model.eval()
        print(f"SAM model loaded on {self.device}")

    def compute_embedding(self, image_rgb: np.ndarray) -> torch.Tensor:
        """
        Compute image embedding from an RGB numpy array.
        Returns the embedding tensor (on device).
        """
        # Resize to 1024x1024
        img_1024 = cv2.resize(image_rgb, (1024, 1024), interpolation=cv2.INTER_CUBIC)

        # Normalize to [0, 1]
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None,
        )

        # To tensor (1, 3, 1024, 1024)
        img_tensor = (
            torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        )

        with MODEL_LOCK:
            with torch.no_grad():
                embedding = self.model.image_encoder(img_tensor)

        return embedding

    @torch.no_grad()
    def segment_box(
        self,
        embedding: torch.Tensor,
        box_coords: list[float],
        height: int,
        width: int,
    ) -> np.ndarray:
        """
        Run SAM inference with a box prompt.

        Args:
            embedding: precomputed image embedding
            box_coords: [xmin, ymin, xmax, ymax] in 1024-space
            height: original image height
            width: original image width

        Returns:
            Binary mask (H, W) as uint8
        """
        embedding = embedding.to(self.device)
        box_tensor = torch.as_tensor(
            [box_coords], dtype=torch.float, device=self.device
        )
        if len(box_tensor.shape) == 2:
            box_tensor = box_tensor[:, None, :]

        with MODEL_LOCK:
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None, boxes=box_tensor, masks=None,
            )
            low_res_logits, _ = self.model.mask_decoder(
                image_embeddings=embedding,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

        low_res_pred = torch.sigmoid(low_res_logits)
        low_res_pred = F.interpolate(
            low_res_pred, size=(height, width), mode="bilinear", align_corners=False,
        )
        low_res_pred = low_res_pred.squeeze().cpu().numpy()
        return (low_res_pred > 0.5).astype(np.uint8)

    def segment_polygon(
        self,
        embedding: torch.Tensor,
        polygon: list[tuple[float, float]],
        height: int,
        width: int,
    ) -> np.ndarray:
        """
        Run SAM inference with a polygon prompt.
        The polygon is used to compute a bounding box for SAM,
        then the result is masked to stay within the polygon.
        """
        pts = np.array(polygon, dtype=np.int32)
        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)

        # Skip if too small
        if (xmax - xmin) < 5 or (ymax - ymin) < 5:
            return np.zeros((height, width), dtype=np.uint8)

        # Scale box to 1024 space
        box_1024 = [
            xmin / width * 1024,
            ymin / height * 1024,
            xmax / width * 1024,
            ymax / height * 1024,
        ]

        # Run SAM inference
        sam_mask = self.segment_box(embedding, box_1024, height, width)

        # Create polygon mask and intersect
        poly_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [pts], 1)

        return (sam_mask * poly_mask).astype(np.uint8)


# Singleton instance
sam_service = SAMService()
