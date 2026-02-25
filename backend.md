## Backend Architecture Overview

Your backend is a **FastAPI** server that wraps the **Segment Anything Model (SAM)** from Meta. Here's the complete architecture:

### SAM Model Components (in `backend/segment_anything/modeling/`)

| Component | File | Purpose |
|-----------|------|---------|
| **Image Encoder** | [`image_encoder.py`](backend/segment_anything/modeling/image_encoder.py) | ViT (Vision Transformer) backbone - 1024x1024 input, 16x16 patches, 12 transformer blocks, 768 embed dim |
| **Prompt Encoder** | [`prompt_encoder.py`](backend/segment_anything/modeling/prompt_encoder.py) | Encodes box/point/text prompts into embeddings |
| **Mask Decoder** | [`mask_decoder.py`](backend/segment_anything/modeling/mask_decoder.py) | Transformer decoder that predicts binary masks from image + prompt embeddings |
| **SAM (Main)** | [`sam.py`](backend/segment_anything/modeling/sam.py) | Combines all three components |

### API Endpoints Exposed

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `GET /` | GET | Health check - returns `{"status": "ok"}` |
| `GET /health` | GET | Health check |
| `POST /api/images/upload` | POST | Upload image, starts embedding computation in background |
| `GET /api/images/{image_id}/status` | GET | Check if embedding is ready |
| `GET /api/images/{image_id}` | GET | Retrieve uploaded image |
| `GET /api/images/{image_id}/mask` | GET | Retrieve current mask PNG |
| `POST /api/segment/box` | POST | Segment with bounding box prompt |
| `POST /api/segment/polygon` | POST | Segment with polygon prompt |

### Backend Service Layer

| Service | File | Role |
|---------|------|------|
| **SAMService** | [`sam_service.py`](backend/server/services/sam_service.py) | Singleton model loader, thread-safe inference, embedding computation |
| **EmbeddingCache** | [`embedding_cache.py`](backend/server/services/embedding_cache.py) | LRU cache + disk persistence for image embeddings |

### Segmentation Flow

```
1. Upload Image (POST /api/images/upload)
   - Saves image to disk
   - Starts background embedding computation
   - Returns image_id

2. Embedding Computation (background)
   - Resizes image to 1024x1024
   - Runs through ViT image encoder
   - Caches embedding (memory + disk)

3. Segment (POST /api/segment/box)
   - Retrieves cached embedding
   - Encodes box prompt
   - Runs mask decoder
   - Applies class color to mask
   - Returns mask URL + polygon
```

### Model Details (from config)

- **Model Type**: `vit_b` (ViT-Base)
- **Checkpoint**: `backend/model/medsam_vit_b.pth` (MedSAM fine-tuned weights)
- **Device**: Auto-detects (CUDA > MPS > CPU)
- **Max Cache**: 100 embeddings

### What's NOT Currently Exposed

The raw encoder/decoder components are **not directly exposed** as separate endpoints. The current API is high-level:
- No endpoint to get raw embeddings
- No endpoint to access prompt encoder directly
- No endpoint for point prompts (only box/polygon)

Would you like me to add additional endpoints for more granular control?