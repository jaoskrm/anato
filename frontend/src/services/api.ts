/**
 * API service for communicating with the MedSeg backend.
 */

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Types matching backend schemas
export interface Point {
    x: number;
    y: number;
}

export interface Box {
    x: number;
    y: number;
    width: number;
    height: number;
}

export interface UploadResponse {
    image_id: string;
    width: number;
    height: number;
    embedding_status: string;
}

export interface EmbeddingStatusResponse {
    image_id: string;
    embedding_ready: boolean;
    message: string;
}

export interface BoxRequest {
    image_id: string;
    box: Box;
}

export interface PolygonRequest {
    image_id: string;
    polygon: Point[];
}

/**
 * Upload an image to the backend and start embedding computation.
 */
export async function uploadImage(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_BASE}/api/images/upload`, {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: "Upload failed" }));
        throw new Error(error.detail || "Upload failed");
    }

    return response.json();
}

/**
 * Check if embedding is ready for an image.
 */
export async function getEmbeddingStatus(imageId: string): Promise<EmbeddingStatusResponse> {
    const response = await fetch(`${API_BASE}/api/images/${imageId}/status`);

    if (!response.ok) {
        throw new Error("Failed to get embedding status");
    }

    return response.json();
}

/**
 * Wait for embedding to be ready, polling at intervals.
 */
export async function waitForEmbedding(
    imageId: string,
    maxAttempts = 60,
    intervalMs = 1000
): Promise<boolean> {
    for (let i = 0; i < maxAttempts; i++) {
        const status = await getEmbeddingStatus(imageId);
        if (status.embedding_ready) {
            return true;
        }
        await new Promise((resolve) => setTimeout(resolve, intervalMs));
    }
    return false;
}

/**
 * Run SAM segmentation with a box prompt.
 * Returns a binary mask as ImageBitmap (white = mask, black = background).
 */
export async function segmentBox(req: BoxRequest): Promise<ImageBitmap> {
    const response = await fetch(`${API_BASE}/api/segment/box`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(req),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: "Segmentation failed" }));
        throw new Error(error.detail || "Segmentation failed");
    }

    const blob = await response.blob();
    return createImageBitmap(blob);
}

/**
 * Run SAM segmentation with a polygon prompt.
 * Returns a binary mask as ImageBitmap (white = mask, black = background).
 */
export async function segmentPolygon(req: PolygonRequest): Promise<ImageBitmap> {
    const response = await fetch(`${API_BASE}/api/segment/polygon`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(req),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: "Segmentation failed" }));
        throw new Error(error.detail || "Segmentation failed");
    }

    const blob = await response.blob();
    return createImageBitmap(blob);
}

/**
 * Get the URL for an uploaded image.
 */
export function getImageUrl(imageId: string): string {
    return `${API_BASE}/api/images/${imageId}`;
}

/**
 * Load an image from URL and return as HTMLImageElement.
 */
export async function loadImageFromUrl(url: string): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error(`Failed to load image: ${url}`));
        img.src = url;
    });
}
