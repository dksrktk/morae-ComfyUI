"""Face similarity scorer using InsightFace/ArcFace."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .scorer import ImageScorer

logger = logging.getLogger(__name__)

_app = None
_device = None


def _load_insightface(device: str = "cuda") -> bool:
    """Load InsightFace model. Returns True on success."""
    global _app, _device

    if _app is not None:
        return True

    try:
        from insightface.app import FaceAnalysis

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )

        app = FaceAnalysis(
            name="buffalo_l",
            providers=providers,
        )
        app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))

        _app = app
        _device = device
        logger.info("InsightFace loaded successfully")
        return True

    except ImportError:
        logger.info("InsightFace not installed — face similarity scoring disabled")
        return False
    except Exception as e:
        logger.warning(f"Failed to load InsightFace: {e}")
        return False


def _get_embedding(image: Image.Image) -> Optional[np.ndarray]:
    """Extract face embedding from image. Returns None if no face detected."""
    arr = np.array(image.convert("RGB"))
    # InsightFace expects BGR
    arr_bgr = arr[:, :, ::-1]

    faces = _app.get(arr_bgr)
    if not faces:
        return None

    # Use the largest face (by bbox area)
    largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return largest.normed_embedding


class FaceSimilarityScorer(ImageScorer):
    """Compare generated face to a reference image using ArcFace embeddings.

    Returns cosine similarity as score (0.0 ~ 1.0).
    Returns 0.0 if no face detected (neutral, not penalizing).
    """

    name = "face"
    weight = 0.2

    def __init__(self, reference_path: Optional[str] = None, device: str = "cuda"):
        self._reference_path = reference_path
        self._reference_embedding: Optional[np.ndarray] = None
        self._device = device
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._available is None:
            if self._reference_path is None:
                self._available = False
                return False
            self._available = _load_insightface(self._device)
            if self._available:
                self._load_reference()
        return self._available and self._reference_embedding is not None

    def _load_reference(self) -> None:
        """Extract and cache reference face embedding."""
        if self._reference_path is None:
            return
        ref_img = Image.open(self._reference_path).convert("RGB")
        self._reference_embedding = _get_embedding(ref_img)
        if self._reference_embedding is None:
            logger.warning(f"No face detected in reference image: {self._reference_path}")

    def score(self, image: Image.Image) -> float:
        if not self.is_available():
            return 0.0

        gen_embedding = _get_embedding(image)
        if gen_embedding is None:
            return 0.0  # No face detected — neutral

        # Cosine similarity (embeddings are already L2-normalized by InsightFace)
        similarity = float(np.dot(self._reference_embedding, gen_embedding))

        # Remap: typical ArcFace similarities for same person are 0.3~0.7
        # Map [0.2, 0.7] -> [0.0, 1.0] for more useful range
        remapped = (similarity - 0.2) / 0.5
        return float(np.clip(remapped, 0.0, 1.0))
