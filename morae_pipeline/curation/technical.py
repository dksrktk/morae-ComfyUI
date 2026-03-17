"""Technical quality scorer: artifact, blur, and noise detection."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .scorer import ImageScorer


class TechnicalQualityScorer(ImageScorer):
    """Detects technical defects: blur, noise, artifacts.

    Measures "was it rendered well?" not "is it pretty?".
    An intentionally ugly character with clean rendering scores high.
    """

    name = "technical"
    weight = 0.5

    def score(self, image: Image.Image) -> float:
        arr = np.array(image, dtype=np.float32)

        blur_score = self._blur_score(arr)
        noise_score = self._noise_score(arr)
        artifact_score = self._artifact_score(arr)

        # Weighted combination
        composite = (
            blur_score * 0.4
            + noise_score * 0.3
            + artifact_score * 0.3
        )
        return float(np.clip(composite, 0.0, 1.0))

    def _blur_score(self, arr: np.ndarray) -> float:
        """Laplacian variance — higher = sharper."""
        gray = np.mean(arr, axis=2)

        # Laplacian kernel
        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

        # Manual 2D convolution (avoid scipy/cv2 dependency)
        from numpy.lib.stride_tricks import sliding_window_view
        h, w = gray.shape
        if h < 3 or w < 3:
            return 0.5

        windows = sliding_window_view(gray, (3, 3))
        lap = np.sum(windows * laplacian, axis=(-2, -1))
        variance = np.var(lap)

        # Normalize: typical sharp images have variance > 500, blurry < 100
        normalized = np.clip(variance / 1000.0, 0.0, 1.0)
        return float(normalized)

    def _noise_score(self, arr: np.ndarray) -> float:
        """Estimate noise level via median absolute deviation. Lower noise = higher score."""
        gray = np.mean(arr, axis=2)

        # High-pass filter: difference from local median (3x3 approximation)
        from numpy.lib.stride_tricks import sliding_window_view
        h, w = gray.shape
        if h < 3 or w < 3:
            return 0.5

        windows = sliding_window_view(gray, (3, 3))
        local_median = np.median(windows, axis=(-2, -1))
        diff = gray[1:-1, 1:-1] - local_median
        mad = np.median(np.abs(diff))

        # Lower MAD = cleaner image. Typical range: 0-20 for clean, 20+ for noisy
        noise_level = mad / 255.0
        score = 1.0 - np.clip(noise_level * 10, 0.0, 1.0)
        return float(score)

    def _artifact_score(self, arr: np.ndarray) -> float:
        """Detect blocking artifacts and color banding."""
        gray = np.mean(arr, axis=2)

        # Block artifact detection: variance of 8x8 block boundaries
        h, w = gray.shape
        if h < 16 or w < 16:
            return 0.8

        # Horizontal block boundaries (every 8 pixels)
        h_diffs = []
        for x in range(8, w - 8, 8):
            boundary_diff = np.abs(gray[:, x] - gray[:, x - 1])
            inner_diff = np.abs(gray[:, x + 1] - gray[:, x])
            # If boundary diffs >> inner diffs, likely JPEG artifacts
            if np.mean(inner_diff) > 0:
                ratio = np.mean(boundary_diff) / (np.mean(inner_diff) + 1e-6)
                h_diffs.append(ratio)

        if not h_diffs:
            return 0.8

        avg_ratio = np.mean(h_diffs)
        # Ratio close to 1.0 = no block artifacts. >2.0 = strong artifacts
        score = 1.0 - np.clip((avg_ratio - 1.0) / 2.0, 0.0, 1.0)

        # Color banding: count unique color levels in gradients
        unique_ratio = len(np.unique(gray.astype(np.uint8))) / 256.0
        banding_score = np.clip(unique_ratio, 0.0, 1.0)

        return float(score * 0.6 + banding_score * 0.4)
