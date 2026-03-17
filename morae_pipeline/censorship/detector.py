"""NSFW region detection using imgutils (anime-optimized) + NudeNet (anus).

Dual-detector strategy:
  - detect_censors: penis, pussy (anime-trained, high accuracy)
  - detect_with_nudenet: ANUS_EXPOSED (NudeNet has anus, detect_censors doesn't)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single detected NSFW region."""

    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    mask: np.ndarray | None = None  # Pixel-level mask from SAM2


class CensorDetector:
    """Combined detector: imgutils detect_censors + NudeNet for anus.

    detect_censors labels: penis, pussy, nipple_f
    nudenet labels: ANUS_EXPOSED, FEMALE_GENITALIA_EXPOSED, MALE_GENITALIA_EXPOSED, etc.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.25,
        anus_confidence_threshold: float = 0.2,
        bbox_expand_ratio: float = 0.15,
    ):
        """
        Args:
            confidence_threshold: For detect_censors (penis/pussy)
            anus_confidence_threshold: For NudeNet anus detection (lower = catch more)
            bbox_expand_ratio: Expand pussy/anus bboxes by this ratio before SAM2
        """
        self.confidence_threshold = confidence_threshold
        self.anus_confidence_threshold = anus_confidence_threshold
        self.bbox_expand_ratio = bbox_expand_ratio

    def detect(self, image: Image.Image) -> list[Detection]:
        """Detect NSFW regions using dual detectors.

        Returns combined detections for penis, pussy, and anus.
        """
        detections = []

        # Primary: detect_censors (anime-optimized)
        detections.extend(self._detect_censors(image))

        # Supplement: NudeNet for anus
        detections.extend(self._detect_anus(image))

        # Deduplicate overlapping detections
        detections = self._deduplicate(detections)

        logger.debug(
            f"Combined detection: {len(detections)} regions "
            f"({', '.join(d.class_name for d in detections)})"
        )
        return detections

    def _detect_censors(self, image: Image.Image) -> list[Detection]:
        """Anime-optimized detection for penis and pussy."""
        from imgutils.detect.censor import detect_censors

        results = detect_censors(
            image,
            conf_threshold=self.confidence_threshold,
            iou_threshold=0.7,
        )

        target = {"penis", "pussy"}
        detections = []
        w, h = image.size

        for bbox, label, confidence in results:
            if label not in target:
                continue

            x1, y1, x2, y2 = bbox

            # Expand pussy bbox for better SAM2 coverage
            if label == "pussy":
                x1, y1, x2, y2 = self._expand_bbox(
                    x1, y1, x2, y2, w, h, self.bbox_expand_ratio
                )

            detections.append(
                Detection(
                    class_name=label,
                    confidence=float(confidence),
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                )
            )

        return detections

    def _detect_anus(self, image: Image.Image) -> list[Detection]:
        """NudeNet detection specifically for anus."""
        from imgutils.detect.nudenet import detect_with_nudenet

        results = detect_with_nudenet(
            image,
            topk=100,
            iou_threshold=0.45,
            score_threshold=self.anus_confidence_threshold,
        )

        detections = []
        w, h = image.size

        for bbox, label, confidence in results:
            if label != "ANUS_EXPOSED":
                continue

            x1, y1, x2, y2 = bbox
            # Expand anus bbox for better SAM2 coverage
            x1, y1, x2, y2 = self._expand_bbox(
                x1, y1, x2, y2, w, h, self.bbox_expand_ratio
            )

            detections.append(
                Detection(
                    class_name="anus",
                    confidence=float(confidence),
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                )
            )

        return detections

    def _expand_bbox(
        self,
        x1: float, y1: float, x2: float, y2: float,
        img_w: int, img_h: int,
        ratio: float,
    ) -> tuple[float, float, float, float]:
        """Expand bbox by ratio to ensure SAM2 gets enough context."""
        bw, bh = x2 - x1, y2 - y1
        dx, dy = bw * ratio, bh * ratio
        return (
            max(0, x1 - dx),
            max(0, y1 - dy),
            min(img_w, x2 + dx),
            min(img_h, y2 + dy),
        )

    def _deduplicate(
        self, detections: list[Detection], iou_threshold: float = 0.5
    ) -> list[Detection]:
        """Remove overlapping detections (keep higher confidence)."""
        if len(detections) <= 1:
            return detections

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)
        keep = []

        for det in detections:
            overlaps = False
            for kept in keep:
                if self._iou(det.bbox, kept.bbox) > iou_threshold:
                    overlaps = True
                    break
            if not overlaps:
                keep.append(det)

        return keep

    @staticmethod
    def _iou(
        box1: tuple[int, int, int, int],
        box2: tuple[int, int, int, int],
    ) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def is_available(self) -> bool:
        try:
            from imgutils.detect.censor import detect_censors  # noqa: F401
            from imgutils.detect.nudenet import detect_with_nudenet  # noqa: F401
            return True
        except ImportError:
            return False
