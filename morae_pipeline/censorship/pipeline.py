"""Censorship pipeline - AutoCensor style."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..config import CensorshipConfig
from .detector import Detection, YOLOSegmentationDetector, CensorDetector, LABEL_GROUPS
from .segmentation import (
    refine_mask_autocensor_style,
    unify_mask_fragments,
    SAM2Refiner,
)

logger = logging.getLogger(__name__)


@dataclass
class CensorshipResult:
    source_path: str
    censored_path: str | None = None
    detections: list[dict] = field(default_factory=list)
    detection_count: int = 0
    was_censored: bool = False

    def to_dict(self) -> dict:
        return {
            "source_path": self.source_path,
            "censored_path": self.censored_path,
            "detections": self.detections,
            "detection_count": self.detection_count,
            "was_censored": self.was_censored,
        }


class CensorshipPipeline:
    """AutoCensor-style censorship pipeline."""

    def __init__(self, config: CensorshipConfig):
        self.config = config
        self._yolo_detector = None
        self._imgutils_detector = None
        self._sam2_refiner = None
        self._setup()

    def _setup(self) -> None:
        # Primary: YOLO Segmentation (masks included)
        yolo_model = getattr(self.config, 'yolo_segm_model',
                            'models/yolo/ntd11_anime_nsfw_segm_v5-variant1.pt')
        yolo = YOLOSegmentationDetector(
            model_path=yolo_model,
            confidence_threshold=self.config.confidence_threshold,
            device=self.config.device,
        )

        if yolo.is_available():
            self._yolo_detector = yolo
            logger.info(f"YOLO Segmentation: {yolo_model}")

        # Secondary: imgutils (for cases YOLO misses)
        imgutils = CensorDetector(
            confidence_threshold=self.config.confidence_threshold,
            anus_confidence_threshold=getattr(self.config, 'anus_confidence_threshold', 0.2),
            bbox_expand_ratio=getattr(self.config, 'bbox_expand_ratio', 0.15),
        )
        if imgutils.is_available():
            self._imgutils_detector = imgutils
            logger.info("imgutils detector loaded (fallback)")

        # SAM2 for generating masks from imgutils bbox
        sam2 = SAM2Refiner(
            model_cfg=self.config.sam2_model_cfg,
            checkpoint=self.config.sam2_checkpoint,
            device=self.config.device,
        )
        if sam2.is_available():
            self._sam2_refiner = sam2
            logger.info("SAM2 loaded for mask generation")

        if not self._yolo_detector and not self._imgutils_detector:
            logger.warning("No detector available")
            return

        # Censor fill color
        self._censor_color = (255, 255, 255, 255)  # white
        if self.config.filter_method == "black_bar":
            self._censor_color = (0, 0, 0, 255)

        logger.info(f"Pipeline ready: YOLO={self._yolo_detector is not None}, imgutils={self._imgutils_detector is not None}, SAM2={self._sam2_refiner is not None}")

    def process_image(self, image_path: Path, output_path: Path) -> CensorshipResult:
        """Process single image - hybrid YOLO + imgutils detection."""
        result = CensorshipResult(source_path=str(image_path))

        if not self._yolo_detector and not self._imgutils_detector:
            return result

        try:
            img_pil = Image.open(image_path).convert("RGBA")
        except Exception as e:
            logger.error(f"Failed to open {image_path}: {e}")
            return result

        w, h = img_pil.size
        img_np = np.array(img_pil, dtype=np.float32)
        img_rgb = img_pil.convert("RGB")

        # Step 1: YOLO detection (with masks)
        yolo_detections = []
        if self._yolo_detector:
            yolo_detections = self._yolo_detector.detect(img_rgb)

        # Step 2: imgutils detection (bbox only)
        imgutils_detections = []
        if self._imgutils_detector:
            imgutils_detections = self._imgutils_detector.detect(img_rgb)

        # Step 3: Find imgutils detections that YOLO missed
        missed_detections = self._find_missed_detections(yolo_detections, imgutils_detections)

        # Step 4: Generate masks for missed detections using SAM2
        if missed_detections and self._sam2_refiner:
            try:
                missed_detections = self._sam2_refiner.refine(img_rgb, missed_detections)
                logger.debug(f"SAM2 generated masks for {len(missed_detections)} missed detections")
            except Exception as e:
                logger.warning(f"SAM2 mask generation failed: {e}")

        # Combine detections
        detections = yolo_detections + missed_detections

        if not detections:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img_pil.convert("RGB").save(output_path, quality=95)
            result.censored_path = str(output_path)
            return result

        # Apply censorship
        applied = False
        for det in detections:
            if det.mask is None:
                continue

            # Unify fragmented mask (Closing + Convex Hull)
            unified_mask = unify_mask_fragments(det.mask, use_convex_hull=True)

            # AutoCensor-style refinement (B-spline + supersample + AA)
            mask_np = refine_mask_autocensor_style(
                unified_mask,
                target_size=(w, h),
                blur_sigma=2.0,
                supersample=4,
            )

            # Alpha blending
            alpha = (mask_np.astype(np.float32) / 255.0)[:, :, np.newaxis]
            fill_arr = np.array(self._censor_color, dtype=np.float32)
            img_np = fill_arr * alpha + img_np * (1.0 - alpha)
            applied = True

        if applied:
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np, "RGBA")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        img_pil.convert("RGB").save(output_path, quality=95)

        result.censored_path = str(output_path)
        result.detection_count = len(detections)
        result.was_censored = applied
        result.detections = [
            {"class": d.class_name, "confidence": round(d.confidence, 4),
             "bbox": list(d.bbox), "has_mask": d.mask is not None,
             "source": "yolo" if d in yolo_detections else "imgutils+sam2"}
            for d in detections
        ]

        return result

    def _find_missed_detections(
        self, yolo_dets: list[Detection], imgutils_dets: list[Detection]
    ) -> list[Detection]:
        """Find detections from imgutils that YOLO missed.

        Uses IoU (Intersection over Union) to match detections.
        If imgutils detection has no matching YOLO detection (IoU < 0.3), it's missed.
        """
        if not imgutils_dets:
            return []

        missed = []

        for img_det in imgutils_dets:
            # Check if any YOLO detection overlaps significantly
            matched = False
            for yolo_det in yolo_dets:
                iou = self._compute_iou(img_det.bbox, yolo_det.bbox)
                if iou > 0.3:  # Threshold for considering it matched
                    matched = True
                    break

            if not matched:
                logger.debug(f"imgutils found missed {img_det.class_name} at {img_det.bbox}")
                missed.append(img_det)

        return missed

    @staticmethod
    def _compute_iou(bbox1: tuple, bbox2: tuple) -> float:
        """Compute Intersection over Union between two bboxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def process_batch(self, image_paths: list[Path], output_dir: Path) -> list[CensorshipResult]:
        results = []
        start = time.time()

        for i, path in enumerate(image_paths):
            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"Censoring {i + 1}/{len(image_paths)}...")
            result = self.process_image(path, output_dir / path.name)
            results.append(result)

        elapsed = time.time() - start
        censored = sum(1 for r in results if r.was_censored)
        total_det = sum(r.detection_count for r in results)
        logger.info(f"Censorship complete: {len(results)} images in {elapsed:.1f}s, "
                    f"{censored} censored ({total_det} detections)")
        return results

    def write_report(self, results: list[CensorshipResult], output_path: Path) -> Path:
        data = {
            "total": len(results),
            "censored": sum(1 for r in results if r.was_censored),
            "clean": sum(1 for r in results if not r.was_censored),
            "total_detections": sum(r.detection_count for r in results),
            "results": [r.to_dict() for r in results],
        }
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        logger.info(f"Report written to {output_path}")
        return output_path
