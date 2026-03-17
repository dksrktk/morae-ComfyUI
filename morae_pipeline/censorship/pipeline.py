"""Censorship pipeline: detect NSFW regions -> segment -> filter."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from ..config import CensorshipConfig
from .detector import YOLODetector, SAMRefiner, Detection
from .filter import RegionFilter, FilterMethod

logger = logging.getLogger(__name__)


@dataclass
class CensorshipResult:
    """Result of censoring a single image."""

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
    """Full censorship pipeline: YOLO detection -> optional SAM2 refinement -> filtering.

    Produces dual output:
      - master/  : uncensored originals
      - service/ : censored versions for compliant distribution
    """

    def __init__(self, config: CensorshipConfig):
        self.config = config
        self.detector: YOLODetector | None = None
        self.refiner: SAMRefiner | None = None
        self.filter: RegionFilter | None = None
        self._setup()

    def _setup(self) -> None:
        # Detector
        self.detector = YOLODetector(
            model_path=self.config.yolo_model_path,
            device=self.config.device,
            confidence_threshold=self.config.confidence_threshold,
            target_classes=self.config.target_classes or None,
        )

        if not self.detector.is_available():
            logger.warning(
                f"YOLO model not available at {self.config.yolo_model_path}. "
                "Censorship pipeline will be skipped."
            )
            self.detector = None
            return

        # Optional SAM2 refiner
        if self.config.sam2_enabled and self.config.sam2_model_path:
            self.refiner = SAMRefiner(
                model_path=self.config.sam2_model_path,
                device=self.config.device,
            )
            if not self.refiner.is_available():
                logger.warning("SAM2 not available, falling back to bbox-only masking")
                self.refiner = None

        # Filter
        self.filter = RegionFilter(
            method=FilterMethod(self.config.filter_method),
            blur_radius=self.config.blur_radius,
            pixelate_factor=self.config.pixelate_factor,
            mask_padding=self.config.mask_padding,
        )

        logger.info(
            f"Censorship pipeline ready: "
            f"detector=YOLO(conf={self.config.confidence_threshold}), "
            f"refiner={'SAM2' if self.refiner else 'bbox'}, "
            f"filter={self.config.filter_method}"
        )

    def process_image(self, image_path: Path, output_path: Path) -> CensorshipResult:
        """Process a single image: detect -> refine -> filter -> save.

        Args:
            image_path: Path to source image
            output_path: Path to save censored version

        Returns:
            CensorshipResult with detection details
        """
        result = CensorshipResult(source_path=str(image_path))

        if self.detector is None:
            return result

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to open {image_path}: {e}")
            return result

        # Step 1: Detect
        detections = self.detector.detect(image)

        if not detections:
            # No NSFW regions found — copy original as-is
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, quality=95)
            result.censored_path = str(output_path)
            return result

        # Step 2: Refine masks (optional)
        if self.refiner:
            try:
                detections = self.refiner.refine(image, detections)
            except Exception as e:
                logger.warning(f"SAM2 refinement failed, using bbox: {e}")

        # Step 3: Filter
        censored = self.filter.apply(image, detections)

        # Save censored image
        output_path.parent.mkdir(parents=True, exist_ok=True)
        censored.save(output_path, quality=95)

        result.censored_path = str(output_path)
        result.detection_count = len(detections)
        result.was_censored = True
        result.detections = [
            {
                "class": d.class_name,
                "confidence": round(d.confidence, 4),
                "bbox": list(d.bbox),
            }
            for d in detections
        ]

        logger.debug(
            f"{image_path.name}: {len(detections)} regions censored "
            f"({', '.join(d.class_name for d in detections)})"
        )
        return result

    def process_batch(
        self, image_paths: list[Path], output_dir: Path
    ) -> list[CensorshipResult]:
        """Process a batch of images.

        Args:
            image_paths: Source image paths
            output_dir: Directory to write censored versions

        Returns:
            List of CensorshipResult
        """
        if self.detector is None:
            logger.warning("Censorship skipped: YOLO model not available")
            return []

        results = []
        censored_count = 0
        start = time.time()

        for i, path in enumerate(image_paths):
            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"Censoring {i + 1}/{len(image_paths)}...")

            output_path = output_dir / path.name
            result = self.process_image(path, output_path)
            results.append(result)

            if result.was_censored:
                censored_count += 1

        elapsed = time.time() - start
        total_detections = sum(r.detection_count for r in results)
        logger.info(
            f"Censorship complete: {len(results)} images processed in {elapsed:.1f}s, "
            f"{censored_count} censored ({total_detections} total detections)"
        )
        return results

    def write_report(
        self, results: list[CensorshipResult], output_path: Path
    ) -> Path:
        """Write censorship report as JSON."""
        data = {
            "total": len(results),
            "censored": sum(1 for r in results if r.was_censored),
            "clean": sum(1 for r in results if not r.was_censored),
            "total_detections": sum(r.detection_count for r in results),
            "results": [r.to_dict() for r in results],
        }
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        logger.info(f"Censorship report written to {output_path}")
        return output_path
