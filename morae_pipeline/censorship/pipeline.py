"""Censorship pipeline: detect (imgutils) -> segment (SAM2) -> refine -> composite.

Flow:
  1. detect_censors() — bbox detection (penis, pussy, etc.)
  2. SAM2Refiner    — pixel-precise mask from bbox
  3. MaskRefiner    — erode to minimize coverage
  4. RegionFilter   — white fill on mask area
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from ..config import CensorshipConfig
from .detector import Detection, CensorDetector
from .filter import RegionFilter, FilterMethod
from .segmentation import SAM2Refiner, GrabCutRefiner, MaskRefiner

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
    """Full censorship pipeline using imgutils detection + SAM2 segmentation.

    Produces dual output:
      - master/  : uncensored originals (graded/)
      - service/ : censored versions for compliant distribution
    """

    def __init__(self, config: CensorshipConfig):
        self.config = config
        self._detect_available = False
        self._segmenter = None
        self._mask_refiner = None
        self._filter = None
        self._setup()

    def _setup(self) -> None:
        # Combined detector (detect_censors + NudeNet anus)
        self._detector = CensorDetector(
            confidence_threshold=self.config.confidence_threshold,
            anus_confidence_threshold=getattr(self.config, 'anus_confidence_threshold', 0.2),
            bbox_expand_ratio=getattr(self.config, 'bbox_expand_ratio', 0.15),
        )
        if not self._detector.is_available():
            logger.warning(
                "imgutils not available. Install with: pip install dghs-imgutils"
            )
            return
        self._detect_available = True
        logger.info("Using dual detector: detect_censors + NudeNet(anus)")

        # Segmentation: prefer SAM2, fallback to GrabCut
        sam2 = SAM2Refiner(
            model_cfg=self.config.sam2_model_cfg,
            checkpoint=self.config.sam2_checkpoint,
            device=self.config.device,
        )
        if sam2.is_available():
            self._segmenter = sam2
            logger.info("Using SAM2 segmentation")
        else:
            self._segmenter = GrabCutRefiner()
            logger.info("SAM2 not available, using GrabCut fallback")

        # Mask refinement (erode to minimize coverage)
        self._mask_refiner = MaskRefiner(
            erode_pixels=self.config.erode_pixels,
            dilate_pixels=self.config.dilate_pixels,
            blur_boundary=self.config.blur_boundary,
        )

        # Filter (white fill by default)
        self._filter = RegionFilter(
            method=FilterMethod(self.config.filter_method),
            blur_radius=self.config.blur_radius,
            pixelate_factor=self.config.pixelate_factor,
            mask_padding=0,  # No padding — SAM2 mask is already precise
        )

        logger.info(
            f"Censorship pipeline ready: "
            f"detector=imgutils(conf={self.config.confidence_threshold}), "
            f"segmenter={'SAM2' if isinstance(self._segmenter, SAM2Refiner) else 'GrabCut'}, "
            f"erode={self.config.erode_pixels}px, "
            f"filter={self.config.filter_method}"
        )

    def _detect(self, image: Image.Image) -> list[Detection]:
        """Run dual detector (detect_censors + NudeNet anus)."""
        return self._detector.detect(image)

    def process_image(self, image_path: Path, output_path: Path) -> CensorshipResult:
        """Process a single image: detect -> segment -> refine -> composite.

        Args:
            image_path: Path to source image
            output_path: Path to save censored version

        Returns:
            CensorshipResult with detection details
        """
        result = CensorshipResult(source_path=str(image_path))

        if not self._detect_available:
            return result

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to open {image_path}: {e}")
            return result

        # Step 1: Detect (imgutils — anime-optimized)
        detections = self._detect(image)

        if not detections:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, quality=95)
            result.censored_path = str(output_path)
            return result

        # Step 2: Segment (SAM2 — pixel-precise mask from bbox)
        if self._segmenter:
            try:
                detections = self._segmenter.refine(image, detections)
            except Exception as e:
                logger.warning(f"Segmentation failed, using bbox: {e}")

        # Step 3: Refine mask (erode to minimize)
        if self._mask_refiner:
            detections = self._mask_refiner.refine(detections)

        # Step 4: Composite (white fill on mask)
        censored = self._filter.apply(image, detections)

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
                "has_mask": d.mask is not None,
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
        """Process a batch of images."""
        if not self._detect_available:
            logger.warning("Censorship skipped: imgutils not available")
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
            f"Censorship complete: {len(results)} images in {elapsed:.1f}s, "
            f"{censored_count} censored ({total_detections} detections)"
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
