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
from .detector import Detection, YOLOSegmentationDetector, CensorDetector
from .segmentation import refine_mask_autocensor_style, SAM2Refiner, GrabCutRefiner, MaskRefiner

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
        self._detector = None
        self._use_yolo = False
        self._segmenter = None
        self._setup()

    def _setup(self) -> None:
        # Primary: YOLO Segmentation (AutoCensor style)
        yolo_model = getattr(self.config, 'yolo_segm_model',
                            'models/yolo/ntd11_anime_nsfw_segm_v5-variant1.pt')
        yolo = YOLOSegmentationDetector(
            model_path=yolo_model,
            confidence_threshold=self.config.confidence_threshold,
            device=self.config.device,
        )

        if yolo.is_available():
            self._detector = yolo
            self._use_yolo = True
            logger.info(f"Using YOLO Segmentation: {yolo_model}")
        else:
            # Fallback: imgutils + SAM2
            self._detector = CensorDetector(
                confidence_threshold=self.config.confidence_threshold,
                anus_confidence_threshold=getattr(self.config, 'anus_confidence_threshold', 0.2),
                bbox_expand_ratio=getattr(self.config, 'bbox_expand_ratio', 0.15),
            )
            if not self._detector.is_available():
                logger.warning("No detector available")
                return
            logger.info("Using fallback: imgutils + SAM2")

            sam2 = SAM2Refiner(
                model_cfg=self.config.sam2_model_cfg,
                checkpoint=self.config.sam2_checkpoint,
                device=self.config.device,
            )
            if sam2.is_available():
                self._segmenter = sam2
            else:
                self._segmenter = GrabCutRefiner()

        # Censor fill color
        self._censor_color = (255, 255, 255, 255)  # white
        if self.config.filter_method == "black_bar":
            self._censor_color = (0, 0, 0, 255)

        logger.info(f"Pipeline ready: {'YOLO' if self._use_yolo else 'imgutils+SAM2'}")

    def process_image(self, image_path: Path, output_path: Path) -> CensorshipResult:
        """Process single image - AutoCensor style."""
        result = CensorshipResult(source_path=str(image_path))

        if self._detector is None:
            return result

        try:
            img_pil = Image.open(image_path).convert("RGBA")
        except Exception as e:
            logger.error(f"Failed to open {image_path}: {e}")
            return result

        w, h = img_pil.size
        img_np = np.array(img_pil, dtype=np.float32)

        # Detect
        detections = self._detector.detect(img_pil.convert("RGB"))

        if not detections:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img_pil.convert("RGB").save(output_path, quality=95)
            result.censored_path = str(output_path)
            return result

        # If fallback (imgutils), need SAM2 for masks
        if not self._use_yolo and self._segmenter:
            try:
                detections = self._segmenter.refine(img_pil.convert("RGB"), detections)
            except Exception as e:
                logger.warning(f"Segmentation failed: {e}")

        # Apply censorship - AutoCensor style
        applied = False
        for det in detections:
            if det.mask is None:
                continue

            # AutoCensor: refine_mask
            mask_np = refine_mask_autocensor_style(
                det.mask,
                target_size=(w, h),
                blur_sigma=2.0,
                supersample=4,
            )

            # AutoCensor: alpha blending
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
             "bbox": list(d.bbox), "has_mask": d.mask is not None}
            for d in detections
        ]

        return result

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
