"""NSFW region detection using YOLO models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Default NSFW classes for YOLO models trained on anime/real datasets
DEFAULT_TARGET_CLASSES = [
    "genitalia_exposed",
    "anus_exposed",
    "female_breast_exposed",
    "male_breast_exposed",
]


@dataclass
class Detection:
    """A single detected NSFW region."""

    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    mask: np.ndarray | None = None  # Optional pixel-level mask from SAM2


class YOLODetector:
    """YOLO-based NSFW body part detector.

    Supports any YOLO .pt model trained on NSFW datasets
    (e.g. anime/booru-trained YOLOv8/v11 models).
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        confidence_threshold: float = 0.3,
        target_classes: list[str] | None = None,
    ):
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes or DEFAULT_TARGET_CLASSES
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for NSFW detection. "
                "Install with: pip install ultralytics"
            )

        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"YOLO model not found at {path}. "
                "Download an NSFW-trained YOLO model (.pt) and set the path in config."
            )

        self._model = YOLO(str(path))
        logger.info(f"YOLO model loaded from {path} (device={self.device})")

    def detect(self, image: Image.Image) -> list[Detection]:
        """Detect NSFW regions in an image.

        Args:
            image: PIL Image (RGB)

        Returns:
            List of Detection objects for regions above confidence threshold
        """
        self._load_model()

        results = self._model.predict(
            source=image,
            device=self.device,
            conf=self.confidence_threshold,
            verbose=False,
        )

        detections = []
        if not results or len(results) == 0:
            return detections

        result = results[0]
        class_names = result.names  # {id: class_name}

        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = class_names.get(cls_id, f"class_{cls_id}")
            conf = float(box.conf[0])

            # Filter by target classes if specified
            if self.target_classes and cls_name not in self.target_classes:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            detections.append(
                Detection(
                    class_name=cls_name,
                    confidence=conf,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                )
            )

        logger.debug(
            f"Detected {len(detections)} NSFW regions "
            f"(threshold={self.confidence_threshold})"
        )
        return detections

    def is_available(self) -> bool:
        """Check if YOLO model and ultralytics are available."""
        try:
            from ultralytics import YOLO  # noqa: F401
        except ImportError:
            return False
        return Path(self.model_path).exists()


class SAMRefiner:
    """Optional SAM2-based mask refinement for pixel-precise segmentation.

    Takes YOLO bounding boxes and produces tight segmentation masks.
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self._predictor = None

    def _load_model(self):
        if self._predictor is not None:
            return

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            raise ImportError(
                "sam2 is required for precise segmentation. "
                "Install with: pip install sam2"
            )

        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(f"SAM2 model not found at {path}")

        sam_model = build_sam2(str(path), device=self.device)
        self._predictor = SAM2ImagePredictor(sam_model)
        logger.info(f"SAM2 model loaded from {path}")

    def refine(
        self, image: Image.Image, detections: list[Detection]
    ) -> list[Detection]:
        """Refine bounding boxes into pixel-level masks.

        Args:
            image: Original PIL Image
            detections: List of detections with bounding boxes

        Returns:
            Same detections with mask field populated
        """
        if not detections:
            return detections

        self._load_model()

        img_array = np.array(image)
        self._predictor.set_image(img_array)

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            input_box = np.array([[x1, y1, x2, y2]])

            masks, scores, _ = self._predictor.predict(
                box=input_box,
                multimask_output=False,
            )
            # Use highest scoring mask
            det.mask = masks[0].astype(np.uint8) * 255

        logger.debug(f"SAM2 refined {len(detections)} masks")
        return detections

    def is_available(self) -> bool:
        try:
            from sam2.build_sam import build_sam2  # noqa: F401
        except ImportError:
            return False
        return Path(self.model_path).exists()
