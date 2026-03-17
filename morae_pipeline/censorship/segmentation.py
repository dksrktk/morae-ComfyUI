"""Pixel-level segmentation refinement for precise censorship masking.

Pipeline:
  1. SAM2 mask (pixel-precise contour)
  2. Semantic subtraction (remove face/tongue from mask)
  3. Distance Transform gradient (smooth edges following contour shape)
  4. Anus: elliptical gradient (SAM2 unreliable for anus)
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .detector import Detection

logger = logging.getLogger(__name__)


class SAM2Refiner:
    """SAM2 segmentation + semantic subtraction + distance transform smoothing.

    Pipeline per detection:
      1. SAM2 produces pixel mask from bbox
      2. Face/tongue regions are subtracted (semantic subtraction)
      3. Largest connected component kept (fragment cleanup)
      4. Distance transform creates smooth gradient edges
      5. Anus uses elliptical gradient instead of SAM2
    """

    def __init__(
        self,
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        checkpoint: str = "./models/sam2/sam2.1_hiera_large.pt",
        device: str = "cuda",
    ):
        self.model_cfg = model_cfg
        self.checkpoint = checkpoint
        self.device = device
        self._predictor = None

    def _load_model(self):
        if self._predictor is not None:
            return

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        path = Path(self.checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found at {path}")

        sam_model = build_sam2(self.model_cfg, str(path), device=self.device)
        self._predictor = SAM2ImagePredictor(sam_model)
        logger.info(f"SAM2 loaded: {self.model_cfg} (device={self.device})")

    def refine(
        self, image: Image.Image, detections: list[Detection]
    ) -> list[Detection]:
        """Refine detections into smooth, anatomically-aware masks.

        Args:
            image: Original PIL Image (RGB)
            detections: Detections with bounding boxes

        Returns:
            Detections with gradient masks (uint8 0-255)
        """
        if not detections:
            return detections

        self._load_model()

        img_array = np.array(image)
        self._predictor.set_image(img_array)
        h, w = img_array.shape[:2]

        for det in detections:
            # Anus: elliptical gradient (SAM2 is unreliable)
            if det.class_name == "anus":
                det.mask = self._anus_gradient(det.bbox, h, w)
                continue

            # Step 1: SAM2 pixel mask
            x1, y1, x2, y2 = det.bbox
            input_box = np.array([[x1, y1, x2, y2]])
            masks, scores, _ = self._predictor.predict(
                box=input_box,
                multimask_output=True,
            )
            best_idx = int(np.argmax(scores))
            raw_mask = (masks[best_idx] * 255).astype(np.uint8)

            # Step 2: Keep largest connected component
            raw_mask = self._keep_largest_component(raw_mask)

            # Step 4: Clean mask ready
            det.mask = raw_mask

        logger.debug(f"Refined {len(detections)} masks (SAM2 + subtraction + distance gradient)")
        return detections

    @staticmethod
    def _get_exclusion_mask(image: Image.Image, h: int, w: int) -> np.ndarray | None:
        """Detect face/tongue regions to subtract from genital masks."""
        try:
            from imgutils.detect.nudenet import detect_with_nudenet
        except ImportError:
            return None

        results = detect_with_nudenet(
            image,
            topk=50,
            iou_threshold=0.45,
            score_threshold=0.3,
        )

        exclude_labels = {"FACE_FEMALE", "FACE_MALE"}
        mask = np.zeros((h, w), dtype=np.uint8)

        for bbox, label, conf in results:
            if label not in exclude_labels:
                continue
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            # Expand face bbox to cover mouth/tongue area (lower 40% of face)
            face_h = y2 - y1
            mouth_y1 = y1 + int(face_h * 0.6)  # Lower 40% = mouth region
            mask[mouth_y1:y2, x1:x2] = 255

        if np.any(mask):
            # Dilate to ensure full tongue coverage
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask = cv2.dilate(mask, kernel, iterations=1)
            return mask

        return None

    @staticmethod
    def _distance_gradient(binary_mask: np.ndarray, fade_pixels: int = 8) -> np.ndarray:
        """Apply distance transform to create smooth gradient edges following contour.

        The mask center stays fully opaque (255), edges fade smoothly to 0.
        This follows the actual contour shape, not a geometric primitive.

        Args:
            binary_mask: Binary mask (0/255)
            fade_pixels: Width of gradient fade zone in pixels
        """
        if np.sum(binary_mask > 0) == 0:
            return binary_mask

        # Distance from each pixel to nearest background
        dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)

        # Normalize: pixels deeper than fade_pixels from edge = 255 (solid)
        # Pixels within fade_pixels of edge = gradient
        result = np.clip(dist / max(fade_pixels, 1) * 255, 0, 255).astype(np.uint8)

        return result

    @staticmethod
    def _anus_gradient(
        bbox: tuple[int, int, int, int],
        img_h: int,
        img_w: int,
        scale: float = 0.30,
        solid_ratio: float = 0.4,
    ) -> np.ndarray:
        """Gradient elliptical mask for anus: solid center fading out."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        rx = (x2 - x1) * scale
        ry = (y2 - y1) * scale

        ys = np.arange(img_h)
        xs = np.arange(img_w)
        xx, yy = np.meshgrid(xs, ys)

        dist = np.sqrt(((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2)

        mask = np.zeros((img_h, img_w), dtype=np.float64)
        mask[dist <= solid_ratio] = 255.0
        gradient_zone = (dist > solid_ratio) & (dist <= 1.0)
        mask[gradient_zone] = 255.0 * (1.0 - (dist[gradient_zone] - solid_ratio) / (1.0 - solid_ratio))

        return mask.astype(np.uint8)

    @staticmethod
    def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
        """Keep only the largest connected component, discard fragments."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return mask
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        clean = np.zeros_like(mask)
        clean[labels == largest_label] = 255
        return clean

    def is_available(self) -> bool:
        try:
            from sam2.build_sam import build_sam2  # noqa: F401
        except ImportError:
            return False
        return Path(self.checkpoint).exists()


class GrabCutRefiner:
    """Fallback: OpenCV GrabCut for pixel-level masks (no model needed)."""

    def __init__(self, iterations: int = 5, margin: int = 5):
        self.iterations = iterations
        self.margin = margin

    def refine(
        self, image: Image.Image, detections: list[Detection]
    ) -> list[Detection]:
        if not detections:
            return detections

        img_array = np.array(image)
        h, w = img_array.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            bw, bh = x2 - x1, y2 - y1
            if bw < 10 or bh < 10:
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[y1:y2, x1:x2] = 255
                det.mask = mask
                continue

            m = self.margin
            rect = (x1 + m, y1 + m, max(1, bw - 2 * m), max(1, bh - 2 * m))

            gc_mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            try:
                cv2.grabCut(
                    img_array, gc_mask, rect,
                    bgd_model, fgd_model,
                    self.iterations, cv2.GC_INIT_WITH_RECT,
                )
                fg_mask = np.where(
                    (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                    255, 0,
                ).astype(np.uint8)

                result_mask = np.zeros((h, w), dtype=np.uint8)
                result_mask[y1:y2, x1:x2] = fg_mask[y1:y2, x1:x2]
                det.mask = result_mask
            except cv2.error as e:
                logger.warning(f"GrabCut failed: {e}, using bbox")
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[y1:y2, x1:x2] = 255
                det.mask = mask

        return detections

    def is_available(self) -> bool:
        return True


class MaskRefiner:
    """Post-process masks: erode, dilate, smooth."""

    def __init__(
        self,
        erode_pixels: int = 0,
        dilate_pixels: int = 0,
        blur_boundary: int = 0,
    ):
        self.erode_pixels = erode_pixels
        self.dilate_pixels = dilate_pixels
        self.blur_boundary = blur_boundary

    def refine(self, detections: list[Detection]) -> list[Detection]:
        for det in detections:
            if det.mask is None:
                continue

            mask = det.mask

            if self.erode_pixels > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (self.erode_pixels * 2 + 1, self.erode_pixels * 2 + 1),
                )
                mask = cv2.erode(mask, kernel, iterations=1)

            if self.dilate_pixels > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (self.dilate_pixels * 2 + 1, self.dilate_pixels * 2 + 1),
                )
                mask = cv2.dilate(mask, kernel, iterations=1)

            if self.blur_boundary > 0:
                ksize = self.blur_boundary * 2 + 1
                mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

            det.mask = mask

        return detections
