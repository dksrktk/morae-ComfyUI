"""NSFW detection using YOLO Segmentation (AutoCensor style)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single detected NSFW region."""
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    mask: np.ndarray | None = None


# AutoCensor LABEL_GROUPS
LABEL_GROUPS = {
    "anus": ['anus', 'anal', 'ass', 'asshole', 'exposed_anus', 'buttocks'],
    "genital": ['penis', 'exposed_penis', 'genitalia', 'genitals', 'vulva',
                'vagina', 'pussy', 'exposed_vulva', 'testicles', 'cunnus', 'female_genital'],
    "breast": ['nipple', 'nipples', 'exposed_nipple', 'breast', 'exposed_breast']
}


class YOLOSegmentationDetector:
    """YOLO Segmentation detector - AutoCensor style, outputs masks directly."""

    def __init__(
        self,
        model_path: str = "models/yolo/ntd11_anime_nsfw_segm_v5-variant1.pt",
        confidence_threshold: float = 0.25,
        device: str = "cuda",
        censor_anus: bool = True,
        censor_genital: bool = True,
        censor_breast: bool = False,
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model = None

        # Build targets list (AutoCensor style)
        self.targets = []
        if censor_anus:
            self.targets.extend(LABEL_GROUPS["anus"])
        if censor_genital:
            self.targets.extend(LABEL_GROUPS["genital"])
        if censor_breast:
            self.targets.extend(LABEL_GROUPS["breast"])

    def _load_model(self):
        if self._model is not None:
            return
        from ultralytics import YOLO
        self._model = YOLO(str(self.model_path))
        logger.info(f"YOLO Segmentation loaded: {self.model_path}")

    def detect(self, image: Image.Image) -> list[Detection]:
        """Detect NSFW regions - returns detections with masks (AutoCensor style)."""
        self._load_model()

        w, h = image.size

        # AutoCensor: model.predict with retina_masks=True
        # YOLO expects BGR (OpenCV style), not RGB
        img_rgb = np.array(image)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        results = self._model.predict(
            img_bgr,
            conf=self.confidence_threshold,
            device=self.device,
            imgsz=1280,
            retina_masks=True,
            verbose=False,
        )

        detections = []

        for r in results:
            if r.masks is None:
                continue

            # AutoCensor: zip(r.masks.data, r.boxes)
            for m_tensor, b in zip(r.masks.data, r.boxes):
                cls_name = self._model.names[int(b.cls[0])]

                # AutoCensor: if model.names[int(b.cls[0])] in targets
                if cls_name not in self.targets:
                    continue

                conf = float(b.conf[0])
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                bbox = (int(x1), int(y1), int(x2), int(y2))

                # AutoCensor: mask_raw = m_tensor.cpu().numpy()
                mask_raw = m_tensor.cpu().numpy()

                # Resize to image size if needed
                if mask_raw.shape[0] != h or mask_raw.shape[1] != w:
                    mask_raw = cv2.resize(mask_raw, (w, h), interpolation=cv2.INTER_LINEAR)

                # Convert to uint8
                mask_u8 = (mask_raw * 255).astype(np.uint8)

                detections.append(Detection(
                    class_name=cls_name,
                    confidence=conf,
                    bbox=bbox,
                    mask=mask_u8,
                ))

        logger.debug(f"YOLO: {len(detections)} detections ({', '.join(d.class_name for d in detections)})")
        return detections

    def is_available(self) -> bool:
        if not self.model_path.exists():
            return False
        try:
            from ultralytics import YOLO
            return True
        except ImportError:
            return False


# Legacy fallback (imgutils + NudeNet)
class CensorDetector:
    """Fallback detector using imgutils + NudeNet (bbox only, needs SAM2)."""

    def __init__(
        self,
        confidence_threshold: float = 0.25,
        anus_confidence_threshold: float = 0.2,
        bbox_expand_ratio: float = 0.15,
    ):
        self.confidence_threshold = confidence_threshold
        self.anus_confidence_threshold = anus_confidence_threshold
        self.bbox_expand_ratio = bbox_expand_ratio

    def detect(self, image: Image.Image) -> list[Detection]:
        detections = []
        detections.extend(self._detect_censors(image))
        detections.extend(self._detect_anus(image))
        detections = self._deduplicate(detections)
        return detections

    def _detect_censors(self, image: Image.Image) -> list[Detection]:
        from imgutils.detect.censor import detect_censors
        results = detect_censors(image, conf_threshold=self.confidence_threshold, iou_threshold=0.7)
        target = {"penis", "pussy"}
        detections = []
        w, h = image.size
        for bbox, label, confidence in results:
            if label not in target:
                continue
            x1, y1, x2, y2 = bbox
            if label == "pussy":
                x1, y1, x2, y2 = self._expand_bbox(x1, y1, x2, y2, w, h, self.bbox_expand_ratio)
            detections.append(Detection(
                class_name=label,
                confidence=float(confidence),
                bbox=(int(x1), int(y1), int(x2), int(y2)),
            ))
        return detections

    def _detect_anus(self, image: Image.Image) -> list[Detection]:
        from imgutils.detect.nudenet import detect_with_nudenet
        results = detect_with_nudenet(image, topk=100, iou_threshold=0.45, score_threshold=self.anus_confidence_threshold)
        detections = []
        w, h = image.size
        for bbox, label, confidence in results:
            if label != "ANUS_EXPOSED":
                continue
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = self._expand_bbox(x1, y1, x2, y2, w, h, self.bbox_expand_ratio)
            detections.append(Detection(
                class_name="anus",
                confidence=float(confidence),
                bbox=(int(x1), int(y1), int(x2), int(y2)),
            ))
        return detections

    def _expand_bbox(self, x1, y1, x2, y2, img_w, img_h, ratio):
        bw, bh = x2 - x1, y2 - y1
        dx, dy = bw * ratio, bh * ratio
        return max(0, x1 - dx), max(0, y1 - dy), min(img_w, x2 + dx), min(img_h, y2 + dy)

    def _deduplicate(self, detections, iou_threshold=0.5):
        if len(detections) <= 1:
            return detections
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
    def _iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def is_available(self):
        try:
            from imgutils.detect.censor import detect_censors
            from imgutils.detect.nudenet import detect_with_nudenet
            return True
        except ImportError:
            return False
