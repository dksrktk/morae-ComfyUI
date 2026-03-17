"""Image filtering methods for censoring detected NSFW regions."""

from __future__ import annotations

import logging
from enum import Enum

import numpy as np
from PIL import Image, ImageFilter

from .detector import Detection

logger = logging.getLogger(__name__)


class FilterMethod(str, Enum):
    GAUSSIAN_BLUR = "gaussian_blur"
    PIXELATE = "pixelate"
    BLACK_BAR = "black_bar"
    WHITE_BAR = "white_bar"


class RegionFilter:
    """Apply censorship filters to detected NSFW regions."""

    def __init__(
        self,
        method: FilterMethod = FilterMethod.GAUSSIAN_BLUR,
        blur_radius: int = 40,
        pixelate_factor: int = 10,
        mask_padding: int = 10,
    ):
        self.method = method
        self.blur_radius = blur_radius
        self.pixelate_factor = pixelate_factor
        self.mask_padding = mask_padding

    def apply(
        self, image: Image.Image, detections: list[Detection]
    ) -> Image.Image:
        """Apply censorship filter to all detected regions.

        Args:
            image: Original PIL Image (RGB)
            detections: Detected NSFW regions

        Returns:
            New PIL Image with regions censored
        """
        if not detections:
            return image.copy()

        result = image.copy()

        for det in detections:
            if det.mask is not None:
                result = self._apply_with_mask(result, det)
            else:
                result = self._apply_with_bbox(result, det)

        logger.debug(
            f"Applied {self.method.value} filter to {len(detections)} regions"
        )
        return result

    def _pad_bbox(
        self, bbox: tuple[int, int, int, int], img_size: tuple[int, int]
    ) -> tuple[int, int, int, int]:
        """Add padding to bounding box, clamped to image bounds."""
        x1, y1, x2, y2 = bbox
        w, h = img_size
        pad = self.mask_padding
        return (
            max(0, x1 - pad),
            max(0, y1 - pad),
            min(w, x2 + pad),
            min(h, y2 + pad),
        )

    def _apply_with_bbox(
        self, image: Image.Image, det: Detection
    ) -> Image.Image:
        """Apply filter using bounding box only."""
        x1, y1, x2, y2 = self._pad_bbox(det.bbox, image.size)
        region = image.crop((x1, y1, x2, y2))

        filtered = self._filter_region(region)
        image.paste(filtered, (x1, y1))
        return image

    def _apply_with_mask(
        self, image: Image.Image, det: Detection
    ) -> Image.Image:
        """Apply filter using pixel-level mask for precise censorship."""
        # Create full-image filter then composite using mask
        x1, y1, x2, y2 = self._pad_bbox(det.bbox, image.size)
        region = image.crop((x1, y1, x2, y2))

        filtered = self._filter_region(region)

        # Crop mask to same region
        mask_region = det.mask[y1:y2, x1:x2]
        mask_pil = Image.fromarray(mask_region, mode="L")

        # Composite: filtered where mask is white, original where black
        image.paste(filtered, (x1, y1), mask=mask_pil)
        return image

    def _filter_region(self, region: Image.Image) -> Image.Image:
        """Apply the chosen filter to a cropped region."""
        if self.method == FilterMethod.GAUSSIAN_BLUR:
            return region.filter(
                ImageFilter.GaussianBlur(radius=self.blur_radius)
            )
        elif self.method == FilterMethod.PIXELATE:
            w, h = region.size
            small_w = max(1, w // self.pixelate_factor)
            small_h = max(1, h // self.pixelate_factor)
            small = region.resize((small_w, small_h), Image.NEAREST)
            return small.resize((w, h), Image.NEAREST)
        elif self.method == FilterMethod.BLACK_BAR:
            return Image.new("RGB", region.size, (0, 0, 0))
        elif self.method == FilterMethod.WHITE_BAR:
            return Image.new("RGB", region.size, (255, 255, 255))
        else:
            return region
