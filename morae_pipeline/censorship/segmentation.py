"""Pixel-level segmentation refinement for precise censorship masking.

Pipeline:
  1. SAM2 mask (pixel-precise contour)
  2. Semantic subtraction (remove face/tongue from mask)
  3. AutoCensor-style refinement (supersample + contour smoothing + AA)
  4. Anus: elliptical gradient (SAM2 unreliable for anus)
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image

from .detector import Detection

logger = logging.getLogger(__name__)


def smooth_contour(cnt: np.ndarray, smooth_factor: float = 2.0) -> np.ndarray:
    """Smooth contour points using B-spline interpolation (AutoCensor algorithm).

    Args:
        cnt: Contour points array (N, 1, 2) or (N, 2)
        smooth_factor: Controls output point density (higher = more points)

    Returns:
        Smoothed contour points (N, 1, 2) format for cv2.drawContours
    """
    from scipy.interpolate import splprep, splev

    # Reshape to (N, 2) if needed
    pts = cnt.reshape(-1, 2)
    n = len(pts)

    if n < 10:
        return cnt

    # Remove duplicate consecutive points
    mask = np.ones(n, dtype=bool)
    mask[1:] = np.any(np.diff(pts, axis=0) != 0, axis=1)
    pts = pts[mask]
    n = len(pts)

    if n < 10:
        return cnt

    try:
        # Convert to complex for spline fitting
        pts_c = np.array([pts[:, 0], pts[:, 1]])

        # Fit periodic cubic B-spline
        # s = smoothing factor, per=True for closed curve, k=3 for cubic
        tck, u = splprep(
            [pts_c[0, :], pts_c[1, :]],
            s=n * 2.0,  # smoothing amount
            per=True,   # periodic (closed curve)
            k=3,        # cubic spline
        )

        # Generate new points
        num_points = max(int(n * smooth_factor), 200)
        u_new = np.linspace(0, 1, num_points, endpoint=False)
        x_new, y_new = splev(u_new, tck)

        # Stack and reshape for cv2.drawContours format
        smooth = np.column_stack([x_new, y_new]).astype(np.float64)
        return smooth.reshape(-1, 1, 2).astype(np.int32)

    except Exception:
        # Fallback: cv2.approxPolyDP
        try:
            epsilon = 0.002 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) >= 3:
                return approx
        except Exception:
            pass
        return cnt


def refine_mask_autocensor_style(
    mask_np: np.ndarray,
    target_size: tuple[int, int],
    blur_sigma: float = 2.0,
    supersample: int = 4,
) -> np.ndarray:
    """Refine mask using AutoCensor algorithm: supersample + B-spline smoothing + AA.

    Pipeline (reverse-engineered from AutoCensor.exe):
      1. Normalize mask to uint8
      2. Resize to target size if needed
      3. Threshold to binary
      4. Extract contours with hierarchy
      5. Smooth each contour using B-spline (smooth_factor = supersample)
      6. Scale contours to supersample resolution
      7. Draw on supersample canvas with anti-aliasing (LINE_AA)
      8. Downsample with INTER_AREA
      9. Edge-band selective blur (only blur dilate-erode transition zone)

    Args:
        mask_np: Raw mask from SAM2 (0-1 float or 0-255 uint8)
        target_size: (width, height) of output
        blur_sigma: Sigma for edge blur (0 = no blur)
        supersample: Supersampling factor (2-4 recommended)

    Returns:
        Refined mask with smooth edges (0-255 uint8)
    """
    w, h = target_size
    ss = max(1, supersample)

    # Normalize to 0-255 uint8
    if mask_np.max() <= 1.0:
        m8 = (mask_np * 255).astype(np.uint8)
    else:
        m8 = mask_np.astype(np.uint8)

    mh, mw = m8.shape[:2]

    # Resize to target if different
    if mw != w or mh != h:
        m8 = cv2.resize(m8, (w, h), interpolation=cv2.INTER_CUBIC)

    # Binarize
    _, binary = cv2.threshold(m8, 127, 255, cv2.THRESH_BINARY)

    # Extract contours with hierarchy (RETR_CCOMP for holes)
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return m8

    # Smooth contours using B-spline (smooth_factor = 5, same as AutoCensor)
    smoothed = [smooth_contour(c, smooth_factor=5.0) for c in contours]

    # Supersample dimensions
    ss_w, ss_h = w * ss, h * ss

    # Scale contours to supersample resolution
    scaled = []
    for c in smoothed:
        sc = c.reshape(-1, 2).astype(np.float64) * ss
        sc = sc.reshape(-1, 1, 2).astype(np.int32)
        scaled.append(sc)

    # Draw on supersample canvas with anti-aliasing
    canvas = np.zeros((ss_h, ss_w), dtype=np.uint8)

    if hierarchy is not None and len(scaled) > 0:
        h_arr = hierarchy[0]
        for i, c in enumerate(scaled):
            if len(c) < 3:
                continue
            # hierarchy: [next, prev, child, parent]
            # parent == -1 means outer contour (fill white)
            # parent != -1 means hole (fill black)
            if h_arr[i][3] == -1:
                cv2.drawContours(canvas, [c], -1, 255, cv2.FILLED, lineType=cv2.LINE_AA)
            else:
                cv2.drawContours(canvas, [c], -1, 0, cv2.FILLED, lineType=cv2.LINE_AA)
    else:
        for c in scaled:
            if len(c) >= 3:
                cv2.drawContours(canvas, [c], -1, 255, cv2.FILLED, lineType=cv2.LINE_AA)

    # AutoCensor: draw 1px outline on top for smoother edges
    for c in scaled:
        if len(c) >= 3:
            cv2.drawContours(canvas, [c], -1, 255, 1, lineType=cv2.LINE_AA)

    # Downsample with INTER_AREA (smooth anti-aliased)
    if ss > 1:
        result_u8 = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)
    else:
        result_u8 = canvas

    result = result_u8.astype(np.float32) / 255.0

    # Edge-band selective blur (AutoCensor algorithm)
    if blur_sigma > 0:
        # Convert to uint8 for morphology
        m8_ds = (result * 255).astype(np.uint8)
        _, bin_ds = cv2.threshold(m8_ds, 127, 255, cv2.THRESH_BINARY)

        # Kernel size: odd number >= 3
        k = max(3, int(blur_sigma * 2) | 1)
        kern = np.ones((k, k), dtype=np.uint8)

        # Create edge band: dilated & ~eroded
        dilated = cv2.dilate(bin_ds, kern)
        eroded = cv2.erode(bin_ds, kern)
        edge_band = ((dilated > 0) & (eroded == 0)).astype(np.float32)

        # Gaussian blur the entire result
        blurred = gaussian_filter(result, sigma=blur_sigma)
        blurred = np.clip(blurred, 0.0, 1.0)

        # Blend: original where not edge, blurred where edge
        result = result * (1.0 - edge_band) + blurred * edge_band

    return (np.clip(result, 0.0, 1.0) * 255).astype(np.uint8)


def refine_mask_autocensor_v2(
    mask_np: np.ndarray,
    target_size: tuple[int, int],
    blur_sigma: float = 2.0,
    supersample: int = 2,
    brush_hardness: float = 0.5,
) -> np.ndarray:
    """Refine mask using AutoCensor Release GPU/CPU algorithm (newer, simpler).

    Pipeline (reverse-engineered from AutoCensor_Release_GPU_CPU.exe):
      1. Normalize mask to float32 (0-1)
      2. Resize to target size if needed
      3. Binary threshold at 0.5 -> core_mask
      4. Calculate feather_px from brush_hardness and blur_sigma
      5. Dilate + GaussianBlur for soft edges
      6. Blend soft edges with core mask

    This version is simpler but faster than v1 (no B-spline smoothing).

    Args:
        mask_np: Raw mask from SAM2 (0-1 float or 0-255 uint8)
        target_size: (width, height) of output
        blur_sigma: Sigma for edge blur (0 = no blur)
        supersample: Supersampling factor (affects feather calculation)
        brush_hardness: Edge hardness 0.0 (soft) to 1.0 (hard)

    Returns:
        Refined mask with smooth edges (0-255 uint8)
    """
    w, h = target_size
    ss = max(1, supersample)

    # Normalize to float32 0-1
    m = mask_np.astype(np.float32)
    mx = float(m.max()) if m.size > 0 else 0.0
    if mx > 1.0:
        m = m / 255.0

    mh, mw = m.shape[:2]

    # Resize to target if different
    if mw != w or mh != h:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)

    # Core mask: binary threshold at 0.5
    core_mask = (m > 0.5).astype(np.float32)

    # Normalize brush_hardness to 0-1
    hardness = float(np.clip(brush_hardness, 0.0, 1.0))

    # Calculate feather pixels
    feather_px = 0.0
    if hardness < 0.999:
        feather_px += (1.0 - hardness) * (6.0 + max(0, ss - 1))
    if blur_sigma > 0:
        feather_px += max(0.0, float(blur_sigma)) * 1.35

    result = core_mask.copy()

    # Apply soft edge if feather_px > 0
    if feather_px > 0 and np.any(core_mask > 0):
        # Binary for morphology
        binary = (core_mask * 255).astype(np.uint8)

        # Kernel size: odd number >= 3
        k = max(3, int(round(feather_px * 2.0)) | 1)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        # Dilate to expand
        expanded = cv2.dilate(binary, kern, iterations=1)

        # Gaussian blur sigma
        sigma = max(0.6, feather_px * 0.55)

        # GaussianBlur for soft edges
        soft = cv2.GaussianBlur(
            expanded.astype(np.float32) / 255.0,
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
        )

        # Use soft where expanded, otherwise core
        result = np.where(expanded > 0, soft, core_mask)

    return (np.clip(result, 0.0, 1.0) * 255).astype(np.uint8)


def refine_mask(
    mask_np: np.ndarray,
    target_size: tuple[int, int],
    blur_sigma: float = 2.0,
    supersample: int = 2,
    algorithm: str = "v1",
    brush_hardness: float = 0.5,
) -> np.ndarray:
    """Unified mask refinement with algorithm selection.

    Args:
        mask_np: Raw mask (0-1 float or 0-255 uint8)
        target_size: (width, height) of output
        blur_sigma: Sigma for edge blur
        supersample: Supersampling factor
        algorithm: "v1" (B-spline + supersample AA) or "v2" (dilate + blur)
        brush_hardness: Edge hardness for v2 (0.0-1.0)

    Returns:
        Refined mask (0-255 uint8)
    """
    if algorithm == "v2":
        return refine_mask_autocensor_v2(
            mask_np, target_size, blur_sigma, supersample, brush_hardness
        )
    else:
        return refine_mask_autocensor_style(
            mask_np, target_size, blur_sigma, supersample
        )


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
            # Add center point as foreground hint for better accuracy
            center_point = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]])
            point_label = np.array([1])  # 1 = foreground
            masks, scores, _ = self._predictor.predict(
                point_coords=center_point,
                point_labels=point_label,
                box=input_box,
                multimask_output=True,
            )
            bbox_area = (x2 - x1) * (y2 - y1)
            best_idx = self._pick_best_mask(masks, scores, bbox_area, det_bbox=(x1, y1, x2, y2))
            raw_mask = (masks[best_idx] * 255).astype(np.uint8)

            # Step 2: Keep largest connected component
            raw_mask = self._keep_largest_component(raw_mask)

            # Step 2.5: Fill internal holes with morphological closing
            # Genitalia is one connected body - no holes inside
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)

            # Step 3: AutoCensor-style refinement (supersample + contour smooth + AA)
            refined_mask = refine_mask_autocensor_style(
                raw_mask,
                target_size=(w, h),
                blur_sigma=2.0,
                supersample=4,
            )

            det.mask = refined_mask

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
    def _pick_best_mask(
        masks: np.ndarray,
        scores: np.ndarray,
        bbox_area: int,
        det_bbox: tuple[int, int, int, int] | None = None,
    ) -> int:
        """Pick best mask: highest score among those fully contained within detection bbox.

        Bbox is always larger than the actual target region.
        SAM2 must segment WITHIN the bbox, not outside it.
        Any mask extending beyond bbox is rejected (0% tolerance).
        """
        best_idx = -1
        best_score = -1.0

        for i, (m, s) in enumerate(zip(masks, scores)):
            ys, xs = np.where(m > 0.5)
            if len(ys) == 0:
                continue

            if det_bbox is not None:
                bx1, by1, bx2, by2 = det_bbox
                # Strict containment: mask must be fully within bbox (0% tolerance)
                if (xs.min() < bx1 or xs.max() > bx2 or
                        ys.min() < by1 or ys.max() > by2):
                    continue  # Mask extends outside bbox - reject

            if s > best_score:
                best_score = s
                best_idx = i

        # Fallback: pick smallest area (least likely to overflow)
        if best_idx == -1:
            areas = [int(np.sum(m > 0.5)) for m in masks]
            best_idx = int(np.argmin(areas))

        return best_idx

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
