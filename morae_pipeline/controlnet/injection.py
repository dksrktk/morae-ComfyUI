"""ControlNet parameter injection into ComfyUI workflow templates."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

from ..config import ControlNetConfig
from .pose_library import PoseEntry, PoseLibrary

logger = logging.getLogger(__name__)

# Common ControlNet node class_types in ComfyUI
CONTROLNET_LOADER_TYPES = {
    "ControlNetLoader",
    "DiffControlNetLoader",
    "ControlNetLoaderAdvanced",
}
CONTROLNET_APPLY_TYPES = {
    "ControlNetApply",
    "ControlNetApplyAdvanced",
    "ACN_AdvancedControlNetApply",
}
IMAGE_LOADER_TYPES = {
    "LoadImage",
    "LoadImageFromPath",
}


@dataclass
class ControlNetParams:
    """Resolved ControlNet parameters for a single generation job."""

    pose_image_path: str
    strength: float
    end_percent: float
    model_name: str
    # For variation seed support
    variation_seed: int | None = None
    variation_strength: float = 0.0


class ControlNetInjector:
    """Detect and parameterize ControlNet nodes in workflow templates.

    Works with templates that already have ControlNet nodes set up.
    Parameterizes: pose image path, strength, end_percent, model name.
    """

    def __init__(self, config: ControlNetConfig):
        self.config = config
        self.pose_library: PoseLibrary | None = None

        if config.pose_library_path:
            lib_path = Path(config.pose_library_path)
            if lib_path.exists():
                self.pose_library = PoseLibrary(lib_path)
            else:
                logger.warning(f"Pose library not found: {lib_path}")

    def detect_controlnet_nodes(self, workflow: dict) -> dict[str, str]:
        """Detect ControlNet-related nodes in a workflow template.

        Returns:
            Mapping of role -> node_id:
            - "cn_loader": ControlNet model loader node
            - "cn_apply": ControlNet apply node
            - "cn_image": Image loader node for pose/depth input
        """
        node_map = {}

        for node_id, node in workflow.items():
            class_type = node.get("class_type", "")

            if class_type in CONTROLNET_LOADER_TYPES:
                node_map.setdefault("cn_loader", node_id)
            elif class_type in CONTROLNET_APPLY_TYPES:
                node_map.setdefault("cn_apply", node_id)
            elif class_type in IMAGE_LOADER_TYPES:
                # Heuristic: if this image loader connects to a ControlNet apply,
                # it's likely the pose/depth input
                inputs = node.get("inputs", {})
                image_name = inputs.get("image", "")
                # Check if node name suggests controlnet usage
                title = node.get("_meta", {}).get("title", "").lower()
                if any(
                    kw in title
                    for kw in ["pose", "depth", "canny", "controlnet", "control"]
                ):
                    node_map.setdefault("cn_image", node_id)
                elif "cn_apply" not in node_map:
                    # Will be resolved by connection tracing
                    node_map.setdefault("cn_image_candidate", node_id)

        # If we found cn_apply but no cn_image, try to trace the image input
        if "cn_apply" in node_map and "cn_image" not in node_map:
            apply_node = workflow[node_map["cn_apply"]]
            image_input = apply_node.get("inputs", {}).get("image")
            if isinstance(image_input, list) and len(image_input) == 2:
                source_id = str(image_input[0])
                if source_id in workflow:
                    source_type = workflow[source_id].get("class_type", "")
                    if source_type in IMAGE_LOADER_TYPES:
                        node_map["cn_image"] = source_id

            # Fallback to candidate
            if "cn_image" not in node_map and "cn_image_candidate" in node_map:
                node_map["cn_image"] = node_map.pop("cn_image_candidate")

        if "cn_image_candidate" in node_map:
            del node_map["cn_image_candidate"]

        if node_map:
            logger.debug(f"ControlNet nodes detected: {node_map}")
        return node_map

    def resolve_params(
        self,
        pose_category: str | None = None,
        pose_tags: list[str] | None = None,
        pose_path: str | None = None,
        seed: int | None = None,
    ) -> ControlNetParams | None:
        """Resolve ControlNet parameters for a single job.

        Applies randomization within configured ranges for strength and end_percent.

        Args:
            pose_category: Category to pick pose from
            pose_tags: Tags to filter poses
            pose_path: Explicit pose image path (overrides library selection)
            seed: Seed for deterministic randomization

        Returns:
            ControlNetParams or None if no pose available
        """
        rng = random.Random(seed) if seed is not None else random.Random()

        # Resolve pose image
        if pose_path:
            image_path = pose_path
            recommended_strength = self.config.strength_default
            recommended_end = self.config.end_percent_default
        elif self.pose_library:
            picks = self.pose_library.pick_random(
                category=pose_category, tags=pose_tags, count=1, rng=rng
            )
            if not picks:
                return None
            pose = picks[0]
            image_path = pose.path
            recommended_strength = pose.recommended_strength
            recommended_end = pose.recommended_end_percent
        else:
            return None

        # Apply randomization within ranges
        strength = self._jitter(
            base=recommended_strength,
            min_val=self.config.strength_min,
            max_val=self.config.strength_max,
            rng=rng,
        )
        end_percent = self._jitter(
            base=recommended_end,
            min_val=self.config.end_percent_min,
            max_val=self.config.end_percent_max,
            rng=rng,
        )

        # Variation seed for subtle output differences
        variation_seed = None
        variation_strength = 0.0
        if self.config.variation_enabled:
            variation_seed = rng.randint(0, 2**32 - 1)
            variation_strength = self.config.variation_strength

        return ControlNetParams(
            pose_image_path=image_path,
            strength=round(strength, 3),
            end_percent=round(end_percent, 3),
            model_name=self.config.model_name,
            variation_seed=variation_seed,
            variation_strength=variation_strength,
        )

    def apply_to_workflow(
        self, workflow: dict, params: ControlNetParams
    ) -> dict:
        """Inject ControlNet parameters into a workflow template.

        Args:
            workflow: Mutable workflow dict
            params: Resolved ControlNet parameters

        Returns:
            Modified workflow dict
        """
        node_map = self.detect_controlnet_nodes(workflow)

        if not node_map:
            logger.warning(
                "No ControlNet nodes found in workflow template. "
                "Ensure the template includes ControlNet nodes."
            )
            return workflow

        # Set pose/depth image
        if "cn_image" in node_map:
            image_node = workflow[node_map["cn_image"]]
            # For LoadImage, the input is "image" (filename in input dir)
            # For LoadImageFromPath, the input is "path" (full path)
            class_type = image_node.get("class_type", "")
            if class_type == "LoadImageFromPath":
                image_node["inputs"]["path"] = params.pose_image_path
            else:
                # LoadImage uses filename relative to ComfyUI input dir
                image_node["inputs"]["image"] = Path(
                    params.pose_image_path
                ).name

        # Set ControlNet model
        if "cn_loader" in node_map:
            loader = workflow[node_map["cn_loader"]]
            loader["inputs"]["control_net_name"] = params.model_name

        # Set strength and end_percent
        if "cn_apply" in node_map:
            apply_node = workflow[node_map["cn_apply"]]
            apply_node["inputs"]["strength"] = params.strength
            # ControlNetApplyAdvanced has end_percent
            if "end_percent" in apply_node["inputs"]:
                apply_node["inputs"]["end_percent"] = params.end_percent

        logger.debug(
            f"ControlNet injected: pose={Path(params.pose_image_path).name}, "
            f"strength={params.strength}, end={params.end_percent}"
        )
        return workflow

    @staticmethod
    def _jitter(
        base: float,
        min_val: float,
        max_val: float,
        rng: random.Random,
    ) -> float:
        """Add random variation to a base value within bounds."""
        lo = max(min_val, base - 0.1)
        hi = min(max_val, base + 0.1)
        return rng.uniform(lo, hi)
