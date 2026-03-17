"""IP-Adapter and LoRA injection into ComfyUI workflow templates."""

from __future__ import annotations

import logging
from pathlib import Path

from .profile import CharacterProfile

logger = logging.getLogger(__name__)

# IP-Adapter node class_types
IPADAPTER_LOADER_TYPES = {
    "IPAdapterModelLoader",
    "IPAdapterUnifiedLoader",
    "IPAdapterUnifiedLoaderFaceID",
}
IPADAPTER_APPLY_TYPES = {
    "IPAdapterApply",
    "IPAdapterApplyFaceID",
    "IPAdapterAdvanced",
    "IPAdapterFaceID",
    "IPAdapter",
}
CLIP_VISION_LOADER_TYPES = {
    "CLIPVisionLoader",
}
LORA_LOADER_TYPES = {
    "LoraLoader",
    "LoraLoaderModelOnly",
}
IMAGE_LOADER_TYPES = {
    "LoadImage",
    "LoadImageFromPath",
}


class CharacterInjector:
    """Inject character profile data (IP-Adapter, LoRA, tags) into workflows.

    Works with templates that already have IP-Adapter and/or LoRA nodes.
    Auto-detects nodes and sets reference images, weights, and model paths.
    """

    def detect_nodes(self, workflow: dict) -> dict[str, list[str]]:
        """Detect character-related nodes in a workflow.

        Returns:
            Mapping of role -> list of node_ids:
            - "ipa_loader": IP-Adapter model loader nodes
            - "ipa_apply": IP-Adapter apply nodes
            - "ipa_image": Image loader for reference face
            - "clip_vision": CLIP Vision model loader
            - "lora_loader": LoRA loader nodes
        """
        node_map: dict[str, list[str]] = {
            "ipa_loader": [],
            "ipa_apply": [],
            "ipa_image": [],
            "clip_vision": [],
            "lora_loader": [],
        }

        for node_id, node in workflow.items():
            class_type = node.get("class_type", "")

            if class_type in IPADAPTER_LOADER_TYPES:
                node_map["ipa_loader"].append(node_id)
            elif class_type in IPADAPTER_APPLY_TYPES:
                node_map["ipa_apply"].append(node_id)
            elif class_type in CLIP_VISION_LOADER_TYPES:
                node_map["clip_vision"].append(node_id)
            elif class_type in LORA_LOADER_TYPES:
                node_map["lora_loader"].append(node_id)
            elif class_type in IMAGE_LOADER_TYPES:
                # Check if this image loader connects to IP-Adapter
                title = node.get("_meta", {}).get("title", "").lower()
                if any(
                    kw in title
                    for kw in ["reference", "face", "ipadapter", "ip-adapter", "character"]
                ):
                    node_map["ipa_image"].append(node_id)

        # Trace connections: find image nodes feeding into IP-Adapter apply
        for apply_id in node_map["ipa_apply"]:
            apply_node = workflow[apply_id]
            image_input = apply_node.get("inputs", {}).get("image")
            if isinstance(image_input, list) and len(image_input) == 2:
                source_id = str(image_input[0])
                if (
                    source_id in workflow
                    and workflow[source_id].get("class_type", "") in IMAGE_LOADER_TYPES
                    and source_id not in node_map["ipa_image"]
                ):
                    node_map["ipa_image"].append(source_id)

        # Clean up empty lists
        node_map = {k: v for k, v in node_map.items() if v}

        if node_map:
            summary = {k: len(v) for k, v in node_map.items()}
            logger.debug(f"Character nodes detected: {summary}")
        return node_map

    def apply_profile(
        self, workflow: dict, profile: CharacterProfile
    ) -> dict:
        """Inject character profile into workflow template.

        Sets:
        - IP-Adapter reference image and weight
        - LoRA model path and weights
        - Prepends character tags to positive prompt
        - Appends negative tags to negative prompt

        Args:
            workflow: Mutable workflow dict
            profile: Character profile to apply

        Returns:
            Modified workflow dict
        """
        node_map = self.detect_nodes(workflow)

        # 1. IP-Adapter: set reference face image and weight
        if profile.reference_images and node_map.get("ipa_image"):
            ref_image = profile.primary_reference
            for img_node_id in node_map["ipa_image"]:
                img_node = workflow[img_node_id]
                class_type = img_node.get("class_type", "")
                if class_type == "LoadImageFromPath":
                    img_node["inputs"]["path"] = ref_image
                else:
                    img_node["inputs"]["image"] = Path(ref_image).name
            logger.debug(f"Set reference image: {Path(ref_image).name}")

        if node_map.get("ipa_apply"):
            for apply_id in node_map["ipa_apply"]:
                apply_node = workflow[apply_id]
                inputs = apply_node["inputs"]
                inputs["weight"] = profile.ip_adapter_weight
                if "weight_type" in inputs:
                    inputs["weight_type"] = profile.ip_adapter_weight_type
            logger.debug(f"Set IP-Adapter weight: {profile.ip_adapter_weight}")

        # 2. LoRA: set model path and weights
        if profile.loras and node_map.get("lora_loader"):
            for i, lora in enumerate(profile.loras):
                if i >= len(node_map["lora_loader"]):
                    logger.warning(
                        f"More LoRAs ({len(profile.loras)}) than loader nodes "
                        f"({len(node_map['lora_loader'])}), skipping extra"
                    )
                    break
                loader_id = node_map["lora_loader"][i]
                loader = workflow[loader_id]
                loader["inputs"]["lora_name"] = Path(lora.path).name
                loader["inputs"]["strength_model"] = lora.weight
                loader["inputs"]["strength_clip"] = lora.clip_weight
            logger.debug(
                f"Set {min(len(profile.loras), len(node_map.get('lora_loader', [])))} LoRA(s)"
            )

        # 3. Tags: prepend to positive prompt, append to negative
        self._inject_tags(workflow, profile)

        logger.info(f"Character '{profile.name}' applied to workflow")
        return workflow

    def _inject_tags(self, workflow: dict, profile: CharacterProfile) -> None:
        """Prepend character tags to prompts."""
        if not profile.tags and not profile.negative_tags:
            return

        for node_id, node in workflow.items():
            if node.get("class_type") != "CLIPTextEncode":
                continue

            text = node["inputs"].get("text", "")
            text_lower = text.lower()

            # Detect positive vs negative prompt
            is_negative = any(
                neg in text_lower
                for neg in ["bad", "worst", "low quality", "ugly", "negative"]
            )

            if is_negative and profile.negative_tags:
                # Append negative tags
                extra = profile.negative_tag_string
                if extra not in text:
                    node["inputs"]["text"] = f"{text}, {extra}" if text else extra
            elif not is_negative and profile.tags:
                # Prepend character tags
                extra = profile.tag_string
                if extra not in text:
                    node["inputs"]["text"] = f"{extra}, {text}" if text else extra

                # Also prepend LoRA trigger words
                for lora in profile.loras:
                    if lora.trigger_word and lora.trigger_word not in node["inputs"]["text"]:
                        node["inputs"]["text"] = (
                            f"{lora.trigger_word}, {node['inputs']['text']}"
                        )
