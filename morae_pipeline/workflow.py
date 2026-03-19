"""Workflow template loading and parameterization."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import GenerationConfig


class WorkflowTemplate:
    """Load an API-format ComfyUI workflow and parameterize it."""

    def __init__(self, data: dict):
        self._original = data
        self._data = copy.deepcopy(data)
        self._node_map: dict[str, str] = {}  # role -> node_id mapping
        self._auto_detect_nodes()

    @classmethod
    def from_file(cls, path: Path) -> WorkflowTemplate:
        with open(path) as f:
            data = json.load(f)
        return cls(data)

    @classmethod
    def from_dict(cls, data: dict) -> WorkflowTemplate:
        return cls(data)

    def _auto_detect_nodes(self) -> None:
        """Auto-detect common node roles by class_type."""
        for node_id, node in self._data.items():
            class_type = node.get("class_type", "")
            inputs = node.get("inputs", {})

            if class_type == "KSampler":
                self._node_map["sampler"] = node_id
            elif class_type == "CheckpointLoaderSimple":
                self._node_map["checkpoint"] = node_id
            elif class_type == "EmptyLatentImage":
                self._node_map["latent"] = node_id
            elif class_type == "SaveImage":
                self._node_map["save"] = node_id
            elif class_type == "CLIPTextEncode":
                # Heuristic: positive prompt usually has non-negative text
                text = inputs.get("text", "").lower()
                if any(neg in text for neg in ["bad", "worst", "low quality", "ugly", "negative"]):
                    self._node_map.setdefault("negative", node_id)
                else:
                    self._node_map.setdefault("positive", node_id)

    def set_param(self, node_id: str, field: str, value: Any) -> WorkflowTemplate:
        """Set a specific input parameter on a node."""
        if node_id not in self._data:
            raise KeyError(f"Node '{node_id}' not found in workflow")
        self._data[node_id]["inputs"][field] = value
        return self

    def set_prompt(self, positive: str, negative: Optional[str] = None) -> WorkflowTemplate:
        """Set text prompts using auto-detected CLIPTextEncode nodes."""
        if "positive" in self._node_map:
            self._data[self._node_map["positive"]]["inputs"]["text"] = positive
        else:
            raise KeyError("No positive CLIPTextEncode node detected")

        if negative is not None and "negative" in self._node_map:
            self._data[self._node_map["negative"]]["inputs"]["text"] = negative

        return self

    def set_seed(self, seed: int) -> WorkflowTemplate:
        if "sampler" in self._node_map:
            self._data[self._node_map["sampler"]]["inputs"]["seed"] = seed
        else:
            raise KeyError("No KSampler node detected")
        return self

    def set_dimensions(self, width: int, height: int) -> WorkflowTemplate:
        if "latent" in self._node_map:
            self._data[self._node_map["latent"]]["inputs"]["width"] = width
            self._data[self._node_map["latent"]]["inputs"]["height"] = height
        else:
            raise KeyError("No EmptyLatentImage node detected")
        return self

    def set_filename_prefix(self, prefix: str) -> WorkflowTemplate:
        if "save" in self._node_map:
            self._data[self._node_map["save"]]["inputs"]["filename_prefix"] = prefix
        return self

    def set_checkpoint(self, ckpt_name: str) -> WorkflowTemplate:
        if "checkpoint" in self._node_map:
            self._data[self._node_map["checkpoint"]]["inputs"]["ckpt_name"] = ckpt_name
        else:
            raise KeyError("No CheckpointLoaderSimple node detected")
        return self

    @property
    def node_map(self) -> dict[str, str]:
        """Return detected node role -> node_id mapping."""
        return dict(self._node_map)

    def reset(self) -> WorkflowTemplate:
        """Reset to original loaded state."""
        self._data = copy.deepcopy(self._original)
        return self

    def to_dict(self) -> dict:
        """Return the parameterized workflow as a dict ready for API submission."""
        return copy.deepcopy(self._data)

    def validate(self) -> list[str]:
        """Basic validation. Returns list of warnings."""
        warnings = []
        if not self._data:
            warnings.append("Workflow is empty")
        if "sampler" not in self._node_map:
            warnings.append("No KSampler node found")
        if "save" not in self._node_map:
            warnings.append("No SaveImage node found")
        if "positive" not in self._node_map:
            warnings.append("No positive prompt node detected")
        return warnings


class WorkflowBuilder:
    """동적 ComfyUI 워크플로우 빌더 (generate.py 기반)."""

    def __init__(
        self,
        config: "GenerationConfig",
        ipadapter_config: Optional[Any] = None,
        controlnet_config: Optional[Any] = None,
    ):
        self.config = config
        self.ipadapter_config = ipadapter_config
        self.controlnet_config = controlnet_config

    def build(
        self,
        positive_prompt: str,
        negative_prompt: str,
        seed: Optional[int] = None,
        filename_prefix: str = "gen",
        character_loras: Optional[list[dict]] = None,
        reference_image: Optional[str] = None,
        controlnet_image: Optional[str] = None,
        controlnet_extract: bool = False,
    ) -> tuple[dict, int]:
        """ComfyUI API 워크플로우 생성.

        Args:
            positive_prompt: 포지티브 프롬프트
            negative_prompt: 네거티브 프롬프트
            seed: 시드 (None이면 랜덤)
            filename_prefix: 파일명 프리픽스
            character_loras: 캐릭터 LoRA 목록 [{"path": ..., "weight": ..., "trigger_word": ...}]
            reference_image: IP-Adapter용 레퍼런스 이미지 경로
            controlnet_image: ControlNet 포즈 이미지 경로 (라이브러리 또는 추출용)
            controlnet_extract: True면 DWPose로 포즈 추출, False면 이미지 직접 사용

        Returns:
            (workflow_dict, seed)
        """
        if seed is None:
            seed = random.randint(1, 2**32 - 1)

        cfg = self.config
        workflow = {}

        # 1. Checkpoint Loader
        workflow["1"] = {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": cfg.model},
        }

        # 2. CLIP Set Last Layer
        workflow["2"] = {
            "class_type": "CLIPSetLastLayer",
            "inputs": {"clip": ["1", 1], "stop_at_clip_layer": -2},
        }

        current_model = ["1", 0]
        current_clip = ["2", 0]
        next_node_id = 15

        # 캐릭터 LoRA (우선 적용)
        if character_loras:
            for i, lora in enumerate(character_loras):
                node_id = str(next_node_id + i)
                workflow[node_id] = {
                    "class_type": "LoraLoader",
                    "inputs": {
                        "model": current_model,
                        "clip": current_clip,
                        "lora_name": lora["path"],
                        "strength_model": lora.get("weight", 0.8),
                        "strength_clip": lora.get("clip_weight", lora.get("weight", 0.8)),
                    },
                }
                current_model = [node_id, 0]
                current_clip = [node_id, 1]
            next_node_id += len(character_loras)

        # 15. LoRA 1 (config 기본 LoRA)
        if cfg.lora1:
            workflow["15"] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": current_model,
                    "clip": current_clip,
                    "lora_name": cfg.lora1,
                    "strength_model": cfg.lora1_strength,
                    "strength_clip": cfg.lora1_strength,
                },
            }
            current_model = ["15", 0]
            current_clip = ["15", 1]

        # 16. LoRA 2
        if cfg.lora2:
            workflow["16"] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": current_model,
                    "clip": current_clip,
                    "lora_name": cfg.lora2,
                    "strength_model": cfg.lora2_strength,
                    "strength_clip": cfg.lora2_strength,
                },
            }
            current_model = ["16", 0]
            current_clip = ["16", 1]

        # IP-Adapter (레퍼런스 이미지 스타일 전달)
        if reference_image and self.ipadapter_config:
            ipa_cfg = self.ipadapter_config

            # 50. Load Image (reference)
            workflow["50"] = {
                "class_type": "LoadImage",
                "inputs": {"image": reference_image},
            }

            # 51. Load CLIP Vision
            workflow["51"] = {
                "class_type": "CLIPVisionLoader",
                "inputs": {"clip_name": ipa_cfg.clip_vision_model},
            }

            # 52. IPAdapter Model Loader
            workflow["52"] = {
                "class_type": "IPAdapterModelLoader",
                "inputs": {"ipadapter_file": ipa_cfg.model},
            }

            # 53. IPAdapter Advanced
            workflow["53"] = {
                "class_type": "IPAdapterAdvanced",
                "inputs": {
                    "model": current_model,
                    "ipadapter": ["52", 0],
                    "clip_vision": ["51", 0],
                    "image": ["50", 0],
                    "weight": ipa_cfg.weight,
                    "weight_type": ipa_cfg.weight_type,
                    "start_at": ipa_cfg.start_at,
                    "end_at": ipa_cfg.end_at,
                    "combine_embeds": "concat",
                    "embeds_scaling": "V only",
                },
            }
            current_model = ["53", 0]

        # 3. Positive CLIP Text Encode
        workflow["3"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": current_clip, "text": positive_prompt},
        }

        # 4. Negative CLIP Text Encode
        workflow["4"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": current_clip, "text": negative_prompt},
        }

        # Conditioning 참조 (ControlNet 적용 시 변경됨)
        current_positive = ["3", 0]
        current_negative = ["4", 0]

        # ControlNet (포즈 제어)
        if controlnet_image and self.controlnet_config:
            cn_cfg = self.controlnet_config

            if controlnet_extract:
                # DWPose로 포즈 추출 (레퍼런스 이미지에서)
                # IP-Adapter LoadImage 노드가 있으면 재사용, 없으면 새로 생성
                if "50" in workflow:
                    pose_image_ref = ["50", 0]
                else:
                    workflow["60"] = {
                        "class_type": "LoadImage",
                        "inputs": {"image": controlnet_image},
                    }
                    pose_image_ref = ["60", 0]

                # 63: DWPose Estimator (comfyui_controlnet_aux)
                workflow["63"] = {
                    "class_type": "DWPreprocessor",
                    "inputs": {
                        "image": pose_image_ref,
                        "detect_hand": "enable",
                        "detect_body": "enable",
                        "detect_face": "enable",
                        "resolution": 1024,
                    },
                }
                controlnet_image_ref = ["63", 0]
            else:
                # 라이브러리 포즈 이미지 직접 사용
                workflow["60"] = {
                    "class_type": "LoadImage",
                    "inputs": {"image": controlnet_image},
                }
                controlnet_image_ref = ["60", 0]

            # 61: ControlNet Loader
            workflow["61"] = {
                "class_type": "ControlNetLoader",
                "inputs": {"control_net_name": cn_cfg.model_name},
            }

            # 62: ControlNet Apply Advanced
            workflow["62"] = {
                "class_type": "ControlNetApplyAdvanced",
                "inputs": {
                    "positive": current_positive,
                    "negative": current_negative,
                    "control_net": ["61", 0],
                    "image": controlnet_image_ref,
                    "strength": cn_cfg.strength,
                    "start_percent": cn_cfg.start_percent,
                    "end_percent": cn_cfg.end_percent,
                },
            }

            # KSampler가 ControlNet 출력 사용
            current_positive = ["62", 0]
            current_negative = ["62", 1]

        # 6. Empty Latent Image
        workflow["6"] = {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": cfg.width, "height": cfg.height, "batch_size": 1},
        }

        # 5. KSampler (1차)
        workflow["5"] = {
            "class_type": "KSampler",
            "inputs": {
                "model": current_model,
                "positive": current_positive,
                "negative": current_negative,
                "latent_image": ["6", 0],
                "seed": seed,
                "steps": cfg.steps,
                "cfg": cfg.cfg,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
            },
        }

        if cfg.hires:
            # Hires Fix Pipeline
            workflow["9"] = {
                "class_type": "VAEDecodeTiled",
                "inputs": {
                    "samples": ["5", 0],
                    "vae": ["1", 2],
                    "tile_size": 512,
                    "overlap": 64,
                    "temporal_size": 64,
                    "temporal_overlap": 8,
                },
            }

            workflow["10"] = {
                "class_type": "UpscaleModelLoader",
                "inputs": {"model_name": cfg.upscaler},
            }

            workflow["11"] = {
                "class_type": "ImageUpscaleWithModel",
                "inputs": {"upscale_model": ["10", 0], "image": ["9", 0]},
            }

            workflow["12"] = {
                "class_type": "ImageScale",
                "inputs": {
                    "image": ["11", 0],
                    "upscale_method": "nearest-exact",
                    "width": cfg.width,
                    "height": cfg.height,
                    "crop": "disabled",
                },
            }

            workflow["13"] = {
                "class_type": "VAEEncodeTiled",
                "inputs": {
                    "pixels": ["12", 0],
                    "vae": ["1", 2],
                    "tile_size": 512,
                    "overlap": 64,
                    "temporal_size": 64,
                    "temporal_overlap": 8,
                },
            }

            workflow["14"] = {
                "class_type": "KSampler",
                "inputs": {
                    "model": current_model,
                    "positive": current_positive,
                    "negative": current_negative,
                    "latent_image": ["13", 0],
                    "seed": seed,
                    "steps": cfg.hires_steps,
                    "cfg": cfg.cfg,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": cfg.hires_denoise,
                },
            }

            workflow["7"] = {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["14", 0], "vae": ["1", 2]},
            }
        else:
            workflow["7"] = {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
            }

        # 8. Save Image
        workflow["8"] = {
            "class_type": "SaveImage",
            "inputs": {"images": ["7", 0], "filename_prefix": filename_prefix},
        }

        return workflow, seed
