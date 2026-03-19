"""Pipeline configuration with YAML support."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class ComfyUIConfig:
    host: str = "127.0.0.1"
    port: int = 8188
    # ComfyUI input 폴더 경로 (IP-Adapter 레퍼런스 이미지 복사용)
    input_dir: str = "/mnt/c/ComfyUI_windows_portable/ComfyUI/input"

    @property
    def http_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def ws_url(self) -> str:
        return f"ws://{self.host}:{self.port}/ws"


@dataclass
class CurationConfig:
    enabled: bool = True
    device: str = "cuda"  # GPU by default for speed

    # Scorer weights (must sum to 1.0 when all enabled)
    technical_weight: float = 0.5
    aesthetic_weight: float = 0.3
    face_weight: float = 0.2

    # Grade thresholds (composite score 0.0 ~ 1.0)
    grade_a_min: float = 0.8
    grade_b_min: float = 0.6
    grade_c_min: float = 0.4

    # Optional: reference face image for similarity scoring
    face_reference_path: Optional[str] = None

    # Enable/disable individual scorers
    technical_enabled: bool = True
    aesthetic_enabled: bool = True
    face_enabled: bool = False  # Off by default, needs reference image


@dataclass
class CharacterConfig:
    enabled: bool = False
    characters_dir: Optional[str] = None  # e.g. "./input/characters"
    default_character: Optional[str] = None  # Character name to use by default

    # Tag injection
    prepend_tags: bool = True  # Prepend character tags to positive prompt
    append_negative_tags: bool = True  # Append character negative tags


@dataclass
class IPAdapterConfig:
    enabled: bool = False
    # IP-Adapter model (SDXL용)
    model: str = "ip-adapter_sdxl_vit-h.safetensors"
    # CLIP Vision model
    clip_vision_model: str = "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
    # Weight (0.0-1.0, 높을수록 reference 영향 큼)
    weight: float = 0.7
    # Weight type: linear, ease in, ease out, style transfer (SDXL)
    weight_type: str = "linear"
    # Start/End (언제부터 언제까지 적용)
    start_at: float = 0.0
    end_at: float = 1.0


@dataclass
class ControlNetConfig:
    enabled: bool = False  # Off by default
    # Illustrious XL 전용 ControlNet (SD 1.5 모델 사용 금지 - 색상 왜곡 발생)
    # https://civitai.com/models/1359846/illustrious-xl-controlnet-openpose
    model_name: str = "illustrious-xl-controlnet-openpose.safetensors"

    # Pose library (presets/controlnet/)
    library_path: str = "morae_pipeline/presets/controlnet"
    extraction_enabled: bool = True  # DWPose 런타임 추출 허용

    # Legacy pose library (deprecated, use library_path instead)
    pose_library_path: Optional[str] = None
    pose_category: Optional[str] = None

    # Strength range for randomization (prevents identical outputs)
    strength: float = 0.7
    strength_min: float = 0.5
    strength_max: float = 0.85

    # Ending step: ControlNet only guides early denoising (allows variation in details)
    start_percent: float = 0.0
    end_percent: float = 0.4
    end_percent_min: float = 0.3
    end_percent_max: float = 0.5

    # Variation seed for subtle differences per image
    variation_enabled: bool = True
    variation_strength: float = 0.05  # Subtle variation


@dataclass
class GenerationConfig:
    """이미지 생성 설정."""
    enabled: bool = True

    # Model
    model: str = "waiIllustriousSDXL_v160.safetensors"
    width: int = 832
    height: int = 1216
    steps: int = 25
    cfg: float = 7.0

    # LoRA (None = 사용 안함, config.yaml에서 명시적으로 설정)
    lora1: Optional[str] = None
    lora1_strength: float = 1.0
    lora2: Optional[str] = None
    lora2_strength: float = 0.8
    trigger_word: str = ""

    # Hires Fix
    hires: bool = False
    hires_steps: int = 40
    hires_denoise: float = 0.4
    upscaler: str = "4x-UltraSharp.pth"

    # DeepSeek API
    deepseek_api_key: str = ""
    deepseek_api_url: str = "https://api.deepseek.com/v1/chat/completions"


@dataclass
class CensorshipConfig:
    enabled: bool = False
    device: str = "cuda"

    # Detection (imgutils detect_censors + NudeNet anus)
    confidence_threshold: float = 0.25
    anus_confidence_threshold: float = 0.2  # Lower for anus (harder to detect)
    bbox_expand_ratio: float = 0.15  # Expand bbox before SAM2 for better coverage
    target_classes: Optional[list[str]] = None  # None = ["penis", "pussy", "anus"]

    # SAM2 pixel-precise segmentation
    sam2_model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_checkpoint: str = "./models/sam2/sam2.1_hiera_large.pt"

    # Mask refinement (dilate = expand outward for full coverage)
    erode_pixels: int = 0
    dilate_pixels: int = 3
    blur_boundary: int = 3

    # Filter settings
    filter_method: str = "white_bar"  # white_bar, gaussian_blur, pixelate, black_bar
    blur_radius: int = 40
    pixelate_factor: int = 10


@dataclass
class QueueConfig:
    max_concurrent: int = 1  # ComfyUI handles one at a time by default
    retry_on_failure: int = 1
    timeout_seconds: int = 600  # 10 min per image


@dataclass
class PipelineConfig:
    comfyui: ComfyUIConfig = field(default_factory=ComfyUIConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    curation: CurationConfig = field(default_factory=CurationConfig)
    character: CharacterConfig = field(default_factory=CharacterConfig)
    controlnet: ControlNetConfig = field(default_factory=ControlNetConfig)
    ipadapter: IPAdapterConfig = field(default_factory=IPAdapterConfig)
    censorship: CensorshipConfig = field(default_factory=CensorshipConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    output_dir: str = "./output/morae"

    @classmethod
    def from_yaml(cls, path: Path) -> PipelineConfig:
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        config = cls()
        if "comfyui" in data:
            config.comfyui = ComfyUIConfig(**data["comfyui"])
        if "generation" in data:
            config.generation = GenerationConfig(**data["generation"])
        if "curation" in data:
            config.curation = CurationConfig(**data["curation"])
        if "character" in data:
            config.character = CharacterConfig(**data["character"])
        if "controlnet" in data:
            config.controlnet = ControlNetConfig(**data["controlnet"])
        if "ipadapter" in data:
            config.ipadapter = IPAdapterConfig(**data["ipadapter"])
        if "censorship" in data:
            config.censorship = CensorshipConfig(**data["censorship"])
        if "queue" in data:
            config.queue = QueueConfig(**data["queue"])
        if "output_dir" in data:
            config.output_dir = data["output_dir"]
        return config

    def to_yaml(self, path: Path) -> None:
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    @classmethod
    def default(cls) -> PipelineConfig:
        return cls()
