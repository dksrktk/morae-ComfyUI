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
class QueueConfig:
    max_concurrent: int = 1  # ComfyUI handles one at a time by default
    retry_on_failure: int = 1
    timeout_seconds: int = 600  # 10 min per image


@dataclass
class PipelineConfig:
    comfyui: ComfyUIConfig = field(default_factory=ComfyUIConfig)
    curation: CurationConfig = field(default_factory=CurationConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    output_dir: str = "./output/morae"

    @classmethod
    def from_yaml(cls, path: Path) -> PipelineConfig:
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        config = cls()
        if "comfyui" in data:
            config.comfyui = ComfyUIConfig(**data["comfyui"])
        if "curation" in data:
            config.curation = CurationConfig(**data["curation"])
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
