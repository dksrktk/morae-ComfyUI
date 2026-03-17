"""Character profile management: reference images, LoRA, tags."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LoRAEntry:
    """A LoRA model associated with a character."""

    path: str  # Path to .safetensors file
    weight: float = 0.8  # Model weight
    clip_weight: float = 1.0  # CLIP weight
    trigger_word: str = ""  # Trigger word to prepend to prompt

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "weight": self.weight,
            "clip_weight": self.clip_weight,
            "trigger_word": self.trigger_word,
        }

    @classmethod
    def from_dict(cls, d: dict) -> LoRAEntry:
        return cls(**d)


@dataclass
class CharacterProfile:
    """A character's visual identity for consistent generation.

    Bundles reference images, LoRA, and prompt tags to maintain
    character consistency across batch generation.
    """

    name: str
    # Reference face image(s) for IP-Adapter FaceID
    reference_images: list[str] = field(default_factory=list)
    # LoRA models for this character
    loras: list[LoRAEntry] = field(default_factory=list)
    # Character-specific tags to prepend to prompts
    tags: list[str] = field(default_factory=list)
    # Tags to add to negative prompt
    negative_tags: list[str] = field(default_factory=list)
    # Description for reference
    description: str = ""

    # IP-Adapter settings
    ip_adapter_weight: float = 0.7
    ip_adapter_weight_type: str = "linear"  # linear, ease in, ease out, etc.

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "reference_images": self.reference_images,
            "loras": [l.to_dict() for l in self.loras],
            "tags": self.tags,
            "negative_tags": self.negative_tags,
            "description": self.description,
            "ip_adapter_weight": self.ip_adapter_weight,
            "ip_adapter_weight_type": self.ip_adapter_weight_type,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CharacterProfile:
        loras = [LoRAEntry.from_dict(l) for l in d.pop("loras", [])]
        profile = cls(**d)
        profile.loras = loras
        return profile

    @property
    def tag_string(self) -> str:
        """Comma-separated tag string for prompt prepending."""
        return ", ".join(self.tags)

    @property
    def negative_tag_string(self) -> str:
        return ", ".join(self.negative_tags)

    @property
    def primary_reference(self) -> Optional[str]:
        """First reference image path, if any."""
        return self.reference_images[0] if self.reference_images else None

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        )
        logger.info(f"Character profile saved: {self.name} -> {path}")

    @classmethod
    def load(cls, path: Path) -> CharacterProfile:
        data = json.loads(path.read_text())
        return cls.from_dict(data)


class CharacterLibrary:
    """Manage a collection of character profiles.

    Directory structure:
        characters_dir/
        ├── character_a/
        │   ├── profile.json
        │   ├── reference_01.png
        │   └── reference_02.png
        └── character_b/
            ├── profile.json
            └── reference_01.png
    """

    def __init__(self, characters_dir: str | Path):
        self.characters_dir = Path(characters_dir)
        self.profiles: dict[str, CharacterProfile] = {}
        self._load()

    def _load(self) -> None:
        if not self.characters_dir.exists():
            logger.warning(f"Characters directory not found: {self.characters_dir}")
            return

        for item in sorted(self.characters_dir.iterdir()):
            if not item.is_dir():
                continue

            profile_file = item / "profile.json"
            if profile_file.exists():
                try:
                    profile = CharacterProfile.load(profile_file)
                    # Resolve relative reference image paths
                    profile.reference_images = [
                        str(item / p) if not Path(p).is_absolute() else p
                        for p in profile.reference_images
                    ]
                    # Resolve relative LoRA paths
                    for lora in profile.loras:
                        if not Path(lora.path).is_absolute():
                            lora.path = str(item / lora.path)
                    self.profiles[profile.name] = profile
                except Exception as e:
                    logger.error(f"Failed to load profile from {profile_file}: {e}")
            else:
                # Auto-discover: create minimal profile from directory contents
                self._auto_profile(item)

        logger.info(
            f"Character library loaded: {len(self.profiles)} characters "
            f"({', '.join(self.profiles.keys())})"
        )

    def _auto_profile(self, char_dir: Path) -> None:
        """Create a minimal profile from image files in directory."""
        extensions = {".png", ".jpg", ".jpeg", ".webp"}
        images = [
            str(f)
            for f in sorted(char_dir.iterdir())
            if f.suffix.lower() in extensions
        ]
        if not images:
            return

        name = char_dir.name
        self.profiles[name] = CharacterProfile(
            name=name,
            reference_images=images,
        )

    def get(self, name: str) -> Optional[CharacterProfile]:
        return self.profiles.get(name)

    @property
    def names(self) -> list[str]:
        return list(self.profiles.keys())

    def create_character(
        self,
        name: str,
        reference_images: list[str] | None = None,
        tags: list[str] | None = None,
        lora_path: str | None = None,
        lora_weight: float = 0.8,
        lora_trigger: str = "",
    ) -> CharacterProfile:
        """Create and save a new character profile."""
        char_dir = self.characters_dir / name
        char_dir.mkdir(parents=True, exist_ok=True)

        loras = []
        if lora_path:
            loras.append(
                LoRAEntry(
                    path=lora_path,
                    weight=lora_weight,
                    trigger_word=lora_trigger,
                )
            )

        profile = CharacterProfile(
            name=name,
            reference_images=reference_images or [],
            tags=tags or [],
            loras=loras,
        )

        profile.save(char_dir / "profile.json")
        self.profiles[name] = profile
        return profile
