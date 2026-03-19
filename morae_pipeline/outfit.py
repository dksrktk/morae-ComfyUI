"""복장 목록 관리: YAML 로드/저장."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# 프리셋 디렉토리 경로
PRESETS_DIR = Path(__file__).parent / "presets" / "outfits"


@dataclass
class OutfitEntry:
    """단일 복장 항목."""
    id: str
    prompt_tags: str
    negative_tags: str = ""

    def to_dict(self) -> dict:
        d = {"id": self.id, "prompt_tags": self.prompt_tags}
        if self.negative_tags:
            d["negative_tags"] = self.negative_tags
        return d

    @classmethod
    def from_dict(cls, d: dict) -> OutfitEntry:
        return cls(
            id=d["id"],
            prompt_tags=d.get("prompt_tags", ""),
            negative_tags=d.get("negative_tags", ""),
        )


@dataclass
class OutfitList:
    """복장 목록 (YAML 파일 기반)."""
    name: str = ""
    outfits: list[OutfitEntry] = field(default_factory=list)
    required: bool = False
    triggers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "outfits": [o.to_dict() for o in self.outfits],
        }
        if self.required:
            d["required"] = self.required
        if self.triggers:
            d["triggers"] = self.triggers
        return d

    @classmethod
    def from_dict(cls, d: dict) -> OutfitList:
        return cls(
            name=d.get("name", d.get("id", "")),
            outfits=[OutfitEntry.from_dict(o) for o in d.get("outfits", [])],
            required=d.get("required", False),
            triggers=d.get("triggers", []),
        )

    @classmethod
    def load(cls, path: Path | str) -> OutfitList:
        """YAML 파일에서 복장 목록 로드."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        outfit_list = cls.from_dict(data)
        logger.info(f"OutfitList loaded: {path} ({len(outfit_list.outfits)} outfits)")
        return outfit_list

    @classmethod
    def load_preset(cls, preset_id: str) -> OutfitList:
        """프리셋에서 복장 목록 로드.

        Args:
            preset_id: 프리셋 ID (예: "common", "mage")

        Returns:
            OutfitList
        """
        preset_path = PRESETS_DIR / f"{preset_id}.yaml"
        if not preset_path.exists():
            raise FileNotFoundError(f"Outfit preset not found: {preset_id}")
        return cls.load(preset_path)

    @classmethod
    def list_presets(cls) -> list[str]:
        """사용 가능한 프리셋 목록 반환."""
        if not PRESETS_DIR.exists():
            return []
        return [p.stem for p in PRESETS_DIR.glob("*.yaml")]

    @classmethod
    def from_presets(cls, preset_ids: list[str]) -> OutfitList:
        """여러 프리셋을 조합하여 복장 목록 생성.

        Args:
            preset_ids: 프리셋 ID 리스트 (예: ["common", "mage"])

        Returns:
            통합된 OutfitList
        """
        combined = cls(name="combined")
        for preset_id in preset_ids:
            try:
                preset = cls.load_preset(preset_id)
                combined.extend(preset)
            except FileNotFoundError:
                logger.warning(f"Outfit preset not found, skipping: {preset_id}")
        return combined

    def save(self, path: Path) -> None:
        """YAML 파일로 저장."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, allow_unicode=True, default_flow_style=False)
        logger.info(f"OutfitList saved: {path}")

    def extend(self, other: OutfitList) -> None:
        """다른 OutfitList의 복장들을 추가 (중복 ID 제외)."""
        existing_ids = {o.id for o in self.outfits}
        for outfit in other.outfits:
            if outfit.id not in existing_ids:
                self.outfits.append(outfit)
                existing_ids.add(outfit.id)

    def get(self, outfit_id: str) -> Optional[OutfitEntry]:
        """ID로 복장 검색."""
        for outfit in self.outfits:
            if outfit.id == outfit_id:
                return outfit
        return None

    def filter(self, outfit_ids: list[str]) -> OutfitList:
        """특정 ID의 복장만 필터링."""
        filtered = [o for o in self.outfits if o.id in outfit_ids]
        return OutfitList(name=self.name, outfits=filtered)

    def __len__(self) -> int:
        return len(self.outfits)

    def __iter__(self):
        return iter(self.outfits)
