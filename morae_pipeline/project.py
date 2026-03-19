"""프로젝트 관리: 캐릭터 × 복장 × 포즈 매트릭스 생성."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from .character.profile import CharacterProfile, CharacterLibrary
from .outfit import OutfitList
from .pose_list import PoseList

logger = logging.getLogger(__name__)

# 기본 프로젝트 디렉토리
DEFAULT_PROJECTS_DIR = Path("projects")


@dataclass
class Project:
    """프로젝트: 캐릭터 × 복장 × 포즈 매트릭스 관리."""
    name: str
    path: Optional[Path] = None
    genre: list[str] = field(default_factory=list)
    poses: PoseList = field(default_factory=PoseList)
    outfits: OutfitList = field(default_factory=OutfitList)
    characters: dict[str, CharacterProfile] = field(default_factory=dict)
    created_at: str = ""
    nsfw: bool = False

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    @classmethod
    def load(cls, path: Path | str) -> Project:
        """프로젝트 디렉토리에서 로드.

        Args:
            path: 프로젝트 디렉토리 경로 또는 프로젝트 이름

        Returns:
            Project
        """
        path = Path(path)

        # 이름만 주어진 경우 기본 디렉토리에서 찾기
        if not path.exists() and not path.is_absolute():
            path = DEFAULT_PROJECTS_DIR / path

        if not path.exists():
            raise FileNotFoundError(f"Project not found: {path}")

        # project.yaml 로드
        project_file = path / "project.yaml"
        if not project_file.exists():
            raise FileNotFoundError(f"project.yaml not found in {path}")

        with open(project_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # 포즈 로드
        poses_file = path / "poses.yaml"
        poses = PoseList.load(poses_file) if poses_file.exists() else PoseList()

        # 복장 로드
        outfits_file = path / "outfits.yaml"
        outfits = OutfitList.load(outfits_file) if outfits_file.exists() else OutfitList()

        # 캐릭터 로드
        characters_dir = path / "characters"
        characters = {}
        if characters_dir.exists():
            library = CharacterLibrary(characters_dir)
            for char_name in library.list_characters():
                characters[char_name] = library.get(char_name)

        project = cls(
            name=data.get("name", path.name),
            path=path,
            genre=data.get("genre", []),
            poses=poses,
            outfits=outfits,
            characters=characters,
            created_at=data.get("created_at", ""),
            nsfw=data.get("nsfw", False),
        )

        logger.info(
            f"Project loaded: {project.name} "
            f"({len(project.characters)} chars, {len(project.poses)} poses, {len(project.outfits)} outfits)"
        )
        return project

    def save(self) -> None:
        """프로젝트를 디렉토리에 저장."""
        if not self.path:
            self.path = DEFAULT_PROJECTS_DIR / self.name

        self.path.mkdir(parents=True, exist_ok=True)

        # project.yaml 저장
        project_data = {
            "name": self.name,
            "genre": self.genre,
            "created_at": self.created_at,
            "nsfw": self.nsfw,
        }
        with open(self.path / "project.yaml", "w", encoding="utf-8") as f:
            yaml.dump(project_data, f, allow_unicode=True, default_flow_style=False)

        # poses.yaml 저장
        self.poses.save(self.path / "poses.yaml")

        # outfits.yaml 저장
        self.outfits.save(self.path / "outfits.yaml")

        # 캐릭터 저장
        characters_dir = self.path / "characters"
        for char_name, char_profile in self.characters.items():
            char_dir = characters_dir / char_name
            char_dir.mkdir(parents=True, exist_ok=True)
            char_profile.save(char_dir / "profile.json")

        logger.info(f"Project saved: {self.path}")

    def get_combinations(
        self,
        characters: Optional[list[str]] = None,
        outfits: Optional[list[str]] = None,
        poses: Optional[list[str]] = None,
    ) -> list[tuple[str, str, str]]:
        """모든 (캐릭터, 복장, 포즈) 조합 반환.

        Args:
            characters: 특정 캐릭터만 필터 (None=전체)
            outfits: 특정 복장만 필터 (None=전체)
            poses: 특정 포즈만 필터 (None=전체)

        Returns:
            (character_name, outfit_id, pose_id) 튜플 리스트
        """
        char_names = characters or list(self.characters.keys())
        outfit_ids = outfits or [o.id for o in self.outfits]
        pose_ids = poses or [p.id for p in self.poses]

        combinations = []
        for char_name in char_names:
            if char_name not in self.characters:
                logger.warning(f"Character not found: {char_name}")
                continue
            for outfit_id in outfit_ids:
                if not self.outfits.get(outfit_id):
                    logger.warning(f"Outfit not found: {outfit_id}")
                    continue
                for pose_id in pose_ids:
                    if not self.poses.get(pose_id):
                        logger.warning(f"Pose not found: {pose_id}")
                        continue
                    combinations.append((char_name, outfit_id, pose_id))

        return combinations

    def get_total_images(self, images_per_combo: int = 1) -> int:
        """예상 총 이미지 수 계산."""
        return len(self.get_combinations()) * images_per_combo

    def summary(self) -> str:
        """프로젝트 요약 문자열."""
        combos = len(self.get_combinations())
        return (
            f"Project: {self.name}\n"
            f"  Characters: {len(self.characters)}\n"
            f"  Poses: {len(self.poses)}\n"
            f"  Outfits: {len(self.outfits)}\n"
            f"  Total combinations: {combos}\n"
            f"  Genre: {', '.join(self.genre)}\n"
            f"  NSFW: {self.nsfw}"
        )

    def add_character(self, profile: CharacterProfile) -> None:
        """캐릭터 추가."""
        self.characters[profile.name] = profile

    def get_character(self, name: str) -> Optional[CharacterProfile]:
        """캐릭터 조회."""
        return self.characters.get(name)
