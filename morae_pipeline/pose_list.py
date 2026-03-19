"""포즈 목록 관리: YAML 로드/저장 + LLM 자동 생성."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
import requests

logger = logging.getLogger(__name__)


# 프리셋 디렉토리 경로
PRESETS_DIR = Path(__file__).parent / "presets" / "poses"


@dataclass
class PoseListEntry:
    """단일 포즈 항목."""
    id: str
    description: str = ""  # 한국어 설명 (LLM 생성용)
    prompt_tags: str = ""  # danbooru 태그 (직접 사용)
    controlnet_image: Optional[str] = None  # ControlNet 포즈 이미지 (옵션)

    def to_dict(self) -> dict:
        d = {"id": self.id}
        if self.description:
            d["description"] = self.description
        if self.prompt_tags:
            d["prompt_tags"] = self.prompt_tags
        if self.controlnet_image:
            d["controlnet"] = self.controlnet_image
        return d

    @classmethod
    def from_dict(cls, d: dict) -> PoseListEntry:
        return cls(
            id=d["id"],
            description=d.get("description", ""),
            prompt_tags=d.get("prompt_tags", ""),
            controlnet_image=d.get("controlnet"),
        )


@dataclass
class PoseList:
    """포즈 목록 (YAML 파일 기반)."""
    name: str = ""
    poses: list[PoseListEntry] = field(default_factory=list)
    required: bool = False
    triggers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "poses": [p.to_dict() for p in self.poses],
        }
        if self.required:
            d["required"] = self.required
        if self.triggers:
            d["triggers"] = self.triggers
        return d

    @classmethod
    def from_dict(cls, d: dict) -> PoseList:
        return cls(
            name=d.get("name", d.get("id", "")),
            poses=[PoseListEntry.from_dict(p) for p in d.get("poses", [])],
            required=d.get("required", False),
            triggers=d.get("triggers", []),
        )

    @classmethod
    def load(cls, path: Path | str) -> PoseList:
        """YAML 파일에서 포즈 목록 로드."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        pose_list = cls.from_dict(data)

        # 상대 경로 해결
        parent = path.parent
        for pose in pose_list.poses:
            if pose.controlnet_image and not Path(pose.controlnet_image).is_absolute():
                pose.controlnet_image = str(parent / pose.controlnet_image)

        logger.info(f"PoseList loaded: {path} ({len(pose_list.poses)} poses)")
        return pose_list

    @classmethod
    def load_preset(cls, preset_id: str) -> PoseList:
        """프리셋에서 포즈 목록 로드.

        Args:
            preset_id: 프리셋 ID (예: "sfw_base", "combat")

        Returns:
            PoseList
        """
        preset_path = PRESETS_DIR / f"{preset_id}.yaml"
        if not preset_path.exists():
            raise FileNotFoundError(f"Pose preset not found: {preset_id}")
        return cls.load(preset_path)

    @classmethod
    def list_presets(cls) -> list[str]:
        """사용 가능한 프리셋 목록 반환."""
        if not PRESETS_DIR.exists():
            return []
        return [p.stem for p in PRESETS_DIR.glob("*.yaml")]

    def save(self, path: Path) -> None:
        """YAML 파일로 저장."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, allow_unicode=True, default_flow_style=False)
        logger.info(f"PoseList saved: {path}")

    def extend(self, other: PoseList) -> None:
        """다른 PoseList의 포즈들을 추가 (중복 ID 제외)."""
        existing_ids = {p.id for p in self.poses}
        for pose in other.poses:
            if pose.id not in existing_ids:
                self.poses.append(pose)
                existing_ids.add(pose.id)

    def get(self, pose_id: str) -> Optional[PoseListEntry]:
        """ID로 포즈 검색."""
        for pose in self.poses:
            if pose.id == pose_id:
                return pose
        return None

    def filter(self, pose_ids: list[str]) -> PoseList:
        """특정 ID의 포즈만 필터링."""
        filtered = [p for p in self.poses if p.id in pose_ids]
        return PoseList(name=self.name, poses=filtered)

    def __len__(self) -> int:
        return len(self.poses)

    def __iter__(self):
        return iter(self.poses)


# LLM 포즈 자동 생성용 시스템 프롬프트
POSE_GENERATION_SYSTEM_PROMPT = """너는 캐릭터 일러스트용 포즈/표정 목록 생성 전문가다.
주어진 스타일과 개수에 맞춰 다양한 포즈와 표정을 설명한다.

## 출력 형식 (YAML)
```yaml
poses:
  - id: pose_01
    description: "포즈/표정 설명 (한국어, 구체적으로)"
  - id: pose_02
    description: "..."
```

## 규칙
1. id는 영문 소문자 + 숫자 (예: smile, angry_01, sitting_crossed_legs)
2. description은 구체적인 포즈/표정 설명 (예: "팔짱 끼고 옆을 바라보며 살짝 웃는 표정")
3. 요청한 스타일/테마에 맞게 다양하게 생성
4. 중복 없이 각각 구별되는 포즈

## 예시
스타일: "학교생활"
```yaml
poses:
  - id: studying
    description: "책상에 앉아 책을 읽으며 집중하는 모습"
  - id: waving
    description: "한 손을 들어 인사하며 밝게 웃는 모습"
  - id: thinking
    description: "턱에 손을 대고 고민하는 표정"
```"""


class PoseGenerator:
    """LLM을 사용해 포즈 목록 자동 생성."""

    def __init__(
        self,
        api_key: str,
        api_url: str = "https://api.deepseek.com/v1/chat/completions",
    ):
        self.api_key = api_key
        self.api_url = api_url

    def generate(
        self,
        count: int,
        style: str = "일반",
        character_description: str = "",
    ) -> PoseList:
        """포즈 목록 자동 생성.

        Args:
            count: 생성할 포즈 개수
            style: 스타일/테마 (예: "학교생활", "일상", "전투")
            character_description: 캐릭터 설명 (옵션, 성격 반영용)

        Returns:
            PoseList
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        user_prompt = f"스타일: {style}\n개수: {count}개"
        if character_description:
            user_prompt += f"\n캐릭터: {character_description}"

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": POSE_GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 2000,
        }

        logger.info(f"[PoseGenerator] 포즈 {count}개 생성 중 (스타일: {style})...")
        response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()

        return self._parse_response(content, style)

    def _parse_response(self, content: str, style: str) -> PoseList:
        """LLM 응답 파싱."""
        # YAML 블록 추출
        yaml_match = re.search(r"```yaml\s*\n(.*?)```", content, re.DOTALL)
        if yaml_match:
            yaml_content = yaml_match.group(1).strip()
        else:
            yaml_content = content

        try:
            data = yaml.safe_load(yaml_content)
            if isinstance(data, dict) and "poses" in data:
                poses = [
                    PoseListEntry(
                        id=p.get("id", f"pose_{i:02d}"),
                        description=p.get("description", ""),
                        prompt_tags=p.get("prompt_tags", ""),
                    )
                    for i, p in enumerate(data["poses"])
                ]
                return PoseList(name=f"auto_{style}", poses=poses)
        except Exception as e:
            logger.warning(f"YAML 파싱 실패: {e}")

        # 폴백: 단순 파싱
        poses = []
        for i, line in enumerate(content.split("\n")):
            if "description" in line.lower() or ":" in line:
                desc = line.split(":", 1)[-1].strip().strip('"\'')
                if desc and len(desc) > 5:
                    poses.append(PoseListEntry(id=f"pose_{i:02d}", description=desc))

        return PoseList(name=f"auto_{style}", poses=poses[:20])  # 최대 20개
