"""LLM 기반 프로젝트 자동 생성."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
import yaml

from .character.profile import CharacterProfile
from .outfit import OutfitList
from .pose_list import PoseList, PoseListEntry
from .project import Project

logger = logging.getLogger(__name__)

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"


@dataclass
class CharacterSpec:
    """파싱된 캐릭터 명세."""
    name: str
    tags: list[str] = field(default_factory=list)
    negative_tags: list[str] = field(default_factory=list)
    character_class: str = ""  # mage, warrior, healer, etc.
    role: str = ""  # maid, student, etc.
    description: str = ""


# 캐릭터 파싱 시스템 프롬프트
CHARACTER_PARSING_PROMPT = """너는 캐릭터 설명을 파싱하는 전문가다.
사용자가 제공한 캐릭터 설명을 분석하여 구조화된 정보를 추출한다.

## 입력
자연어 캐릭터 설명 (예: "보라머리 거유 마법사 메이드")

## 출력 형식 (JSON)
```json
{
  "characters": [
    {
      "name": "영문 이름 (자동 생성, 예: luna)",
      "tags": ["1girl", "purple hair", "large breasts", "mage", "maid"],
      "negative_tags": ["small breasts"],
      "class": "mage",
      "role": "maid",
      "description": "원본 설명"
    }
  ]
}
```

## 규칙
1. name: 외모/성격에서 영감받은 영문 이름 (예: 보라머리 → luna, violet)
2. tags: danbooru 스타일 태그로 변환
   - 머리색: purple hair, red hair, blonde hair 등
   - 체형: large breasts, petite, tall 등
   - 직업: mage, warrior, healer, maid 등
3. class: 전투 직업 (mage, warrior, archer, healer, knight 등)
4. role: 비전투 역할 (maid, student, princess 등)
5. negative_tags: 피해야 할 태그 (체형 반대 등)

## 예시
입력: "보라머리 거유 마법사, 붉은 머리 전사"
```json
{
  "characters": [
    {
      "name": "violet",
      "tags": ["1girl", "purple hair", "large breasts", "mage", "staff"],
      "negative_tags": ["small breasts", "flat chest"],
      "class": "mage",
      "role": "",
      "description": "보라머리 거유 마법사"
    },
    {
      "name": "scarlet",
      "tags": ["1girl", "red hair", "warrior", "sword", "armor"],
      "negative_tags": [],
      "class": "warrior",
      "role": "",
      "description": "붉은 머리 전사"
    }
  ]
}
```"""


# 컨텍스트 감지 시스템 프롬프트
CONTEXT_DETECTION_PROMPT = """너는 게임/프로젝트 컨텍스트를 분석하는 전문가다.
프로젝트 설명과 캐릭터 클래스를 분석하여 필요한 포즈/복장 카테고리를 결정한다.

## 사용 가능한 카테고리

### 포즈 카테고리
- combat: 전투 (mage, warrior, knight, archer, healer 등 전투 클래스가 있을 때)
- service: 서비스직 (maid, waitress, butler 등이 있을 때)
- daily_life: 일상 (학교, 일상물 등)
- nsfw_base: 성인용 (명시적으로 성인용/nsfw 언급 시에만)

### 복장 카테고리
- common: 공통 (항상 포함)
- mage: 마법사
- warrior: 전사
- healer: 힐러
- maid: 메이드

## 출력 형식 (JSON)
```json
{
  "genre": ["fantasy", "maid"],
  "pose_presets": ["combat", "service"],
  "outfit_presets": ["common", "mage", "maid"],
  "custom_poses_needed": true,
  "custom_pose_context": "마법사+메이드 조합"
}
```

## 규칙
1. genre: 프로젝트 장르 태그
2. pose_presets: 필요한 포즈 프리셋 (sfw_base는 자동 포함)
3. outfit_presets: 필요한 복장 프리셋 (common은 자동 포함)
4. custom_poses_needed: Layer 3 커스텀 포즈 필요 여부
5. custom_pose_context: 커스텀 포즈 생성 시 참고할 컨텍스트"""


# 커스텀 포즈 생성 프롬프트
CUSTOM_POSE_GENERATION_PROMPT = """너는 게임 캐릭터용 특수 포즈를 생성하는 전문가다.
주어진 컨텍스트에 맞는 고유한 포즈를 생성한다.

## 규칙
1. 기본 표정 12종(default, smile, laugh 등)과 중복되지 않는 포즈
2. 전투 포즈(attack, defense 등)와 중복되지 않는 포즈
3. 컨텍스트에 특화된 고유한 포즈 5-10개
4. 각 포즈는 명확히 구별 가능해야 함

## 출력 형식 (YAML)
```yaml
poses:
  - id: pose_id
    prompt_tags: "danbooru tags for the pose"
```

## 예시
컨텍스트: "마법사 + 메이드"
```yaml
poses:
  - id: magic_cleaning
    prompt_tags: "casting magic, cleaning, floating objects, magical broom"
  - id: tea_ceremony_magic
    prompt_tags: "serving tea, magic circle, floating teacup"
  - id: spell_book_reading
    prompt_tags: "reading spell book, glowing pages, focused"
```"""


class ProjectGenerator:
    """LLM 기반 프로젝트 자동 생성기."""

    def __init__(
        self,
        api_key: str,
        api_url: str = DEEPSEEK_API_URL,
    ):
        self.api_key = api_key
        self.api_url = api_url

    def generate(
        self,
        prompt: str,
        name: str,
        nsfw: bool = False,
    ) -> Project:
        """자연어 프롬프트에서 전체 프로젝트 생성.

        Args:
            prompt: 프로젝트 설명 (예: "메이드 카페 판타지 게임, 마법사 메이드, 전사 메이드")
            name: 프로젝트 이름
            nsfw: NSFW 포즈 포함 여부

        Returns:
            Project
        """
        logger.info(f"[ProjectGenerator] 프로젝트 생성 시작: {name}")

        # Step 1: 캐릭터 파싱
        characters = self._parse_characters(prompt)
        logger.info(f"  캐릭터 {len(characters)}명 파싱 완료")

        # Step 2: 컨텍스트 감지
        context = self._detect_context(prompt, characters, nsfw)
        logger.info(f"  컨텍스트: {context}")

        # Step 3: 포즈 리스트 조합
        poses = self._build_pose_list(context, nsfw)
        logger.info(f"  포즈 {len(poses)}개 조합 완료")

        # Step 4: 복장 리스트 조합
        outfits = self._build_outfit_list(context)
        logger.info(f"  복장 {len(outfits)}개 조합 완료")

        # Step 5: CharacterProfile 생성
        profiles = {}
        for spec in characters:
            profile = CharacterProfile(
                name=spec.name,
                tags=spec.tags,
                negative_tags=spec.negative_tags,
                description=spec.description,
            )
            profiles[spec.name] = profile

        project = Project(
            name=name,
            genre=context.get("genre", []),
            poses=poses,
            outfits=outfits,
            characters=profiles,
            nsfw=nsfw,
        )

        logger.info(f"[ProjectGenerator] 프로젝트 생성 완료: {project.summary()}")
        return project

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """LLM API 호출."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.4,
            "max_tokens": 2000,
        }
        response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def _parse_json_response(self, content: str) -> dict:
        """LLM 응답에서 JSON 추출."""
        json_match = re.search(r"```json\s*\n(.*?)```", content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1).strip())
        # JSON 블록 없으면 전체 파싱 시도
        return json.loads(content)

    def _parse_yaml_response(self, content: str) -> dict:
        """LLM 응답에서 YAML 추출."""
        yaml_match = re.search(r"```yaml\s*\n(.*?)```", content, re.DOTALL)
        if yaml_match:
            return yaml.safe_load(yaml_match.group(1).strip())
        return yaml.safe_load(content)

    def _parse_characters(self, prompt: str) -> list[CharacterSpec]:
        """프롬프트에서 캐릭터 정보 파싱."""
        logger.info("[ProjectGenerator] 캐릭터 파싱 중...")
        response = self._call_llm(CHARACTER_PARSING_PROMPT, prompt)

        try:
            data = self._parse_json_response(response)
            characters = []
            for c in data.get("characters", []):
                characters.append(CharacterSpec(
                    name=c.get("name", "character"),
                    tags=c.get("tags", []),
                    negative_tags=c.get("negative_tags", []),
                    character_class=c.get("class", ""),
                    role=c.get("role", ""),
                    description=c.get("description", ""),
                ))
            return characters
        except Exception as e:
            logger.error(f"캐릭터 파싱 실패: {e}")
            # 폴백: 기본 캐릭터 1개
            return [CharacterSpec(name="character", tags=["1girl"], description=prompt)]

    def _detect_context(
        self,
        prompt: str,
        characters: list[CharacterSpec],
        nsfw: bool,
    ) -> dict:
        """프롬프트와 캐릭터에서 컨텍스트 감지."""
        logger.info("[ProjectGenerator] 컨텍스트 감지 중...")

        char_info = "\n".join([
            f"- {c.name}: class={c.character_class}, role={c.role}"
            for c in characters
        ])
        user_prompt = f"프로젝트: {prompt}\n\n캐릭터:\n{char_info}\n\nNSFW: {nsfw}"

        response = self._call_llm(CONTEXT_DETECTION_PROMPT, user_prompt)

        try:
            return self._parse_json_response(response)
        except Exception as e:
            logger.error(f"컨텍스트 감지 실패: {e}")
            # 폴백: 기본 컨텍스트
            return {
                "genre": ["general"],
                "pose_presets": [],
                "outfit_presets": ["common"],
                "custom_poses_needed": False,
            }

    def _build_pose_list(self, context: dict, nsfw: bool) -> PoseList:
        """컨텍스트 기반 포즈 리스트 조합."""
        # Layer 1: 코어 (필수)
        poses = PoseList.load_preset("sfw_base")
        poses.name = "project_poses"

        # Layer 2: 프리셋 선택
        for preset_id in context.get("pose_presets", []):
            if preset_id == "nsfw_base" and not nsfw:
                continue
            try:
                preset = PoseList.load_preset(preset_id)
                poses.extend(preset)
            except FileNotFoundError:
                logger.warning(f"포즈 프리셋 없음: {preset_id}")

        # NSFW 명시적 요청 시 추가
        if nsfw and "nsfw_base" not in context.get("pose_presets", []):
            try:
                nsfw_preset = PoseList.load_preset("nsfw_base")
                poses.extend(nsfw_preset)
            except FileNotFoundError:
                pass

        # Layer 3: 커스텀 포즈 (필요시)
        if context.get("custom_poses_needed", False):
            custom_context = context.get("custom_pose_context", "")
            custom_poses = self._generate_custom_poses(custom_context)
            poses.extend(custom_poses)

        return poses

    def _generate_custom_poses(self, context: str) -> PoseList:
        """LLM으로 커스텀 포즈 생성."""
        logger.info(f"[ProjectGenerator] 커스텀 포즈 생성: {context}")

        response = self._call_llm(CUSTOM_POSE_GENERATION_PROMPT, f"컨텍스트: {context}")

        try:
            data = self._parse_yaml_response(response)
            custom_poses = []
            for p in data.get("poses", []):
                custom_poses.append(PoseListEntry(
                    id=p.get("id", "custom"),
                    prompt_tags=p.get("prompt_tags", ""),
                    description=p.get("description", ""),
                ))
            return PoseList(name="custom", poses=custom_poses)
        except Exception as e:
            logger.error(f"커스텀 포즈 생성 실패: {e}")
            return PoseList(name="custom")

    def _build_outfit_list(self, context: dict) -> OutfitList:
        """컨텍스트 기반 복장 리스트 조합."""
        preset_ids = ["common"]  # 항상 포함
        preset_ids.extend(context.get("outfit_presets", []))

        # 중복 제거
        preset_ids = list(dict.fromkeys(preset_ids))

        return OutfitList.from_presets(preset_ids)
