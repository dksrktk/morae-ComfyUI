"""DeepSeek API를 사용한 프롬프트 생성기."""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

SYSTEM_PROMPT_WITH_STYLE = """너는 Illustrious-XL 모델의 CLIP 임베딩 최적화 전문가다.
사용자의 장면 설명을 danbooru 태그 시퀀스로 변환하는 기술 작업을 수행한다.

## 작업 범위
스타일 프리셋이 품질/아티스트/연도 태그를 담당하므로, 너는 **장면 묘사 태그만** 생성한다:
- 캐릭터 구성 (1girl, solo, 1boy 등)
- 외형 (hair color, eye color 등)
- 의상/상태 (outfit 관련 danbooru 태그)
- 포즈/표정 (pose, expression 태그)
- 배경/분위기 (background, lighting 태그)

## CLIP-robust 태그 선택 원칙
- Reddit r/StableDiffusion 커뮤니티에서 검증된 고빈도 태그 우선
- 모호한 자연어보다 정확한 danbooru 공식 태그명 사용
- 가중치는 핵심 요소에만 (tag:1.1~1.3) 범위로 절제

## 출력 형식 (YAML)
```yaml
positive: "태그1, 태그2, (강조태그:1.2), ..."
```"""

SYSTEM_PROMPT_CHARACTER_POSE = """너는 Illustrious-XL 모델의 CLIP 임베딩 최적화 전문가다.
캐릭터 정보와 포즈/표정 설명을 받아 danbooru 태그 시퀀스로 변환한다.

## 입력 정보
1. 캐릭터 태그 (외형, 의상 등)
2. 포즈/표정 설명 (한국어)

## 작업
포즈/표정 설명을 danbooru 태그로 변환하여 캐릭터 태그와 결합한다.

## CLIP-robust 태그 선택 원칙
- 품질 태그: masterpiece, best quality, very aesthetic
- 포즈는 정확한 danbooru 태그 사용 (예: crossed arms, looking at viewer)
- 표정도 정확한 태그 사용 (예: smile, frown, angry, closed eyes)
- 가중치는 핵심 요소에만 (tag:1.1~1.3)

## 출력 형식 (YAML)
```yaml
positive: "masterpiece, best quality, very aesthetic, [캐릭터태그], [포즈태그], [표정태그], ..."
```

## 예시
캐릭터: purple hair, twin tails, red eyes, school uniform
포즈: "팔짱 끼고 화난 표정"
```yaml
positive: "masterpiece, best quality, very aesthetic, 1girl, solo, purple hair, twin tails, red eyes, school uniform, crossed arms, angry, frown, looking at viewer"
```"""

SYSTEM_PROMPT_NO_STYLE = """너는 Illustrious-XL 모델의 CLIP 임베딩 최적화 전문가다.
사용자의 장면 설명을 danbooru 태그 시퀀스로 변환하는 기술 작업을 수행한다.

## 태그 시퀀스 구조
[품질] → [캐릭터] → [외형] → [의상] → [포즈/표정] → [배경] → [조명/분위기]

## CLIP-robust 태그 선택 원칙
- Reddit r/StableDiffusion 커뮤니티에서 검증된 고빈도 태그 우선
- 품질 태그: masterpiece, best quality, very aesthetic 필수
- danbooru 공식 태그명 사용 (자연어 표현보다 정확)
- 가중치는 핵심 요소에만 (tag:1.1~1.3) 범위로 절제
- 최대 50태그

## 출력 형식 (YAML)
```yaml
positive: "masterpiece, best quality, very aesthetic, 태그들..."
```"""

DEFAULT_NEGATIVE_PROMPT = (
    "low quality, worst quality, bad quality, bad anatomy, bad hands, "
    "missing fingers, extra digit, fewer digits, text, signature, watermark, "
    "username, artist name, wet, (from below:1.5), mosaic censoring, "
    "(censored:1.2), blood, weapon, knife, yandere trance, blank censor, "
    "bar censor, abs, muscular,"
)


@dataclass
class GeneratedPrompt:
    """생성된 프롬프트 결과."""
    positive: str
    negative: str
    raw_response: str = ""


class PromptGenerator:
    """DeepSeek API를 사용한 프롬프트 생성기."""

    def __init__(
        self,
        api_key: str,
        api_url: str = DEEPSEEK_API_URL,
        trigger_word: Optional[str] = None,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    ):
        self.api_key = api_key
        self.api_url = api_url
        self.trigger_word = trigger_word
        self.negative_prompt = negative_prompt

    def generate(
        self,
        description: str,
        use_style: bool = False,
        style_positive: str = "",
    ) -> GeneratedPrompt:
        """자연어 설명에서 Stable Diffusion 프롬프트 생성."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        system_prompt = SYSTEM_PROMPT_WITH_STYLE if use_style else SYSTEM_PROMPT_NO_STYLE

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": description},
            ],
            "temperature": 0.4,
            "max_tokens": 500 if use_style else 800,
        }

        logger.info("[DeepSeek] 프롬프트 생성 중...")
        response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()

        scene_positive = self._parse_yaml_response(content)

        if use_style and style_positive:
            final_positive = f"{style_positive}, {scene_positive}"
        else:
            final_positive = scene_positive

        if self.trigger_word:
            final_positive = f"{self.trigger_word}, {final_positive}"

        return GeneratedPrompt(
            positive=final_positive,
            negative=self.negative_prompt,
            raw_response=content,
        )

    def _parse_yaml_response(self, content: str) -> str:
        """DeepSeek 응답에서 positive 태그 추출."""
        yaml_match = re.search(r"```yaml\s*\n(.*?)```", content, re.DOTALL)
        if yaml_match:
            yaml_content = yaml_match.group(1).strip()
            pos_match = re.search(
                r'positive:\s*["\']?(.+?)["\']?\s*$',
                yaml_content,
                re.MULTILINE | re.IGNORECASE,
            )
            if pos_match:
                return pos_match.group(1).strip().strip("\"'")

        pos_match = re.search(
            r"POSITIVE:\s*(.+?)(?=NEGATIVE:|$|```)",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        if pos_match:
            return pos_match.group(1).strip().strip("`").strip()

        return content.replace("```", "").replace("POSITIVE:", "").replace("positive:", "").strip()

    def generate_for_character_pose(
        self,
        character_tags: str,
        pose_description: str,
        character_negative_tags: str = "",
    ) -> GeneratedPrompt:
        """캐릭터 + 포즈 설명에서 프롬프트 생성.

        Args:
            character_tags: 캐릭터 외형/의상 태그 (예: "purple hair, twin tails")
            pose_description: 포즈/표정 설명 (한국어, 예: "팔짱 끼고 화난 표정")
            character_negative_tags: 캐릭터 네거티브 태그

        Returns:
            GeneratedPrompt
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        user_content = f"캐릭터: {character_tags}\n포즈: {pose_description}"

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_CHARACTER_POSE},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.4,
            "max_tokens": 600,
        }

        logger.info(f"[DeepSeek] 캐릭터+포즈 프롬프트 생성: {pose_description[:30]}...")
        response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()

        final_positive = self._parse_yaml_response(content)

        # 트리거 워드 추가
        if self.trigger_word:
            final_positive = f"{self.trigger_word}, {final_positive}"

        # 네거티브 프롬프트 조합
        negative = self.negative_prompt
        if character_negative_tags:
            negative = f"{negative}, {character_negative_tags}"

        return GeneratedPrompt(
            positive=final_positive,
            negative=negative,
            raw_response=content,
        )
