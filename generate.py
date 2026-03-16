#!/usr/bin/env python3
"""
generate.py - 자연어 → DeepSeek → ComfyUI 이미지 생성 (스타일 프리셋 지원)
Usage:
  python generate.py "설명"                    # 기본 (스타일 없이)
  python generate.py "설명" --style v127       # 스타일 프리셋 사용
  python generate.py --list-styles             # 스타일 목록 보기
"""

import sys
import os
import re
import json
import time
import random
import argparse
import requests
import yaml
from pathlib import Path

# Configuration
DEEPSEEK_API_KEY = "sk-8e521ad3a771401b9b225ea4c37152dd"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
COMFYUI_URL = "http://127.0.0.1:8188"

# Paths
SCRIPT_DIR = Path(__file__).parent
STYLES_DIR = SCRIPT_DIR / "styles"
PROMPTS_FILE = STYLES_DIR / "prompts_versions.yaml"
CATEGORIES_FILE = STYLES_DIR / "artist_categories.yaml"

# Default generation settings
DEFAULT_MODEL = "Illustrious-XL-v1.0.safetensors"
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_STEPS = 28
DEFAULT_CFG = 7.0

# 스타일 프리셋 사용 시: LLM은 캐릭터/상황만 생성
SYSTEM_PROMPT_WITH_STYLE = """너는 Danbooru 태그 전문가야. 사용자의 요청을 **캐릭터/상황 묘사 태그**로 변환해.

**중요: 아래 태그는 생성하지 마 (스타일 프리셋이 담당):**
- 품질 태그 (masterpiece, best quality 등)
- 아티스트 태그 (artist:xxx)
- 연도 태그 (year 2024 등)

**너의 역할: 캐릭터, 의상, 포즈, 배경, 분위기만 생성 (최대 30태그)**

**구조:** [캐릭터] → [외형] → [의상] → [포즈/표정] → [배경] → [조명/분위기]

**가중치:** 핵심 요소에만 (tag:1.1~1.3) 사용. 과도한 가중치 금지.

**출력 형식 (태그만, 설명 없이):**
```
POSITIVE: 태그들
```

**예시:**
입력: "비 오는 밤 편의점 앞 고양이귀 소녀"
```
POSITIVE: 1girl, solo, (purple hair:1.2), long hair, (cat ears:1.1), standing, looking at viewer, convenience store, night, rain, wet, wet clothes, neon lights, urban, atmospheric, moody
```"""

# 스타일 없이 사용 시: 기존 전체 생성
SYSTEM_PROMPT_NO_STYLE = """너는 Illustrious-XL(ILXL) 모델의 Danbooru 태그 프롬프팅 전문가야.

**작성 규칙:**
1. 품질 태그 필수: masterpiece, best quality, very aesthetic, absurdres, year 2025
2. 구조: [품질] → [캐릭터] → [의상] → [포즈] → [배경] → [조명]
3. 가중치: 핵심 요소에 (tag:1.1~1.3) 사용
4. 최대 50태그

**출력 형식:**
```
POSITIVE: 태그들
NEGATIVE: 네거티브 태그들
```"""

NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, bad feet, poorly drawn hands, poorly drawn face, mutation, deformed, extra limbs, extra arms, extra legs, malformed limbs, fused fingers, too many fingers, long neck"


def load_styles():
    """Load style presets from YAML files"""
    styles = {}
    categories = {}

    if PROMPTS_FILE.exists():
        with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            styles = data.get('prompts', {})

    if CATEGORIES_FILE.exists():
        with open(CATEGORIES_FILE, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            categories = data.get('version_details', {})

    return styles, categories


def convert_nai_to_comfyui(nai_prompt: str) -> str:
    """Convert NAI format to ComfyUI format
    NAI: 2::beeeeen ::  →  ComfyUI: (beeeeen:1.2)
    NAI weights are scaled: NAI 1.0 = baseline, 2.0 = strong, 0.5 = weak
    ComfyUI weights: 1.0 = baseline, 1.2 = strong, 0.8 = weak
    """
    result = nai_prompt

    # Pattern: weight::content :: (note: space before closing ::)
    # Example: 2::beeeeen :: or -3::artist collaboration ::
    pattern = r'(-?[\d.]+)::([^:]+?)\s*::'

    def replace_weight(match):
        nai_weight = float(match.group(1))
        content = match.group(2).strip()

        # Skip negative weights (these are NAI's way of de-emphasizing)
        if nai_weight < 0:
            # For negative weights, either skip or use very low weight
            if nai_weight <= -3:
                return ""  # Remove completely
            else:
                # Convert to low ComfyUI weight
                comfy_weight = max(0.5, 1.0 + nai_weight * 0.1)
                return f"({content}:{comfy_weight:.1f})"

        # Scale NAI weights to ComfyUI range
        # NAI: 0.5-1.5 is normal range, 2-4 is strong, 5+ is very strong
        # ComfyUI: 0.8-1.4 is usable range
        if nai_weight <= 0.5:
            comfy_weight = 0.8
        elif nai_weight <= 1.5:
            comfy_weight = 0.9 + (nai_weight - 0.5) * 0.2  # 0.9-1.1
        elif nai_weight <= 3:
            comfy_weight = 1.1 + (nai_weight - 1.5) * 0.1  # 1.1-1.25
        else:
            comfy_weight = min(1.4, 1.25 + (nai_weight - 3) * 0.02)  # cap at 1.4

        # Skip if weight is ~1.0
        if 0.95 <= comfy_weight <= 1.05:
            return content

        return f"({content}:{comfy_weight:.1f})"

    result = re.sub(pattern, replace_weight, result)

    # Clean up
    result = re.sub(r',\s*,+', ',', result)  # Remove empty spots
    result = re.sub(r'\s+', ' ', result)
    result = result.strip().strip(',').strip()

    return result


def list_styles(styles: dict, categories: dict):
    """Print available styles"""
    print("\n=== 사용 가능한 스타일 프리셋 ===\n")

    # Group by category
    cat_groups = {
        'general': [],
        'mature': [],
        'semi_realistic': [],
        'realistic': [],
        'vulgar': []
    }

    for version, details in categories.items():
        cats = details.get('category', [])
        notes = details.get('notes', '')
        artists = details.get('main_artists', [])[:3]

        # Skip ISSUE versions
        if 'ISSUE' in notes or '망가짐' in notes:
            continue

        info = {
            'version': version,
            'artists': artists,
            'notes': notes
        }

        for cat in cats:
            if cat in cat_groups:
                cat_groups[cat].append(info)

    cat_names = {
        'general': '일반 (모에/웹툰/일러스트)',
        'mature': '성숙 (글래머/오네쇼타)',
        'semi_realistic': '세미리얼 (3D/반실사)',
        'realistic': '실사풍',
        'vulgar': '천박 (하드코어)'
    }

    for cat, name in cat_names.items():
        versions = cat_groups.get(cat, [])
        if versions:
            print(f"【{name}】")
            for v in versions[:10]:  # Show max 10 per category
                artists_str = ', '.join(v['artists']) if v['artists'] else 'N/A'
                print(f"  {v['version']}: {artists_str}")
            if len(versions) > 10:
                print(f"  ... 외 {len(versions)-10}개")
            print()


def call_deepseek(user_input: str, use_style: bool = False) -> tuple[str, str]:
    """Call DeepSeek API to generate optimized prompt"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    system_prompt = SYSTEM_PROMPT_WITH_STYLE if use_style else SYSTEM_PROMPT_NO_STYLE

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        "temperature": 0.7,
        "max_tokens": 300 if use_style else 500
    }

    print(f"[DeepSeek] 프롬프트 생성 중...")
    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()

    result = response.json()
    content = result["choices"][0]["message"]["content"].strip()

    # Parse POSITIVE from response
    positive = ""
    negative = NEGATIVE_PROMPT

    pos_match = re.search(r'POSITIVE:\s*(.+?)(?=NEGATIVE:|$|```)', content, re.DOTALL | re.IGNORECASE)
    if pos_match:
        positive = pos_match.group(1).strip().strip('`').strip()
    else:
        positive = content.replace('```', '').replace('POSITIVE:', '').strip()

    # Parse NEGATIVE if present
    neg_match = re.search(r'NEGATIVE:\s*(.+?)(?=$|```)', content, re.DOTALL | re.IGNORECASE)
    if neg_match:
        negative = neg_match.group(1).strip().strip('`').strip()

    return positive, negative


def queue_prompt(positive_prompt: str, negative_prompt: str = NEGATIVE_PROMPT,
                 model: str = DEFAULT_MODEL, width: int = DEFAULT_WIDTH,
                 height: int = DEFAULT_HEIGHT, steps: int = DEFAULT_STEPS,
                 cfg: float = DEFAULT_CFG, lora: str = None, lora_strength: float = 0.8) -> str:
    """Queue image generation in ComfyUI"""

    seed = random.randint(1, 2**32 - 1)

    workflow = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": model}
        },
    }

    # LoRA 사용 시
    if lora:
        workflow["9"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["1", 0],
                "clip": ["1", 1],
                "lora_name": lora,
                "strength_model": lora_strength,
                "strength_clip": lora_strength
            }
        }
        model_output = ["9", 0]
        clip_output = ["9", 1]
    else:
        model_output = ["1", 0]
        clip_output = ["1", 1]

    workflow["8"] = {
        "class_type": "CLIPSetLastLayer",
        "inputs": {
            "clip": clip_output,
            "stop_at_clip_layer": -2
        }
    }
    workflow["2"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": ["8", 0],
            "text": positive_prompt
        }
    }
    workflow["3"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": ["8", 0],
            "text": negative_prompt
        }
    }
    workflow["4"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": width, "height": height, "batch_size": 1}
    }
    workflow["5"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model_output,
            "positive": ["2", 0],
            "negative": ["3", 0],
            "latent_image": ["4", 0],
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0
        }
    }
    workflow["6"] = {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["5", 0],
            "vae": ["1", 2]
        }
    }
    workflow["7"] = {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["6", 0],
            "filename_prefix": "gen"
        }
    }

    print(f"[ComfyUI] 이미지 생성 요청 (seed: {seed})...")
    response = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow}, timeout=10)
    response.raise_for_status()

    return response.json()["prompt_id"]


def wait_for_completion(prompt_id: str, timeout: int = 300) -> dict:
    """Wait for image generation to complete"""
    print(f"[ComfyUI] 생성 중", end="", flush=True)

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if prompt_id in data:
                    result = data[prompt_id]
                    if result.get("status", {}).get("status_str") == "success":
                        print(" 완료!")
                        return result
                    elif result.get("status", {}).get("status_str") == "error":
                        print(" 에러!")
                        return result
        except:
            pass

        print(".", end="", flush=True)
        time.sleep(2)

    print(" 타임아웃!")
    return None


def get_output_image(result: dict) -> str:
    """Extract output image path from result"""
    try:
        images = result["outputs"]["7"]["images"]
        if images:
            filename = images[0]["filename"]
            subfolder = images[0].get("subfolder", "")
            if subfolder:
                return f"output/{subfolder}/{filename}"
            return f"output/{filename}"
    except:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description='자연어 → 이미지 생성')
    parser.add_argument('prompt', nargs='*', help='생성할 이미지 설명')
    parser.add_argument('--style', '-s', type=str, help='스타일 프리셋 (예: v127, v07)')
    parser.add_argument('--list-styles', '-l', action='store_true', help='스타일 목록 보기')
    parser.add_argument('--model', '-m', type=str, default=DEFAULT_MODEL, help='체크포인트 모델')
    parser.add_argument('--lora', type=str, help='LoRA 파일명 (예: wakitan.safetensors)')
    parser.add_argument('--lora-strength', type=float, default=0.8, help='LoRA 강도 (기본: 0.8)')
    parser.add_argument('--trigger', '-t', type=str, help='LoRA trigger word (예: wakitan)')

    args = parser.parse_args()

    # Load styles
    styles, categories = load_styles()

    # List styles and exit
    if args.list_styles:
        list_styles(styles, categories)
        return

    # Check prompt
    if not args.prompt:
        parser.print_help()
        sys.exit(1)

    user_input = " ".join(args.prompt)
    style_version = args.style

    print(f"\n{'='*50}")
    print(f"[입력] {user_input}")
    if style_version:
        print(f"[스타일] {style_version}")
    print(f"{'='*50}\n")

    # Get style preset if specified
    style_positive = ""
    style_negative = ""

    if style_version:
        if style_version not in styles:
            print(f"[ERROR] 스타일 '{style_version}' 없음. --list-styles로 확인하세요.")
            sys.exit(1)

        style_data = styles[style_version]
        style_positive = convert_nai_to_comfyui(style_data.get('positive', ''))
        style_negative = style_data.get('negative', '')

        # Show style info
        if style_version in categories:
            cat_info = categories[style_version]
            artists = cat_info.get('main_artists', [])
            notes = cat_info.get('notes', '')
            print(f"[스타일 정보] {notes}")
            print(f"[아티스트] {', '.join(artists[:5])}")
            print()

    # Generate scene prompt via DeepSeek
    try:
        scene_positive, llm_negative = call_deepseek(user_input, use_style=bool(style_version))
        print(f"\n[LLM 생성 태그]\n{scene_positive}\n")
    except Exception as e:
        print(f"[ERROR] DeepSeek API 호출 실패: {e}")
        sys.exit(1)

    # Combine style + scene
    if style_version:
        final_positive = f"{style_positive}, {scene_positive}"
        final_negative = style_negative if style_negative else llm_negative
    else:
        final_positive = scene_positive
        final_negative = llm_negative

    # Add trigger word at the beginning if specified
    if args.trigger:
        final_positive = f"{args.trigger}, {final_positive}"

    print(f"[최종 Positive] ({len(final_positive.split(','))}태그)")
    print(f"{final_positive[:200]}..." if len(final_positive) > 200 else final_positive)
    print()

    # Queue image generation
    try:
        prompt_id = queue_prompt(final_positive, final_negative, model=args.model,
                                 lora=args.lora, lora_strength=args.lora_strength)
    except Exception as e:
        print(f"[ERROR] ComfyUI 연결 실패: {e}")
        print("ComfyUI가 실행 중인지 확인하세요: ./run_comfy.sh")
        sys.exit(1)

    # Wait for completion
    result = wait_for_completion(prompt_id)
    if not result:
        print("[ERROR] 이미지 생성 타임아웃")
        sys.exit(1)

    # Get output path
    output_path = get_output_image(result)
    if output_path:
        full_path = SCRIPT_DIR / output_path
        print(f"\n{'='*50}")
        print(f"[완료] {full_path}")
        print(f"{'='*50}\n")
    else:
        print("[WARN] 출력 이미지를 찾을 수 없습니다")


if __name__ == "__main__":
    main()
