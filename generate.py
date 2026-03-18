#!/usr/bin/env python3
"""
generate.py - 자연어 → DeepSeek → ComfyUI 이미지 생성 (Hires Fix + 듀얼 LoRA)
Based on 꼴짤1 workflow

Usage:
  python generate.py "설명"                              # 기본 (Hires Fix ON)
  python generate.py "설명" --no-hires                   # Hires Fix 끄기
  python generate.py "설명" --lora wakitan.safetensors   # 커스텀 LoRA
  python generate.py --list-styles                       # 스타일 목록 보기
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

# Default generation settings (from 꼴짤1 workflow)
DEFAULT_MODEL = "waiIllustriousSDXL_v160.safetensors"
DEFAULT_WIDTH = 832
DEFAULT_HEIGHT = 1216
DEFAULT_STEPS = 25
DEFAULT_CFG = 7.0

# Default LoRA settings
DEFAULT_LORA1 = "wakitan.safetensors"
DEFAULT_LORA1_STRENGTH = 1.0
DEFAULT_LORA2 = None  # illustrious_masterpieces_v3 - 다운로드 필요
DEFAULT_LORA2_STRENGTH = 0.8

# 꼴짤1 테스트용 프롬프트
TEST_POSITIVE = """wakitan,,, masterpiece, best quality, very aesthetic, uncensored,
1girl, solo, breasts, looking at viewer, open mouth, bangs, hair ornament, dress, cleavage, hair between eyes, bare shoulders, nipples, blue hair, purple eyes, sidelocks, sky, tongue, hairclip, tongue out, pink eyes, armpits, huge breasts, side ponytail, , night, halterneck, building, revealing clothes, night sky, blue nails, backless outfit, selfie, cityscape, backless dress, skyscraper, halter dress, evening gown, plunging neckline, fellatio gesture, silver dress, st. louis (azur lane), st. louis (luxurious wheels) (azur lane)"""

TEST_NEGATIVE = """low quality, worst quality, bad quality, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, text, signature, watermark, username, artist name, wet, (from below:1.5), mosaic censoring, (censored:1.2), blood, weapon, knife, yandere trance, blank censor, bar censor, abs, muscular,"""

# Hires Fix settings
DEFAULT_UPSCALER = "4x-UltraSharp.pth"
DEFAULT_HIRES_STEPS = 40
DEFAULT_HIRES_DENOISE = 0.4

# Standing illustration settings (스탠딩 일러스트용)
STANDING_POSITIVE = "(cowboy shot:1.3), thighs visible, white background, simple background"
STANDING_NEGATIVE = "full body, feet, shoes, close-up, portrait, face only, detailed background, scenery, outdoors, indoors"

# Default negative prompt (from 꼴짤1)
NEGATIVE_PROMPT = "low quality, worst quality, bad quality, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, text, signature, watermark, username, artist name, wet, (from below:1.5), mosaic censoring, (censored:1.2), blood, weapon, knife, yandere trance, blank censor, bar censor, abs, muscular,"

# 스타일 프리셋 사용 시: LLM은 캐릭터/상황만 생성
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
```

## 예시
입력: "비 오는 밤 편의점 앞 고양이귀 소녀"
```yaml
positive: "1girl, solo, (purple hair:1.2), long hair, (cat ears:1.1), standing, looking at viewer, convenience store, night, rain, wet, wet clothes, neon lights, urban, atmospheric"
```"""

# 스타일 없이 사용 시: 전체 생성
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
    """Convert NAI format to ComfyUI format"""
    result = nai_prompt
    pattern = r'(-?[\d.]+)::([^:]+?)\s*::'

    def replace_weight(match):
        nai_weight = float(match.group(1))
        content = match.group(2).strip()

        if nai_weight < 0:
            if nai_weight <= -3:
                return ""
            else:
                comfy_weight = max(0.5, 1.0 + nai_weight * 0.1)
                return f"({content}:{comfy_weight:.1f})"

        if nai_weight <= 0.5:
            comfy_weight = 0.8
        elif nai_weight <= 1.5:
            comfy_weight = 0.9 + (nai_weight - 0.5) * 0.2
        elif nai_weight <= 3:
            comfy_weight = 1.1 + (nai_weight - 1.5) * 0.1
        else:
            comfy_weight = min(1.4, 1.25 + (nai_weight - 3) * 0.02)

        if 0.95 <= comfy_weight <= 1.05:
            return content

        return f"({content}:{comfy_weight:.1f})"

    result = re.sub(pattern, replace_weight, result)
    result = re.sub(r',\s*,+', ',', result)
    result = re.sub(r'\s+', ' ', result)
    result = result.strip().strip(',').strip()

    return result


def list_styles(styles: dict, categories: dict):
    """Print available styles"""
    print("\n=== 사용 가능한 스타일 프리셋 ===\n")

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

        if 'ISSUE' in notes or '망가짐' in notes:
            continue

        info = {'version': version, 'artists': artists, 'notes': notes}
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
            for v in versions[:10]:
                artists_str = ', '.join(v['artists']) if v['artists'] else 'N/A'
                print(f"  {v['version']}: {artists_str}")
            if len(versions) > 10:
                print(f"  ... 외 {len(versions)-10}개")
            print()


def call_deepseek(user_input: str, use_style: bool = False) -> str:
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
        "temperature": 0.4,  # 낮은 temperature로 더 결정적인 출력
        "max_tokens": 500 if use_style else 800
    }

    print(f"[DeepSeek] 프롬프트 생성 중...")
    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()

    result = response.json()
    content = result["choices"][0]["message"]["content"].strip()

    # YAML 형식 파싱 시도 (새 형식)
    yaml_match = re.search(r'```yaml\s*\n(.*?)```', content, re.DOTALL)
    if yaml_match:
        yaml_content = yaml_match.group(1).strip()
        # positive: "..." 형식에서 추출
        pos_match = re.search(r'positive:\s*["\']?(.+?)["\']?\s*$', yaml_content, re.MULTILINE | re.IGNORECASE)
        if pos_match:
            return pos_match.group(1).strip().strip('"\'')

    # 기존 POSITIVE: 형식 파싱 (폴백)
    pos_match = re.search(r'POSITIVE:\s*(.+?)(?=NEGATIVE:|$|```)', content, re.DOTALL | re.IGNORECASE)
    if pos_match:
        positive = pos_match.group(1).strip().strip('`').strip()
    else:
        positive = content.replace('```', '').replace('POSITIVE:', '').replace('positive:', '').strip()

    return positive


def build_workflow(positive_prompt: str, negative_prompt: str, args) -> dict:
    """Build ComfyUI workflow (꼴짤1 style with Hires Fix)"""

    seed = random.randint(1, 2**32 - 1)

    workflow = {}

    # 1. Checkpoint Loader
    workflow["1"] = {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": args.model}
    }

    # 2. CLIP Set Last Layer
    workflow["2"] = {
        "class_type": "CLIPSetLastLayer",
        "inputs": {
            "clip": ["1", 1],
            "stop_at_clip_layer": -2
        }
    }

    # 현재 model/clip 출력 추적
    current_model = ["1", 0]
    current_clip = ["2", 0]

    # 15. LoRA 1 (wakitan)
    if args.lora1:
        workflow["15"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": current_model,
                "clip": current_clip,
                "lora_name": args.lora1,
                "strength_model": args.lora1_strength,
                "strength_clip": args.lora1_strength
            }
        }
        current_model = ["15", 0]
        current_clip = ["15", 1]

    # 16. LoRA 2 (illustrious_masterpieces)
    if args.lora2:
        workflow["16"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": current_model,
                "clip": current_clip,
                "lora_name": args.lora2,
                "strength_model": args.lora2_strength,
                "strength_clip": args.lora2_strength
            }
        }
        current_model = ["16", 0]
        current_clip = ["16", 1]

    # 3. Positive CLIP Text Encode
    workflow["3"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": current_clip,
            "text": positive_prompt
        }
    }

    # 4. Negative CLIP Text Encode
    workflow["4"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": current_clip,
            "text": negative_prompt
        }
    }

    # 6. Empty Latent Image
    workflow["6"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "width": args.width,
            "height": args.height,
            "batch_size": 1
        }
    }

    # 5. KSampler (1차)
    workflow["5"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": current_model,
            "positive": ["3", 0],
            "negative": ["4", 0],
            "latent_image": ["6", 0],
            "seed": seed,
            "steps": args.steps,
            "cfg": args.cfg,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0
        }
    }

    if args.hires:
        # Hires Fix Pipeline

        # 9. VAE Decode Tiled (1차 결과)
        workflow["9"] = {
            "class_type": "VAEDecodeTiled",
            "inputs": {
                "samples": ["5", 0],
                "vae": ["1", 2],
                "tile_size": 512,
                "overlap": 64,
                "temporal_size": 64,
                "temporal_overlap": 8
            }
        }

        # 10. Upscale Model Loader
        workflow["10"] = {
            "class_type": "UpscaleModelLoader",
            "inputs": {"model_name": args.upscaler}
        }

        # 11. Image Upscale With Model
        workflow["11"] = {
            "class_type": "ImageUpscaleWithModel",
            "inputs": {
                "upscale_model": ["10", 0],
                "image": ["9", 0]
            }
        }

        # 12. Image Scale (다운스케일 to original size)
        workflow["12"] = {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["11", 0],
                "upscale_method": "nearest-exact",
                "width": args.width,
                "height": args.height,
                "crop": "disabled"
            }
        }

        # 13. VAE Encode Tiled
        workflow["13"] = {
            "class_type": "VAEEncodeTiled",
            "inputs": {
                "pixels": ["12", 0],
                "vae": ["1", 2],
                "tile_size": 512,
                "overlap": 64,
                "temporal_size": 64,
                "temporal_overlap": 8
            }
        }

        # 14. KSampler (2차 - Hires Fix)
        workflow["14"] = {
            "class_type": "KSampler",
            "inputs": {
                "model": current_model,
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["13", 0],
                "seed": seed,
                "steps": args.hires_steps,
                "cfg": args.cfg,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": args.hires_denoise
            }
        }

        # 7. VAE Decode (최종)
        workflow["7"] = {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["14", 0],
                "vae": ["1", 2]
            }
        }
    else:
        # No Hires Fix - 바로 디코드
        workflow["7"] = {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["5", 0],
                "vae": ["1", 2]
            }
        }

    # 8. Save Image
    workflow["8"] = {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["7", 0],
            "filename_prefix": "gen"
        }
    }

    return workflow, seed


def queue_prompt(workflow: dict) -> str:
    """Queue workflow in ComfyUI"""
    response = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow}, timeout=10)
    if response.status_code != 200:
        import json
        print(f"[DEBUG] Error response: {json.dumps(response.json(), indent=2)}")
    response.raise_for_status()
    return response.json()["prompt_id"]


def wait_for_completion(prompt_id: str, timeout: int = 600) -> dict:
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
        images = result["outputs"]["8"]["images"]
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
    parser = argparse.ArgumentParser(description='자연어 → 이미지 생성 (Hires Fix)')
    parser.add_argument('prompt', nargs='*', help='생성할 이미지 설명')
    parser.add_argument('--style', '-s', type=str, help='스타일 프리셋 (예: v127, v07)')
    parser.add_argument('--list-styles', '-l', action='store_true', help='스타일 목록 보기')

    # Model settings
    parser.add_argument('--model', '-m', type=str, default=DEFAULT_MODEL, help=f'체크포인트 모델 (기본: {DEFAULT_MODEL})')
    parser.add_argument('--width', type=int, default=DEFAULT_WIDTH, help=f'너비 (기본: {DEFAULT_WIDTH})')
    parser.add_argument('--height', type=int, default=DEFAULT_HEIGHT, help=f'높이 (기본: {DEFAULT_HEIGHT})')
    parser.add_argument('--steps', type=int, default=DEFAULT_STEPS, help=f'샘플링 스텝 (기본: {DEFAULT_STEPS})')
    parser.add_argument('--cfg', type=float, default=DEFAULT_CFG, help=f'CFG 스케일 (기본: {DEFAULT_CFG})')

    # LoRA settings
    parser.add_argument('--lora1', type=str, default=DEFAULT_LORA1, help=f'LoRA 1 (기본: {DEFAULT_LORA1})')
    parser.add_argument('--lora1-strength', type=float, default=DEFAULT_LORA1_STRENGTH, help=f'LoRA 1 강도 (기본: {DEFAULT_LORA1_STRENGTH})')
    parser.add_argument('--lora2', type=str, default=DEFAULT_LORA2, help=f'LoRA 2 (기본: {DEFAULT_LORA2})')
    parser.add_argument('--lora2-strength', type=float, default=DEFAULT_LORA2_STRENGTH, help=f'LoRA 2 강도 (기본: {DEFAULT_LORA2_STRENGTH})')
    parser.add_argument('--no-lora1', action='store_true', help='LoRA 1 비활성화')
    parser.add_argument('--no-lora2', action='store_true', help='LoRA 2 비활성화')
    parser.add_argument('--trigger', '-t', type=str, default='wakitan', help='LoRA trigger word (기본: wakitan)')

    # Hires Fix settings
    parser.add_argument('--hires', action='store_true', default=False, help='Hires Fix 사용')
    parser.add_argument('--no-hires', action='store_true', help='Hires Fix 비활성화 (기본: OFF)')
    parser.add_argument('--upscaler', type=str, default=DEFAULT_UPSCALER, help=f'업스케일러 (기본: {DEFAULT_UPSCALER})')
    parser.add_argument('--hires-steps', type=int, default=DEFAULT_HIRES_STEPS, help=f'Hires 스텝 (기본: {DEFAULT_HIRES_STEPS})')
    parser.add_argument('--hires-denoise', type=float, default=DEFAULT_HIRES_DENOISE, help=f'Hires denoise (기본: {DEFAULT_HIRES_DENOISE})')

    # Test mode
    parser.add_argument('--dry-run', action='store_true', help='꼴짤1 프롬프트로 테스트 (DeepSeek API 사용 안함)')

    # Standing illustration mode (스탠딩 일러스트)
    parser.add_argument('--standing', action='store_true', help='스탠딩 일러스트 모드 (cowboy shot + white background)')

    args = parser.parse_args()

    # Handle --no-* flags
    if args.no_hires:
        args.hires = False
    if args.no_lora1:
        args.lora1 = None
    if args.no_lora2:
        args.lora2 = None

    # Load styles
    styles, categories = load_styles()

    # List styles and exit
    if args.list_styles:
        list_styles(styles, categories)
        return

    # Check prompt (not required for dry-run)
    if not args.prompt and not args.dry_run:
        parser.print_help()
        sys.exit(1)

    user_input = " ".join(args.prompt) if args.prompt else "dry-run test"
    style_version = args.style

    print(f"\n{'='*60}")
    print(f"[입력] {user_input}")
    print(f"[모델] {args.model}")
    print(f"[해상도] {args.width}x{args.height}")
    print(f"[LoRA] {args.lora1 or 'None'} + {args.lora2 or 'None'}")
    print(f"[Hires Fix] {'ON' if args.hires else 'OFF'}")
    if args.standing:
        print(f"[스탠딩] ON (cowboy shot + white bg)")
    if style_version:
        print(f"[스타일] {style_version}")
    print(f"{'='*60}\n")

    # Get style preset if specified
    style_positive = ""

    if style_version:
        if style_version not in styles:
            print(f"[ERROR] 스타일 '{style_version}' 없음. --list-styles로 확인하세요.")
            sys.exit(1)

        style_data = styles[style_version]
        style_positive = convert_nai_to_comfyui(style_data.get('positive', ''))

        if style_version in categories:
            cat_info = categories[style_version]
            artists = cat_info.get('main_artists', [])
            notes = cat_info.get('notes', '')
            print(f"[스타일 정보] {notes}")
            print(f"[아티스트] {', '.join(artists[:5])}")
            print()

    # Generate prompt
    if args.dry_run:
        # 꼴짤1 테스트 모드
        print(f"[DRY-RUN] 꼴짤1 프롬프트 사용\n")
        final_positive = TEST_POSITIVE
        final_negative = TEST_NEGATIVE
    else:
        # DeepSeek API 호출
        try:
            scene_positive = call_deepseek(user_input, use_style=bool(style_version))
            print(f"\n[LLM 생성 태그]\n{scene_positive}\n")
        except Exception as e:
            print(f"[ERROR] DeepSeek API 호출 실패: {e}")
            sys.exit(1)

        # Combine prompts
        if style_version:
            final_positive = f"{style_positive}, {scene_positive}"
        else:
            final_positive = scene_positive

        # Add trigger word at the beginning
        if args.trigger:
            final_positive = f"{args.trigger}, {final_positive}"

        final_negative = NEGATIVE_PROMPT

    # Standing illustration mode
    if args.standing:
        final_positive = f"{final_positive}, {STANDING_POSITIVE}"
        final_negative = f"{final_negative}, {STANDING_NEGATIVE}"
        print(f"[스탠딩 모드] cowboy shot + white background 적용")

    print(f"[최종 Positive] ({len(final_positive.split(','))}태그)")
    print(f"{final_positive[:300]}..." if len(final_positive) > 300 else final_positive)
    print()

    # Build and queue workflow
    try:
        workflow, seed = build_workflow(final_positive, final_negative, args)
        print(f"[ComfyUI] 워크플로우 전송 (seed: {seed})...")
        prompt_id = queue_prompt(workflow)
    except Exception as e:
        print(f"[ERROR] ComfyUI 연결 실패: {e}")
        print("ComfyUI가 실행 중인지 확인하세요: ./run_comfy.sh")
        sys.exit(1)

    # Wait for completion (longer timeout for Hires Fix)
    timeout = 600 if args.hires else 300
    result = wait_for_completion(prompt_id, timeout=timeout)
    if not result:
        print("[ERROR] 이미지 생성 타임아웃")
        sys.exit(1)

    # Get output path
    output_path = get_output_image(result)
    if output_path:
        full_path = SCRIPT_DIR / output_path
        print(f"\n{'='*60}")
        print(f"[완료] {full_path}")
        print(f"{'='*60}\n")
    else:
        print("[WARN] 출력 이미지를 찾을 수 없습니다")


if __name__ == "__main__":
    main()
