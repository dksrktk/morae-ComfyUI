# Morae Pipeline

ComfyUI 기반 이미지 생성 자동화 파이프라인.

## 기능

### 1. 자연어 → 이미지 (autopilot)
자연어 설명을 DeepSeek API로 danbooru 태그로 변환 후 ComfyUI로 배치 생성.

```bash
morae autopilot -s session_name -p "자연어 설명" --count 10
```

### 2. 캐릭터 + 포즈 배치 생성 (character-batch)
캐릭터 프로필(태그, LoRA, 레퍼런스 이미지) + 포즈 목록으로 일관된 캐릭터 이미지 배치 생성.

```bash
# 외부 포즈 목록 사용
morae character-batch -c ayane -p poses/game.yaml -s session --images-per-pose 3

# LLM 자동 포즈 생성
morae character-batch -c ayane --auto-poses 10 --pose-style "학교생활" -s session
```

#### 옵션
- `--use-lora`: 캐릭터 LoRA 적용 (profile.json의 loras 필드)
- `--use-reference`: IP-Adapter 레퍼런스 이미지 적용 (profile.json의 reference_images 필드)

### 3. 큐레이션
생성된 이미지를 품질 점수로 A/B/C/rejected 등급 분류.

| 점수 | 설명 | 가중치 |
|------|------|--------|
| technical | 이미지 품질 (블러, 노이즈) | 50% |
| aesthetic | 미적 점수 (LAION 모델) | 30% |
| face | 얼굴 유사도 (비활성화 기본) | 20% |

등급 기준: A ≥ 0.8, B ≥ 0.6, C ≥ 0.4, rejected < 0.4

```bash
morae curate -s session --threshold-a 0.85
```

### 4. 검열 (Censorship)
YOLO + imgutils 하이브리드 검출 → SAM2 픽셀 정밀 마스킹 → 필터 적용.

- 검출: YOLO (1차) + imgutils/NudeNet (2차, YOLO 미검출분)
- 마스킹: SAM2로 정밀 세그멘테이션
- 필터: white_bar, gaussian_blur, pixelate, black_bar

## 파일 구조

```
morae_pipeline/
├── cli.py              # CLI 진입점 (morae 명령어)
├── runner.py           # 파이프라인 오케스트레이터
├── client.py           # ComfyUI WebSocket 클라이언트
├── config.py           # 설정 dataclass들
├── workflow.py         # ComfyUI 워크플로우 빌더
├── prompt_generator.py # DeepSeek API 프롬프트 생성
├── pose_list.py        # 포즈 목록 관리 + LLM 자동 생성
├── storage.py          # 출력 파일 정리
├── queue_manager.py    # 배치 작업 큐 관리
├── censorship/         # 검열 파이프라인
│   ├── pipeline.py     # 메인 검열 파이프라인
│   ├── detector.py     # YOLO + imgutils 검출
│   ├── sam2_refiner.py # SAM2 마스크 정제
│   └── filter.py       # 모자이크/블러 필터
├── curation/           # 큐레이션 파이프라인
│   ├── pipeline.py     # 메인 큐레이션 파이프라인
│   ├── aesthetic.py    # CLIP + LAION 미적 점수
│   └── technical.py    # 기술 품질 점수
└── character/          # 캐릭터 프로필 관리
    └── profile.py      # CharacterProfile, CharacterLibrary
```

## 캐릭터 프로필

```
input/characters/
└── ayane/
    ├── profile.json
    └── reference_01.png
```

profile.json 예시:
```json
{
  "name": "ayane",
  "tags": ["1girl", "purple hair", "twin tails", "school uniform"],
  "negative_tags": ["blonde hair", "short hair"],
  "reference_images": ["reference_01.png"],
  "loras": [
    {"path": "ayane_v1.safetensors", "weight": 0.8, "trigger_word": "ayane"}
  ]
}
```

## 포즈 목록

```yaml
# poses/game_standing.yaml
name: 게임_스탠딩
poses:
  - id: normal
    description: "정면을 바라보며 서있음, 기본 표정"
  - id: smile
    description: "밝게 웃으며 손을 흔듦"
  - id: angry
    description: "팔짱 끼고 화난 표정"
```

## 설정 (config.yaml)

```yaml
comfyui:
  host: "127.0.0.1"
  port: 8188
  input_dir: "/path/to/ComfyUI/input"

generation:
  model: "waiIllustriousSDXL_v160.safetensors"
  width: 832
  height: 1216
  steps: 25
  cfg: 7.0
  lora1: "v127style_lora.safetensors"  # 그림체 LoRA (없으면 null)
  lora1_strength: 0.8
  deepseek_api_key: "sk-..."

ipadapter:
  enabled: true
  model: "ip-adapter_sdxl_vit-h.safetensors"
  clip_vision_model: "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
  weight: 0.7

curation:
  enabled: true
  grade_a_min: 0.8
  grade_b_min: 0.6
  grade_c_min: 0.4

censorship:
  enabled: true
  filter_method: "white_bar"  # white_bar, gaussian_blur, pixelate
```

## 출력 구조

```
output/morae/{session}/
├── raw/              # 원본 이미지
├── graded/
│   ├── A/            # 최상급
│   ├── B/            # 양호
│   ├── C/            # 사용 가능
│   └── rejected/     # 불량
├── service/          # 검열된 이미지
├── scores.json       # 점수 데이터
├── summary.txt       # 요약
└── censorship_report.json
```

## 필요 모델

| 모델 | 경로 | 용도 |
|------|------|------|
| Checkpoint | models/checkpoints/ | 기본 모델 |
| LoRA | models/loras/ | 그림체/캐릭터 |
| IP-Adapter | models/ipadapter/ | 레퍼런스 스타일 |
| CLIP Vision | models/clip_vision/ | IP-Adapter용 |
| SAM2 | models/sam2/ | 검열 마스킹 |
| YOLO | models/yolo/ | 검열 검출 |
| Aesthetic | models/aesthetic/ | 큐레이션 점수 |

## Git 커밋 규칙

- 커밋 시 Co-Authored-By 라인 추가하지 않음
- 커밋 메시지는 conventional commits 형식 (feat:, fix:, docs: 등)
