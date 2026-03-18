# Censorship Pipeline

YOLO + imgutils 하이브리드 검열 파이프라인. AutoCensor 스타일 마스크 처리.

## 아키텍처

```
Input Image
    ↓
┌─────────────────────────────────────┐
│  YOLO Segmentation (primary)        │  ← mask 포함 감지
│  ntd11_anime_nsfw_segm_v5-variant1  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  imgutils (fallback)                │  ← YOLO 놓친 케이스 보완
│  detect_censors + NudeNet           │     bbox만 반환
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  IoU Deduplication (0.3)            │  ← 중복 제거
│  YOLO에 없는 imgutils 감지만 추출   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  SAM2 (for imgutils detections)     │  ← bbox → mask 생성
│  sam2.1_hiera_large                 │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Mask Post-processing               │
│  - unify_mask_fragments             │  ← 분절 마스크 통합 (Closing + Hull)
│  - refine_mask_autocensor_style     │  ← B-spline + supersample + AA
└─────────────────────────────────────┘
    ↓
Alpha Blending (white/black fill)
    ↓
Output Image
```

## 파일 구조

| 파일 | 역할 |
|------|------|
| `pipeline.py` | 메인 파이프라인, 하이브리드 감지 로직 |
| `detector.py` | YOLO/imgutils 감지기 클래스 |
| `segmentation.py` | SAM2, 마스크 후처리 함수들 |
| `filter.py` | 필터 효과 (blur, pixelate 등) |

## 모델

| 모델 | 경로 | 용도 |
|------|------|------|
| YOLO Segm | `models/yolo/ntd11_anime_nsfw_segm_v5-variant1.pt` | 1차 감지 (mask 포함) |
| SAM2 | `models/sam2/sam2.1_hiera_large.pt` | bbox → mask 변환 |
| imgutils | HuggingFace cache | 2차 감지 (YOLO fallback) |

## 설치

```bash
# 의존성 설치
pip install -r morae_pipeline/requirements.txt

# YOLO 모델 다운로드 (수동)
# models/yolo/ntd11_anime_nsfw_segm_v5-variant1.pt
# → Civitai 또는 HuggingFace에서 다운로드

# SAM2 모델 다운로드
mkdir -p models/sam2
wget -P models/sam2 https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

## 사용법

```bash
# 기본 사용
python run_censor.py <input_dir> <output_dir>

# 예시
python run_censor.py tests/164730986 output_censored
```

## 주요 함수

### detector.py
- `YOLOSegmentationDetector.detect()` - YOLO 감지 (mask 포함)
- `CensorDetector.detect()` - imgutils 감지 (bbox만)

### segmentation.py
- `SAM2Refiner.refine()` - bbox → SAM2 mask 생성
- `unify_mask_fragments()` - Morphological Closing + Convex Hull
- `refine_mask_autocensor_style()` - B-spline smoothing + supersample AA

### pipeline.py
- `CensorshipPipeline.process_image()` - 단일 이미지 처리
- `CensorshipPipeline._find_missed_detections()` - IoU 기반 누락 감지

## 감지 대상

```python
LABEL_GROUPS = {
    "anus": ['anus', 'anal', 'exposed_anus', ...],
    "genital": ['penis', 'pussy', 'vagina', 'testicles', ...],
    "breast": ['nipple', 'exposed_breast', ...]  # 기본 OFF
}
```

## 성능

4개 테스트셋 (254 이미지):
- YOLO: 621 감지
- imgutils 보완: +294 감지 (32% 추가)
- **총 915 감지**
