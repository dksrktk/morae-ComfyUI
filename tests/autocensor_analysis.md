# AutoCensor 역공학 분석

PyInstaller로 패키징된 `AutoCensor.exe`에서 bytecode 디컴파일을 통해 추출한 핵심 알고리즘 분석.

## 개요

- **원본**: `AutoCensor.exe` (PyInstaller onedir, Python 3.11, Win64)
- **모델**: `ntd11_anime_nsfw_segm_v5-variant1.pt` (~20MB, YOLO segmentation)
- **핵심 라이브러리**: ultralytics (YOLO), torch, cv2, scipy, PIL
- **GUI**: PyQt6 (ComfyUI 통합 시 불필요)

## 모델 정보

- **프레임워크**: `ultralytics.YOLO` (YOLOv8 segmentation)
- **추론 파라미터**: `imgsz=1280`, `retina_masks=True`
- **입력**: 이미지 경로 또는 numpy array
- **출력**: segmentation mask + bounding box + class

### 검출 클래스 (3그룹)

```python
TARGETS_ANUS = ('anus', 'anal', 'ass', 'asshole', 'exposed_anus', 'buttocks')

TARGETS_GENITAL = ('penis', 'exposed_penis', 'genitalia', 'genitals', 'vulva',
                   'vagina', 'pussy', 'exposed_vulva', 'testicles', 'cunnus',
                   'female_genital')

TARGETS_BREAST = ('nipple', 'nipples', 'exposed_nipple', 'breast', 'exposed_breast')
```

## 검열 방식 (4가지)

| color_idx | 모드 | 설명 |
|-----------|------|------|
| 0 | 검정 채움 | `(0, 0, 0, 255)` 단색 |
| 1 | 흰색 채움 | `(255, 255, 255, 255)` 단색 |
| 'mosaic' | 모자이크 | cv2 resize 축소→확대 |
| 'blur' | 가우시안 블러 | cv2.GaussianBlur |

## 핵심 알고리즘

### 1. 모델 로딩 (`_get_thread_model`, line 1908)

```python
import threading
from ultralytics import YOLO

_thread_local = threading.local()

def _get_thread_model(model_path, device, num_threads_per_worker):
    model = getattr(_thread_local, 'model', None)
    if model is None:
        try:
            torch.set_num_threads(num_threads_per_worker)
        except Exception:
            pass
        model = YOLO(model_path)
        model.to(device)
        _thread_local.model = model
    return model
```

### 2. 이미지 검열 (`_censor_one_image`, line 1941)

```python
def _censor_one_image(args):
    (item_path, input_root, output_dirs, subfolder, targets, threshold,
     color_idx, blur_sigma, supersample, model_path, device, threads_per_worker,
     censor_mode, mosaic_block_size, blur_censor_radius, censor_opacity,
     output_format, output_quality, preserve_exif) = args

    censor_fill = (0, 0, 0, 255) if color_idx < 1 else (255, 255, 255, 255)
    model = _get_thread_model(model_path, device, threads_per_worker)

    # YOLO segmentation 추론
    results = model.predict(item_path, conf=threshold, verbose=False,
                            device=device, imgsz=1280, retina_masks=True)

    orig_pil = Image.open(item_path)
    exif_data = orig_pil.info.get('exif', None) if preserve_exif else None
    img = orig_pil.convert('RGBA')
    img_np = np.array(img, dtype=np.float32)
    orig_np = img_np.copy()
    applied = False

    for r in results:
        for m_tensor, b in zip(r.masks.data, r.boxes):
            class_name = model.names[int(b.cls.cpu().numpy())]
            if class_name in targets:
                mask_raw = m_tensor.cpu().numpy()
                mask_np = refine_mask(mask_raw, img.size,
                                      blur_sigma=blur_sigma,
                                      supersample=supersample)

                alpha = mask_np[:, :, np.newaxis] * censor_opacity

                if censor_mode == 'mosaic':
                    h, w = orig_np.shape[:2]
                    bs = max(1, mosaic_block_size)
                    small = cv2.resize(orig_np.astype(np.uint8),
                                       (max(1, w // bs), max(1, h // bs)),
                                       interpolation=cv2.INTER_AREA)
                    fill_img = cv2.resize(small, (w, h),
                                          interpolation=cv2.INTER_NEAREST)
                    fill_img = np.array(fill_img, dtype=np.float32)
                    img_np = fill_img * alpha + img_np * (1.0 - alpha)

                elif censor_mode == 'blur':
                    k = max(3, blur_censor_radius * 2 + 1)
                    if k % 2 == 0:
                        k += 1
                    blurred = cv2.GaussianBlur(orig_np.astype(np.uint8),
                                               (k, k), 0)
                    blurred = np.array(blurred, dtype=np.float32)
                    img_np = blurred * alpha + img_np * (1.0 - alpha)

                else:  # 단색 채움 (검정/흰색)
                    fill_arr = np.array(censor_fill, dtype=np.float32)
                    img_np = fill_arr * alpha + img_np * (1.0 - alpha)

                applied = True

    # 저장 처리 (생략)
    # ...
```

### 3. 마스크 정제 (`refine_mask`, line 57)

YOLO가 출력한 저해상도 mask를 원본 이미지 크기로 정제하는 함수.
슈퍼샘플링 + 컨투어 스무딩 + 가우시안 블러로 경계를 부드럽게 처리.

```python
from scipy.ndimage import gaussian_filter

def refine_mask(mask_np, target_size, blur_sigma, supersample):
    w, h = target_size
    ss = max(1, supersample)

    # 0~1 범위면 255 스케일로 변환
    if mask_np.max() <= 1.0:
        m8 = (mask_np * 255).astype(np.uint8)
    else:
        m8 = mask_np.astype(np.uint8)

    mh, mw = m8.shape[:2]

    # 크기가 다르면 리사이즈
    if mw != w or mh != h:
        m8 = cv2.resize(m8, (w, h), interpolation=cv2.INTER_CUBIC)

    # 이진화
    _, binary = cv2.threshold(m8, 127, 255, cv2.THRESH_BINARY)

    # 컨투어 추출 → 스무딩 → 슈퍼샘플링 해상도로 재그리기
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP,
                                            cv2.CHAIN_APPROX_NONE)
    smoothed = []
    ss_w, ss_h = w * ss, h * ss

    for c in contours:
        sc = _smooth_contour(c.reshape(-1, 2).astype(np.float64))
        sc = (sc * ss).astype(np.int32)
        smoothed.append(sc)

    canvas = np.zeros((ss_h, ss_w), dtype=np.uint8)
    for i, c in enumerate(smoothed):
        # hierarchy 기반 hole 처리 포함
        cv2.drawContours(canvas, [c], -1, 255, cv2.FILLED, lineType=cv2.LINE_AA)

    # 다운샘플링
    result = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)
    result = result.astype(np.float32) / 255.0

    # 경계 블러 처리
    if blur_sigma > 0:
        k = int(blur_sigma * 3)
        kern = np.ones((k, k), dtype=np.uint8)
        dilated = cv2.dilate(binary, kern, iterations=1)
        eroded = cv2.erode(binary, kern, iterations=1)
        edge_band = dilated - eroded
        blurred = gaussian_filter(result, sigma=blur_sigma)
        # edge_band 영역만 블러된 값 적용
        result = np.clip(result, 0.0, 1.0)

    return result
```

### 4. 컨투어 스무딩 (`_smooth_contour`, line 25)

```python
def _smooth_contour(cnt, smooth_factor):
    # 컨투어 포인트를 smooth_factor 기반으로 보간/스무딩
    # (bytecode에서 상세 로직 미복원 - 이동평균 또는 스플라인 추정)
    pass
```

## ComfyUI 통합 시 필요 사항

### 필요 파일
- `ntd11_anime_nsfw_segm_v5-variant1.pt` (모델, ~20MB)

### 필요 패키지
- `ultralytics` (ComfyUI 기본 환경에 없음, 추가 설치 필요)
- `torch`, `numpy`, `cv2`, `scipy`, `Pillow` (ComfyUI에 이미 포함)

### 통합 방향
1. 커스텀 노드로 구현
2. 입력: IMAGE (ComfyUI 텐서)
3. 출력: IMAGE (검열된 이미지) + MASK (검출 마스크)
4. GUI 관련 코드(PyQt6) 전부 불필요
5. 파일 I/O 로직 → ComfyUI 텐서 변환으로 대체
