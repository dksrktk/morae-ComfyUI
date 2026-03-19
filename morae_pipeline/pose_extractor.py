"""OpenPose 포즈 추출기 - ControlNet 라이브러리용.

레퍼런스 이미지에서 OpenPose 스켈레톤을 추출하여 ControlNet 포즈 라이브러리 생성.
다인 포즈 (2인 이상) 지원.

사용법:
    # 단일 이미지
    python -m morae_pipeline.pose_extractor input.jpg -o standing/default.png

    # 디렉토리 배치 처리
    python -m morae_pipeline.pose_extractor ./references/ -o ./presets/controlnet/ --batch

    # 다인 포즈 (NSFW 등)
    python -m morae_pipeline.pose_extractor couple.jpg -o nsfw/missionary.png --multi-person
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# 기본 출력 디렉토리
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "presets" / "controlnet"


class PoseExtractor:
    """DWPose 기반 포즈 추출기."""

    def __init__(
        self,
        resolution: int = 1024,
        detect_hand: bool = True,
        detect_body: bool = True,
        detect_face: bool = True,
        device: str = "cuda",
    ):
        """초기화.

        Args:
            resolution: 출력 해상도
            detect_hand: 손 감지 여부
            detect_body: 몸 감지 여부
            detect_face: 얼굴 감지 여부
            device: cuda 또는 cpu
        """
        self.resolution = resolution
        self.detect_hand = detect_hand
        self.detect_body = detect_body
        self.detect_face = detect_face
        self.device = device
        self._detector = None

    def _load_detector(self):
        """포즈 감지기 로드 (지연 로딩).

        우선순위:
        1. OpenPose (의존성 적음, 안정적)
        2. DWPose (mmcv 필요, 더 정확)
        """
        if self._detector is not None:
            return

        # 1차: OpenPose 시도 (권장 - 의존성 적음)
        try:
            from controlnet_aux import OpenposeDetector
            self._detector = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
            self._detector_type = "openpose"
            logger.info("OpenPose detector loaded")
            return
        except Exception as e:
            logger.debug(f"OpenPose failed: {e}")

        # 2차: DWPose 시도 (mmcv/mmpose/mmdet 필요)
        try:
            from controlnet_aux import DWposeDetector
            self._detector = DWposeDetector()
            self._detector_type = "dwpose"
            logger.info("DWPose detector loaded")
            return
        except ImportError:
            pass

        raise ImportError(
            "controlnet_aux 패키지가 필요합니다.\n"
            "설치: pip install controlnet_aux\n"
            "또는 ComfyUI custom_nodes/comfyui_controlnet_aux 사용"
        )

    def extract(
        self,
        image: Image.Image | str | Path,
        output_path: Optional[Path] = None,
    ) -> Image.Image:
        """이미지에서 포즈 추출.

        Args:
            image: 입력 이미지 (PIL Image, 경로, 또는 Path)
            output_path: 저장 경로 (None이면 저장 안 함)

        Returns:
            OpenPose 스켈레톤 이미지
        """
        self._load_detector()

        # 이미지 로드
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        # DWPose 추출
        pose_image = self._detector(
            image,
            detect_resolution=self.resolution,
            image_resolution=self.resolution,
        )

        # 저장
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pose_image.save(output_path)
            logger.info(f"Pose saved: {output_path}")

        return pose_image

    def extract_batch(
        self,
        input_dir: Path,
        output_dir: Path,
        recursive: bool = True,
        extensions: tuple = (".jpg", ".jpeg", ".png", ".webp"),
    ) -> list[Path]:
        """디렉토리의 모든 이미지에서 포즈 추출.

        Args:
            input_dir: 입력 디렉토리
            output_dir: 출력 디렉토리
            recursive: 하위 디렉토리 포함 여부
            extensions: 처리할 확장자

        Returns:
            생성된 포즈 이미지 경로 목록
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        # 이미지 파일 수집
        if recursive:
            files = []
            for ext in extensions:
                files.extend(input_dir.rglob(f"*{ext}"))
        else:
            files = [
                f for f in input_dir.iterdir()
                if f.suffix.lower() in extensions
            ]

        logger.info(f"Found {len(files)} images in {input_dir}")

        results = []
        for i, file in enumerate(sorted(files)):
            # 상대 경로 유지
            rel_path = file.relative_to(input_dir)
            output_path = output_dir / rel_path.with_suffix(".png")

            logger.info(f"[{i + 1}/{len(files)}] {file.name}")

            try:
                self.extract(file, output_path)
                results.append(output_path)
            except Exception as e:
                logger.error(f"  Failed: {e}")

        return results


class ComfyUIPoseExtractor:
    """ComfyUI를 통한 포즈 추출 (더 안정적).

    ComfyUI 서버에 워크플로우를 전송하여 포즈 추출.
    controlnet_aux 직접 설치 없이 사용 가능.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8188,
        resolution: int = 1024,
    ):
        self.host = host
        self.port = port
        self.resolution = resolution

    def _build_workflow(self, image_name: str, output_prefix: str) -> dict:
        """DWPose 추출 워크플로우 생성."""
        return {
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": image_name},
            },
            "2": {
                "class_type": "DWPreprocessor",
                "inputs": {
                    "image": ["1", 0],
                    "detect_hand": "enable",
                    "detect_body": "enable",
                    "detect_face": "enable",
                    "resolution": self.resolution,
                    "bbox_detector": "yolox_l.onnx",
                    "pose_estimator": "dw-ll_ucoco_384_bs5.torchscript.pt",
                },
            },
            "3": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["2", 0],
                    "filename_prefix": output_prefix,
                },
            },
        }

    async def extract(
        self,
        image_path: Path,
        output_prefix: str,
    ) -> Path:
        """ComfyUI를 통해 포즈 추출.

        Args:
            image_path: 입력 이미지 경로
            output_prefix: 출력 파일명 프리픽스

        Returns:
            생성된 포즈 이미지 경로
        """
        import aiohttp
        import requests

        # 1. 이미지 업로드
        with open(image_path, "rb") as f:
            files = {"image": (image_path.name, f, "image/png")}
            resp = requests.post(
                f"http://{self.host}:{self.port}/upload/image",
                files=files,
                data={"overwrite": "true"},
                timeout=30,
            )
            resp.raise_for_status()
            uploaded_name = resp.json()["name"]

        # 2. 워크플로우 실행
        workflow = self._build_workflow(uploaded_name, output_prefix)

        async with aiohttp.ClientSession() as session:
            # 큐에 추가
            async with session.post(
                f"http://{self.host}:{self.port}/prompt",
                json={"prompt": workflow},
            ) as resp:
                result = await resp.json()
                prompt_id = result["prompt_id"]

            # 완료 대기 (간단한 폴링)
            import asyncio
            for _ in range(60):  # 최대 60초
                async with session.get(
                    f"http://{self.host}:{self.port}/history/{prompt_id}"
                ) as resp:
                    history = await resp.json()
                    if prompt_id in history:
                        outputs = history[prompt_id]["outputs"]
                        if "3" in outputs:
                            filename = outputs["3"]["images"][0]["filename"]
                            return Path(f"output/{filename}")
                await asyncio.sleep(1)

        raise TimeoutError("Pose extraction timed out")

    async def extract_batch(
        self,
        input_dir: Path,
        output_dir: Path,
    ) -> list[Path]:
        """배치 추출 (ComfyUI 사용)."""
        input_dir = Path(input_dir)
        results = []

        extensions = (".jpg", ".jpeg", ".png", ".webp")
        files = [f for f in input_dir.iterdir() if f.suffix.lower() in extensions]

        for i, file in enumerate(sorted(files)):
            output_prefix = f"pose_{file.stem}"
            logger.info(f"[{i + 1}/{len(files)}] {file.name}")

            try:
                result = await self.extract(file, output_prefix)
                results.append(result)
            except Exception as e:
                logger.error(f"  Failed: {e}")

        return results


def generate_manifest(pose_dir: Path) -> dict:
    """포즈 디렉토리에서 manifest.yaml 생성.

    Args:
        pose_dir: 포즈 이미지 디렉토리

    Returns:
        manifest 딕셔너리
    """
    import yaml

    pose_dir = Path(pose_dir)
    poses = {}

    # 모든 PNG 파일 수집
    for png_file in pose_dir.rglob("*.png"):
        rel_path = png_file.relative_to(pose_dir)
        # 포즈 ID: 파일명 (확장자 제외)
        pose_id = png_file.stem

        # 카테고리가 있으면 포함
        if len(rel_path.parts) > 1:
            category = rel_path.parts[0]
            pose_id = f"{category}_{pose_id}"

        poses[pose_id] = str(rel_path)

    manifest = {"poses": poses}

    # manifest.yaml 저장
    manifest_path = pose_dir / "manifest.yaml"
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.dump(manifest, f, allow_unicode=True, default_flow_style=False)

    logger.info(f"Manifest generated: {manifest_path} ({len(poses)} poses)")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="ControlNet 포즈 이미지 추출기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 단일 이미지 추출
  python -m morae_pipeline.pose_extractor image.jpg -o standing/default.png

  # 디렉토리 배치 처리
  python -m morae_pipeline.pose_extractor ./references/ --batch

  # manifest.yaml 생성
  python -m morae_pipeline.pose_extractor --generate-manifest

  # ComfyUI 서버 사용 (더 안정적)
  python -m morae_pipeline.pose_extractor image.jpg --use-comfyui
        """,
    )

    parser.add_argument(
        "input",
        nargs="?",
        help="입력 이미지 또는 디렉토리",
    )
    parser.add_argument(
        "-o", "--output",
        help="출력 경로 (기본: presets/controlnet/)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="디렉토리 배치 처리 모드",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="출력 해상도 (기본: 1024)",
    )
    parser.add_argument(
        "--use-comfyui",
        action="store_true",
        help="ComfyUI 서버를 통해 추출 (더 안정적)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="ComfyUI 호스트 (기본: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8188,
        help="ComfyUI 포트 (기본: 8188)",
    )
    parser.add_argument(
        "--generate-manifest",
        action="store_true",
        help="기존 포즈 이미지에서 manifest.yaml 생성",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="상세 로깅",
    )

    args = parser.parse_args()

    # 로깅 설정
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # manifest 생성 모드
    if args.generate_manifest:
        pose_dir = Path(args.input) if args.input else DEFAULT_OUTPUT_DIR
        generate_manifest(pose_dir)
        return

    if not args.input:
        parser.error("입력 이미지 또는 디렉토리를 지정하세요")

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR

    if args.use_comfyui:
        # ComfyUI 사용
        import asyncio
        extractor = ComfyUIPoseExtractor(
            host=args.host,
            port=args.port,
            resolution=args.resolution,
        )
        if args.batch or input_path.is_dir():
            asyncio.run(extractor.extract_batch(input_path, output_path))
        else:
            asyncio.run(extractor.extract(input_path, output_path.stem))
    else:
        # 직접 추출
        extractor = PoseExtractor(resolution=args.resolution)

        if args.batch or input_path.is_dir():
            results = extractor.extract_batch(input_path, output_path)
            logger.info(f"Extracted {len(results)} poses")

            # manifest 자동 생성
            if results:
                generate_manifest(output_path)
        else:
            extractor.extract(input_path, output_path)


if __name__ == "__main__":
    main()
