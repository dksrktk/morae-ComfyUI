"""ControlNet 포즈 이미지 라이브러리."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# 기본 라이브러리 경로
DEFAULT_LIBRARY_PATH = Path(__file__).parent / "presets" / "controlnet"


class ControlNetLibrary:
    """ControlNet 포즈 이미지 라이브러리.

    포즈 ID와 OpenPose 스켈레톤 이미지를 매핑하여 관리.
    manifest.yaml에서 매핑 정보를 로드.

    사용법:
        lib = ControlNetLibrary()
        pose_path = lib.get_pose_image("default")
        if pose_path:
            # ControlNet에 포즈 이미지 전달
    """

    def __init__(self, base_path: Optional[Path | str] = None):
        """라이브러리 초기화.

        Args:
            base_path: 라이브러리 기본 경로 (None이면 기본값 사용)
        """
        self.base_path = Path(base_path) if base_path else DEFAULT_LIBRARY_PATH
        self.manifest: dict[str, str] = {}
        self._load_manifest()

    def _load_manifest(self) -> None:
        """manifest.yaml 로드."""
        manifest_file = self.base_path / "manifest.yaml"

        if not manifest_file.exists():
            logger.warning(f"ControlNet manifest not found: {manifest_file}")
            return

        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            self.manifest = data.get("poses") or {}
            logger.info(f"ControlNet library loaded: {len(self.manifest)} poses")
        except Exception as e:
            logger.error(f"Failed to load ControlNet manifest: {e}")
            self.manifest = {}

    def get_pose_image(self, pose_id: str) -> Optional[Path]:
        """포즈 ID로 이미지 경로 반환.

        Args:
            pose_id: 포즈 ID (예: "default", "smile", "attack")

        Returns:
            이미지 경로 (없으면 None)
        """
        if pose_id not in self.manifest:
            return None

        image_path = self.base_path / self.manifest[pose_id]

        if not image_path.exists():
            logger.warning(f"Pose image not found: {image_path}")
            return None

        return image_path

    def has_pose(self, pose_id: str) -> bool:
        """포즈가 라이브러리에 있는지 확인."""
        return pose_id in self.manifest

    def list_poses(self) -> list[str]:
        """사용 가능한 포즈 ID 목록."""
        return list(self.manifest.keys())

    def get_poses_by_category(self, category: str) -> list[str]:
        """카테고리별 포즈 ID 목록.

        Args:
            category: 카테고리 (예: "standing", "combat", "sitting")

        Returns:
            해당 카테고리의 포즈 ID 목록
        """
        prefix = f"{category}/"
        return [
            pose_id for pose_id, path in self.manifest.items()
            if path.startswith(prefix)
        ]


def resolve_controlnet_image(
    pose_id: str,
    reference_image: Optional[str],
    cn_library: Optional[ControlNetLibrary] = None,
) -> tuple[Optional[str], bool]:
    """ControlNet 이미지 소스 결정 (하이브리드 로직).

    1. 라이브러리에 포즈가 있으면 → 라이브러리 이미지 사용
    2. 없으면 → 레퍼런스 이미지에서 DWPose로 추출

    Args:
        pose_id: 포즈 ID
        reference_image: 레퍼런스 이미지 경로 (IP-Adapter용)
        cn_library: ControlNet 라이브러리 (None이면 기본값)

    Returns:
        (image_path_or_ref, needs_extraction)
        - 라이브러리에 있으면: (library_path, False)
        - 없으면: (reference_image, True) → DWPose 런타임 추출 필요
        - 둘 다 없으면: (None, False) → ControlNet 사용 안 함
    """
    if cn_library is None:
        cn_library = ControlNetLibrary()

    # 1. 라이브러리에서 찾기
    library_path = cn_library.get_pose_image(pose_id)
    if library_path and library_path.exists():
        logger.debug(f"Using library pose: {pose_id} -> {library_path}")
        return str(library_path), False

    # 2. 레퍼런스 이미지에서 추출
    if reference_image:
        logger.debug(f"Pose '{pose_id}' not in library, will extract from reference")
        return reference_image, True

    # 3. ControlNet 사용 불가
    logger.debug(f"No ControlNet source for pose '{pose_id}'")
    return None, False
