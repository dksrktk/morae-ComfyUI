"""Pose library: categorized pose images with random selection and jittering."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Standard pose categories
POSE_CATEGORIES = [
    "standing",
    "sitting",
    "lying",
    "kneeling",
    "action",
    "pov",
    "complex",
]


@dataclass
class PoseEntry:
    """A single pose image with metadata."""

    path: str
    category: str = "uncategorized"
    tags: list[str] = field(default_factory=list)
    description: str = ""
    # Recommended ControlNet settings for this pose
    recommended_strength: float = 0.7
    recommended_end_percent: float = 0.4

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "category": self.category,
            "tags": self.tags,
            "description": self.description,
            "recommended_strength": self.recommended_strength,
            "recommended_end_percent": self.recommended_end_percent,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PoseEntry:
        return cls(**d)


class PoseLibrary:
    """Manage a collection of pose images organized by category.

    Directory structure:
        poses_dir/
        ├── manifest.json        (optional, with metadata)
        ├── standing/
        │   ├── stand_01.png
        │   └── stand_02.png
        ├── pov/
        │   ├── kiss_pov_01.png
        │   └── kiss_pov_02.png
        └── complex/
            ├── pose_a_01.png
            └── pose_a_02.png
    """

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

    def __init__(self, poses_dir: str | Path):
        self.poses_dir = Path(poses_dir)
        self.poses: dict[str, list[PoseEntry]] = {}
        self._load()

    def _load(self) -> None:
        """Load poses from directory structure and/or manifest."""
        if not self.poses_dir.exists():
            logger.warning(f"Pose library directory not found: {self.poses_dir}")
            return

        manifest_path = self.poses_dir / "manifest.json"
        if manifest_path.exists():
            self._load_manifest(manifest_path)
        else:
            self._scan_directory()

        total = sum(len(v) for v in self.poses.values())
        categories = list(self.poses.keys())
        logger.info(
            f"Pose library loaded: {total} poses in {len(categories)} categories "
            f"({', '.join(categories)})"
        )

    def _scan_directory(self) -> None:
        """Auto-discover poses from subdirectory structure."""
        for item in sorted(self.poses_dir.iterdir()):
            if item.is_dir():
                category = item.name
                entries = []
                for img in sorted(item.iterdir()):
                    if img.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                        entries.append(
                            PoseEntry(path=str(img), category=category)
                        )
                if entries:
                    self.poses[category] = entries

        # Also pick up any loose images in root as "uncategorized"
        root_images = [
            f
            for f in sorted(self.poses_dir.iterdir())
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]
        if root_images:
            self.poses["uncategorized"] = [
                PoseEntry(path=str(f), category="uncategorized")
                for f in root_images
            ]

    def _load_manifest(self, path: Path) -> None:
        """Load from manifest.json with full metadata."""
        data = json.loads(path.read_text())
        for entry_data in data.get("poses", []):
            entry = PoseEntry.from_dict(entry_data)
            # Resolve relative paths
            if not Path(entry.path).is_absolute():
                entry.path = str(self.poses_dir / entry.path)
            cat = entry.category
            self.poses.setdefault(cat, []).append(entry)

    @property
    def categories(self) -> list[str]:
        return list(self.poses.keys())

    @property
    def total_poses(self) -> int:
        return sum(len(v) for v in self.poses.values())

    def get_poses(
        self,
        category: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> list[PoseEntry]:
        """Get poses filtered by category and/or tags."""
        if category:
            candidates = self.poses.get(category, [])
        else:
            candidates = [p for poses in self.poses.values() for p in poses]

        if tags:
            tag_set = set(tags)
            candidates = [
                p for p in candidates if tag_set.intersection(p.tags)
            ]

        return candidates

    def pick_random(
        self,
        category: Optional[str] = None,
        tags: Optional[list[str]] = None,
        count: int = 1,
        rng: Optional[random.Random] = None,
    ) -> list[PoseEntry]:
        """Randomly select poses from the library.

        Args:
            category: Filter by category
            tags: Filter by tags
            count: Number of poses to pick
            rng: Random number generator (for reproducibility)

        Returns:
            List of randomly selected PoseEntry objects
        """
        candidates = self.get_poses(category=category, tags=tags)
        if not candidates:
            logger.warning(
                f"No poses found for category={category}, tags={tags}"
            )
            return []

        r = rng or random
        count = min(count, len(candidates))
        return r.sample(candidates, count)

    def save_manifest(self) -> Path:
        """Save current library state as manifest.json."""
        self.poses_dir.mkdir(parents=True, exist_ok=True)
        path = self.poses_dir / "manifest.json"
        all_poses = [p for poses in self.poses.values() for p in poses]
        data = {"poses": [p.to_dict() for p in all_poses]}
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        logger.info(f"Manifest saved: {path} ({len(all_poses)} poses)")
        return path
