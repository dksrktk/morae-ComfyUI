"""Output organization: sort images into grade folders with reports."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from .curation.pipeline import ImageScore

logger = logging.getLogger(__name__)


class OutputManager:
    """Organize scored images into grade-based folder structure."""

    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
        self.grades_dir = session_dir / "graded"

    def organize(self, scores: list[ImageScore]) -> dict[str, int]:
        """Copy images into grade subfolders. Returns grade counts."""
        counts = {"A": 0, "B": 0, "C": 0, "rejected": 0}

        for grade in counts:
            (self.grades_dir / grade).mkdir(parents=True, exist_ok=True)

        for item in scores:
            src = Path(item.path)
            if not src.exists():
                logger.warning(f"Source image not found: {src}")
                continue

            dest_dir = self.grades_dir / item.grade
            dest = dest_dir / src.name
            shutil.copy2(src, dest)
            counts[item.grade] += 1

        logger.info(
            f"Organized {sum(counts.values())} images: "
            f"A={counts['A']} B={counts['B']} C={counts['C']} rejected={counts['rejected']}"
        )
        return counts

    def write_scores(self, scores: list[ImageScore]) -> Path:
        """Write full scoring data as JSON."""
        path = self.session_dir / "scores.json"
        data = [s.to_dict() for s in scores]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        logger.info(f"Scores written to {path}")
        return path

    def write_summary(self, scores: list[ImageScore], counts: dict[str, int]) -> Path:
        """Write human-readable summary."""
        path = self.session_dir / "summary.txt"

        total = sum(counts.values())
        lines = [
            f"=== Morae Pipeline Curation Summary ===",
            f"Total images: {total}",
            f"",
            f"Grade distribution:",
            f"  A (best):    {counts.get('A', 0):4d}  ({counts.get('A', 0) / max(total, 1) * 100:.1f}%)",
            f"  B (good):    {counts.get('B', 0):4d}  ({counts.get('B', 0) / max(total, 1) * 100:.1f}%)",
            f"  C (usable):  {counts.get('C', 0):4d}  ({counts.get('C', 0) / max(total, 1) * 100:.1f}%)",
            f"  Rejected:    {counts.get('rejected', 0):4d}  ({counts.get('rejected', 0) / max(total, 1) * 100:.1f}%)",
            f"",
        ]

        # Top 10 by composite score
        top = sorted(scores, key=lambda s: s.composite, reverse=True)[:10]
        if top:
            lines.append("Top 10 images:")
            for i, s in enumerate(top, 1):
                name = Path(s.path).name
                detail = " | ".join(f"{k}={v:.3f}" for k, v in s.scores.items())
                lines.append(f"  {i:2d}. [{s.grade}] {s.composite:.3f}  {name}  ({detail})")
            lines.append("")

        # Bottom 5
        bottom = sorted(scores, key=lambda s: s.composite)[:5]
        if bottom:
            lines.append("Bottom 5 images:")
            for i, s in enumerate(bottom, 1):
                name = Path(s.path).name
                lines.append(f"  {i:2d}. [{s.grade}] {s.composite:.3f}  {name}")

        path.write_text("\n".join(lines))
        logger.info(f"Summary written to {path}")
        return path
