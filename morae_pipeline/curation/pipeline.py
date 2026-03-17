"""Curation pipeline: orchestrate scorers, grade images."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from PIL import Image

from ..config import CurationConfig
from .scorer import ImageScorer
from .technical import TechnicalQualityScorer
from .aesthetic import AestheticScorer
from .face import FaceSimilarityScorer

logger = logging.getLogger(__name__)

Grade = Literal["A", "B", "C", "rejected"]


@dataclass
class ImageScore:
    path: str
    scores: dict[str, float] = field(default_factory=dict)
    composite: float = 0.0
    grade: Grade = "rejected"

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "scores": self.scores,
            "composite": round(self.composite, 4),
            "grade": self.grade,
        }


class CurationPipeline:
    """Run multiple scorers on images and assign grades."""

    def __init__(self, config: CurationConfig):
        self.config = config
        self.scorers: list[ImageScorer] = []
        self._setup_scorers()

    def _setup_scorers(self) -> None:
        device = self.config.device

        if self.config.technical_enabled:
            scorer = TechnicalQualityScorer()
            scorer.weight = self.config.technical_weight
            self.scorers.append(scorer)

        if self.config.aesthetic_enabled:
            scorer = AestheticScorer(device=device)
            scorer.weight = self.config.aesthetic_weight
            if scorer.is_available():
                self.scorers.append(scorer)
            else:
                logger.warning("Aesthetic scorer unavailable, skipping")

        if self.config.face_enabled and self.config.face_reference_path:
            scorer = FaceSimilarityScorer(
                reference_path=self.config.face_reference_path,
                device=device,
            )
            scorer.weight = self.config.face_weight
            if scorer.is_available():
                self.scorers.append(scorer)
            else:
                logger.warning("Face scorer unavailable, skipping")

        active = [s.name for s in self.scorers]
        logger.info(f"Active scorers: {active}")

        # Normalize weights so they sum to 1.0
        total_weight = sum(s.weight for s in self.scorers)
        if total_weight > 0 and self.scorers:
            for s in self.scorers:
                s.weight = s.weight / total_weight

    def _grade(self, composite: float) -> Grade:
        if composite >= self.config.grade_a_min:
            return "A"
        elif composite >= self.config.grade_b_min:
            return "B"
        elif composite >= self.config.grade_c_min:
            return "C"
        else:
            return "rejected"

    def score_image(self, path: Path) -> ImageScore:
        """Score a single image."""
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to open {path}: {e}")
            return ImageScore(path=str(path), grade="rejected")

        scores = {}
        composite = 0.0

        for scorer in self.scorers:
            try:
                s = scorer.score(img)
                scores[scorer.name] = round(s, 4)
                composite += s * scorer.weight
            except Exception as e:
                logger.warning(f"Scorer '{scorer.name}' failed on {path}: {e}")
                scores[scorer.name] = 0.0

        grade = self._grade(composite)

        result = ImageScore(
            path=str(path),
            scores=scores,
            composite=composite,
            grade=grade,
        )

        logger.debug(f"{path.name}: {scores} -> {composite:.3f} ({grade})")
        return result

    def score_batch(self, paths: list[Path]) -> list[ImageScore]:
        """Score a batch of images."""
        results = []
        for i, path in enumerate(paths):
            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"Scoring {i + 1}/{len(paths)}...")
            results.append(self.score_image(path))

        # Summary
        grades = {"A": 0, "B": 0, "C": 0, "rejected": 0}
        for r in results:
            grades[r.grade] += 1
        logger.info(
            f"Curation complete: {len(results)} images -> "
            f"A:{grades['A']} B:{grades['B']} C:{grades['C']} rejected:{grades['rejected']}"
        )
        return results
