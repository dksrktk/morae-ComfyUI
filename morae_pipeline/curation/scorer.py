"""Base class for image scorers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image


class ImageScorer(ABC):
    """Base class for all image scoring modules."""

    name: str = "base"
    weight: float = 1.0

    @abstractmethod
    def score(self, image: Image.Image) -> float:
        """Score an image. Returns 0.0 ~ 1.0."""
        ...

    def is_available(self) -> bool:
        """Check if this scorer's dependencies are installed."""
        return True

    def score_file(self, path: Path) -> float:
        """Convenience: score from file path."""
        img = Image.open(path).convert("RGB")
        return self.score(img)
