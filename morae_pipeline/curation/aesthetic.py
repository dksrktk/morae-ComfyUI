"""Aesthetic scorer using CLIP + LAION aesthetic predictor MLP."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .scorer import ImageScorer

logger = logging.getLogger(__name__)

# Model will be loaded lazily
_model = None
_preprocess = None
_aesthetic_mlp = None
_device = None


def _load_models(device: str = "cuda") -> bool:
    """Load CLIP and aesthetic predictor. Returns True on success."""
    global _model, _preprocess, _aesthetic_mlp, _device

    if _model is not None:
        return True

    try:
        import torch
        import torch.nn as nn

        # Use open_clip for CLIP ViT-L/14
        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="openai"
            )
        except ImportError:
            # Fallback to transformers
            from transformers import CLIPModel, CLIPProcessor
            _clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

            class CLIPWrapper:
                def __init__(self, m, p):
                    self._model = m
                    self._processor = p
                def encode_image(self, x):
                    return self._model.get_image_features(pixel_values=x)
                def to(self, device):
                    self._model = self._model.to(device)
                    return self
                def eval(self):
                    self._model.eval()
                    return self

            model = CLIPWrapper(_clip_model, _clip_processor)
            preprocess = lambda img: _clip_processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

        _device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        model = model.to(_device).eval()
        _model = model
        _preprocess = preprocess

        # Aesthetic MLP: simple linear predictor on CLIP embeddings
        # LAION aesthetic predictor v2: 768 -> 128 -> 64 -> 16 -> 1
        class AestheticMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(768, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1),
                )

            def forward(self, x):
                return self.layers(x)

        mlp = AestheticMLP()

        # Try to load pretrained weights
        weights_path = Path(__file__).parent.parent.parent / "models" / "aesthetic" / "aesthetic_predictor_v2.pth"
        if weights_path.exists():
            state = torch.load(weights_path, map_location=_device, weights_only=True)
            mlp.load_state_dict(state)
            logger.info(f"Loaded aesthetic predictor from {weights_path}")
        else:
            logger.warning(
                f"Aesthetic predictor weights not found at {weights_path}. "
                "Scores will be based on CLIP embedding norm (less accurate). "
                "Download from: https://github.com/christophschuhmann/improved-aesthetic-predictor"
            )
            _aesthetic_mlp = None
            return True

        mlp = mlp.to(_device).eval()
        _aesthetic_mlp = mlp
        return True

    except Exception as e:
        logger.error(f"Failed to load aesthetic models: {e}")
        return False


class AestheticScorer(ImageScorer):
    """CLIP-based aesthetic score.

    Uses LAION aesthetic predictor v2 if weights available,
    falls back to CLIP embedding norm otherwise.

    NOTE: This measures "visual pleasantness" which is biased toward
    conventionally beautiful images. Use as a reference signal, not a
    hard filter. Intentionally ugly/dark characters will score lower
    and that's expected.
    """

    name = "aesthetic"
    weight = 0.3

    def __init__(self, device: str = "cuda"):
        self._device = device
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._available is None:
            self._available = _load_models(self._device)
        return self._available

    def score(self, image: Image.Image) -> float:
        if not self.is_available():
            return 0.5  # Neutral score if unavailable

        import torch

        image = image.convert("RGB")
        img_tensor = _preprocess(image).unsqueeze(0).to(_device)

        with torch.no_grad():
            embedding = _model.encode_image(img_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

            if _aesthetic_mlp is not None:
                # Predictor outputs ~1-10 range
                raw_score = _aesthetic_mlp(embedding).item()
                # Normalize to 0-1: map [1, 10] -> [0, 1]
                normalized = (raw_score - 1.0) / 9.0
            else:
                # Fallback: use embedding norm as rough proxy
                normalized = float(embedding.norm().item()) / 30.0

        return float(np.clip(normalized, 0.0, 1.0))
