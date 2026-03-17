"""Workflow template loading and parameterization."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Optional


class WorkflowTemplate:
    """Load an API-format ComfyUI workflow and parameterize it."""

    def __init__(self, data: dict):
        self._original = data
        self._data = copy.deepcopy(data)
        self._node_map: dict[str, str] = {}  # role -> node_id mapping
        self._auto_detect_nodes()

    @classmethod
    def from_file(cls, path: Path) -> WorkflowTemplate:
        with open(path) as f:
            data = json.load(f)
        return cls(data)

    @classmethod
    def from_dict(cls, data: dict) -> WorkflowTemplate:
        return cls(data)

    def _auto_detect_nodes(self) -> None:
        """Auto-detect common node roles by class_type."""
        for node_id, node in self._data.items():
            class_type = node.get("class_type", "")
            inputs = node.get("inputs", {})

            if class_type == "KSampler":
                self._node_map["sampler"] = node_id
            elif class_type == "CheckpointLoaderSimple":
                self._node_map["checkpoint"] = node_id
            elif class_type == "EmptyLatentImage":
                self._node_map["latent"] = node_id
            elif class_type == "SaveImage":
                self._node_map["save"] = node_id
            elif class_type == "CLIPTextEncode":
                # Heuristic: positive prompt usually has non-negative text
                text = inputs.get("text", "").lower()
                if any(neg in text for neg in ["bad", "worst", "low quality", "ugly", "negative"]):
                    self._node_map.setdefault("negative", node_id)
                else:
                    self._node_map.setdefault("positive", node_id)

    def set_param(self, node_id: str, field: str, value: Any) -> WorkflowTemplate:
        """Set a specific input parameter on a node."""
        if node_id not in self._data:
            raise KeyError(f"Node '{node_id}' not found in workflow")
        self._data[node_id]["inputs"][field] = value
        return self

    def set_prompt(self, positive: str, negative: Optional[str] = None) -> WorkflowTemplate:
        """Set text prompts using auto-detected CLIPTextEncode nodes."""
        if "positive" in self._node_map:
            self._data[self._node_map["positive"]]["inputs"]["text"] = positive
        else:
            raise KeyError("No positive CLIPTextEncode node detected")

        if negative is not None and "negative" in self._node_map:
            self._data[self._node_map["negative"]]["inputs"]["text"] = negative

        return self

    def set_seed(self, seed: int) -> WorkflowTemplate:
        if "sampler" in self._node_map:
            self._data[self._node_map["sampler"]]["inputs"]["seed"] = seed
        else:
            raise KeyError("No KSampler node detected")
        return self

    def set_dimensions(self, width: int, height: int) -> WorkflowTemplate:
        if "latent" in self._node_map:
            self._data[self._node_map["latent"]]["inputs"]["width"] = width
            self._data[self._node_map["latent"]]["inputs"]["height"] = height
        else:
            raise KeyError("No EmptyLatentImage node detected")
        return self

    def set_filename_prefix(self, prefix: str) -> WorkflowTemplate:
        if "save" in self._node_map:
            self._data[self._node_map["save"]]["inputs"]["filename_prefix"] = prefix
        return self

    def set_checkpoint(self, ckpt_name: str) -> WorkflowTemplate:
        if "checkpoint" in self._node_map:
            self._data[self._node_map["checkpoint"]]["inputs"]["ckpt_name"] = ckpt_name
        else:
            raise KeyError("No CheckpointLoaderSimple node detected")
        return self

    @property
    def node_map(self) -> dict[str, str]:
        """Return detected node role -> node_id mapping."""
        return dict(self._node_map)

    def reset(self) -> WorkflowTemplate:
        """Reset to original loaded state."""
        self._data = copy.deepcopy(self._original)
        return self

    def to_dict(self) -> dict:
        """Return the parameterized workflow as a dict ready for API submission."""
        return copy.deepcopy(self._data)

    def validate(self) -> list[str]:
        """Basic validation. Returns list of warnings."""
        warnings = []
        if not self._data:
            warnings.append("Workflow is empty")
        if "sampler" not in self._node_map:
            warnings.append("No KSampler node found")
        if "save" not in self._node_map:
            warnings.append("No SaveImage node found")
        if "positive" not in self._node_map:
            warnings.append("No positive prompt node detected")
        return warnings
