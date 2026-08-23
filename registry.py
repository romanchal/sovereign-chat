"""
Model registry — loads models.yaml and exposes it to the rest of the app.

The whole point: adding a model must never require editing Python.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REGISTRY_PATH = Path(__file__).parent / "models.yaml"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    ollama_tag: str
    label: str
    role: str
    vram_gb: float
    resident: bool
    options: dict[str, Any]


class Registry:
    def __init__(self, path: Path = REGISTRY_PATH) -> None:
        self.path = path
        self.reload()

    def reload(self) -> None:
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        self.models: dict[str, ModelSpec] = {}
        for entry in raw.get("models", []):
            spec = ModelSpec(
                id=entry["id"],
                ollama_tag=entry["ollama_tag"],
                label=entry.get("label", entry["ollama_tag"]),
                role=entry.get("role", ""),
                vram_gb=float(entry.get("vram_gb", 0)),
                resident=bool(entry.get("resident", False)),
                options=entry.get("options", {}) or {},
            )
            self.models[spec.id] = spec

        self.default_model: str = raw.get("default_model", "reasoning")
        self.vram_budget_gb: float = float(raw.get("vram_budget_gb", 7.2))

        if self.default_model not in self.models:
            # Fail loudly at startup rather than mysteriously at request time.
            raise ValueError(
                f"default_model '{self.default_model}' is not defined in {self.path.name}"
            )

    def get(self, model_id: str) -> ModelSpec:
        return self.models.get(model_id) or self.models[self.default_model]

    def has(self, model_id: str) -> bool:
        return model_id in self.models

    def as_list(self) -> list[dict[str, Any]]:
        return [
            {
                "id": m.id,
                "tag": m.ollama_tag,
                "label": m.label,
                "role": m.role,
                "vram_gb": m.vram_gb,
            }
            for m in self.models.values()
        ]
