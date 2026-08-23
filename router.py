"""
Intent router — decides WHICH model handles a request, and explains why.

Deliberately rule-based to begin with. Rules are fast, deterministic and
debuggable, which is what you want while everything else is in flux.
Swap in embedding similarity later; the interface below will not change.

MRPL's stated minimum is auto-selection across at least two task types.
With qwen3 + qwen2.5-coder you can satisfy that requirement today.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any

# Signals that a request is about writing or fixing code.
CODE_PATTERNS = [
    r"\bcode\b", r"\bscript\b", r"\bfunction\b", r"\bdebug\b", r"\brefactor\b",
    r"\bpython\b", r"\bjavascript\b", r"\bsql\b", r"\bregex\b", r"\bapi\b",
    r"\bclass\b", r"\bmethod\b", r"\bcompile\b", r"\bstack ?trace\b",
    r"\bexception\b", r"\btraceback\b", r"\bunit test\b", r"\bpytest\b",
    r"\bwrite a (?:program|parser|scraper|cli)\b", r"\bfix (?:this|the) bug\b",
]

# Signals that a request is about a picture or a scan.
VISION_PATTERNS = [
    r"\bdrawing\b", r"\bp&id\b", r"\bpid\b", r"\bdiagram\b", r"\bscan(?:ned)?\b",
    r"\bimage\b", r"\bphoto\b", r"\bhandwrit", r"\bblueprint\b", r"\bscreenshot\b",
]

CODE_RE = re.compile("|".join(CODE_PATTERNS), re.IGNORECASE)
VISION_RE = re.compile("|".join(VISION_PATTERNS), re.IGNORECASE)

# Fenced code blocks are a very strong code signal on their own.
FENCE_RE = re.compile(r"```")


@dataclass
class RoutingDecision:
    model_id: str
    ollama_tag: str
    label: str
    confidence: float
    reason: str
    signals: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Router:
    def __init__(self, registry) -> None:
        self.registry = registry

    def classify(self, text: str, has_image: bool = False) -> RoutingDecision:
        signals: list[str] = []

        # --- Vision wins outright: an attached image cannot be read by a text model.
        if has_image:
            signals.append("image attached")
            if self.registry.has("vision"):
                return self._decide("vision", 0.99, "an image was attached", signals)
            # Vision model not pulled yet — say so honestly instead of failing silently.
            return self._decide(
                self.registry.default_model, 0.30,
                "an image was attached but no vision model is registered", signals,
            )

        code_hits = CODE_RE.findall(text or "")
        vision_hits = VISION_RE.findall(text or "")
        fenced = bool(FENCE_RE.search(text or ""))

        # --- Code path
        if fenced or code_hits:
            if fenced:
                signals.append("fenced code block")
            if code_hits:
                signals.append(f"{len(code_hits)} code keyword(s)")
            confidence = min(0.95, 0.62 + 0.09 * len(code_hits) + (0.18 if fenced else 0))
            if self.registry.has("coder"):
                return self._decide("coder", confidence, "code generation intent", signals)

        # --- Vision-by-description path (user talks about a drawing but attached nothing)
        if vision_hits and self.registry.has("vision"):
            signals.append(f"{len(vision_hits)} visual-document keyword(s)")
            return self._decide(
                "vision", min(0.88, 0.58 + 0.10 * len(vision_hits)),
                "document or drawing understanding", signals,
            )

        # --- Default: general reasoning
        signals.append("no specialist signal")
        return self._decide(
            self.registry.default_model, 0.55,
            "general reasoning or document work", signals,
        )

    def _decide(self, model_id: str, confidence: float, reason: str,
                signals: list[str]) -> RoutingDecision:
        spec = self.registry.get(model_id)
        return RoutingDecision(
            model_id=spec.id,
            ollama_tag=spec.ollama_tag,
            label=spec.label,
            confidence=round(confidence, 2),
            reason=reason,
            signals=signals,
        )
