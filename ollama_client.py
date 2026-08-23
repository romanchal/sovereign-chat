"""
Ollama client + VRAM swap manager.

Your RTX 5050 holds one large model at a time. This module tracks what is
loaded, times the swap, and reports it — so a model change reads as
deliberate orchestration in the UI instead of an unexplained pause.
"""
from __future__ import annotations

import time
from typing import Any, AsyncIterator

import httpx

OLLAMA = "http://127.0.0.1:11434"

# How long Ollama keeps a model in VRAM after its last use.
# "30m" is right for a demo; you do not want an unload mid-presentation.
KEEP_ALIVE = "30m"


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA) -> None:
        self.base_url = base_url
        self._last_loaded: str | None = None

    # ---------------------------------------------------------------- health

    async def health(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(f"{self.base_url}/api/tags")
                r.raise_for_status()
                tags = [m["name"] for m in r.json().get("models", [])]
            return {"up": True, "installed": tags}
        except Exception as exc:
            return {"up": False, "error": str(exc), "installed": []}

    async def loaded_models(self) -> list[dict[str, Any]]:
        """What is currently sitting in VRAM (this is `ollama ps`)."""
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(f"{self.base_url}/api/ps")
                r.raise_for_status()
                out = []
                for m in r.json().get("models", []):
                    total = m.get("size", 0) or 0
                    gpu = m.get("size_vram", 0) or 0
                    out.append({
                        "name": m.get("name"),
                        "size_gb": round(total / 1e9, 2),
                        "vram_gb": round(gpu / 1e9, 2),
                        # 100 means fully on GPU. Anything less is spilling to CPU.
                        "gpu_pct": round(100 * gpu / total) if total else 0,
                    })
                return out
        except Exception:
            return []

    # ------------------------------------------------------------------ swap

    async def ensure_loaded(self, tag: str) -> dict[str, Any]:
        """
        Make sure `tag` is resident. Returns swap telemetry for the UI.

        An empty prompt tells Ollama to load the model and stop, which is
        the cheapest way to warm it.
        """
        loaded = {m["name"] for m in await self.loaded_models()}
        if tag in loaded:
            self._last_loaded = tag
            return {"swapped": False, "swap_ms": 0, "unloaded": None}

        previous = self._last_loaded
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=300) as c:
                await c.post(
                    f"{self.base_url}/api/generate",
                    json={"model": tag, "prompt": "", "keep_alive": KEEP_ALIVE},
                )
        except Exception:
            pass  # the chat call below will surface any real failure
        swap_ms = int((time.perf_counter() - started) * 1000)
        self._last_loaded = tag
        return {"swapped": True, "swap_ms": swap_ms, "unloaded": previous}

    async def warm_all(self, tags: list[str]) -> list[dict[str, Any]]:
        """Call this before you present. Never demo on a cold model."""
        results = []
        for tag in tags:
            info = await self.ensure_loaded(tag)
            results.append({"tag": tag, **info})
        return results

    # ------------------------------------------------------------------ chat

    async def stream_chat(
        self,
        tag: str,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield {'token': str} frames, then one {'stats': {...}} frame."""
        payload = {
            "model": tag,
            "messages": messages,
            "stream": True,
            "keep_alive": KEEP_ALIVE,
            "options": options or {},
        }
        async with httpx.AsyncClient(timeout=None) as c:
            async with c.stream("POST", f"{self.base_url}/api/chat", json=payload) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    import json as _json
                    try:
                        chunk = _json.loads(line)
                    except ValueError:
                        continue

                    piece = (chunk.get("message") or {}).get("content", "")
                    if piece:
                        yield {"token": piece}

                    if chunk.get("done"):
                        eval_count = chunk.get("eval_count") or 0
                        eval_ns = chunk.get("eval_duration") or 0
                        rate = (eval_count / (eval_ns / 1e9)) if eval_ns else 0.0
                        yield {"stats": {
                            "tokens": eval_count,
                            "tokens_per_sec": round(rate, 1),
                            "total_ms": int((chunk.get("total_duration") or 0) / 1e6),
                        }}
