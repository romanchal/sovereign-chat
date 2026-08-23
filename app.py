"""
Sovereign AI Workbench — Phase 0 + Phase 1 skeleton.
SIH26117 · MRPL

Run:
    uvicorn app:app --reload --host 127.0.0.1 --port 8000
Then open http://127.0.0.1:8000
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import egress
from ollama_client import OllamaClient
from registry import Registry
from router import Router

BASE = Path(__file__).parent

app = FastAPI(title="Sovereign AI Workbench", version="0.1.0")

registry = Registry()
router = Router(registry)
ollama = OllamaClient()

SYSTEM_PROMPT = (
    "You are an on-premise assistant for an industrial refinery. "
    "All data you see is confidential and never leaves this machine. "
    "Be precise and concise. When you are not certain, say so plainly "
    "rather than guessing."
)


class ChatRequest(BaseModel):
    message: str
    has_image: bool = False


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(BASE / "static" / "index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    health = await ollama.health()
    return {
        "ollama": health,
        "registry": registry.as_list(),
        "loaded": await ollama.loaded_models(),
    }


@app.get("/api/egress")
async def egress_status() -> dict[str, Any]:
    return egress.snapshot()


@app.get("/api/loaded")
async def loaded() -> dict[str, Any]:
    return {"loaded": await ollama.loaded_models()}


@app.post("/api/warm")
async def warm() -> dict[str, Any]:
    """Run this before you present. Never demo on a cold model."""
    tags = [m.ollama_tag for m in registry.models.values()]
    return {"warmed": await ollama.warm_all(tags)}


@app.post("/api/reload-registry")
async def reload_registry() -> dict[str, Any]:
    """Live-add a model: edit models.yaml, hit this, watch it appear."""
    registry.reload()
    return {"registry": registry.as_list()}


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    decision = router.classify(req.message, has_image=req.has_image)
    spec = registry.get(decision.model_id)

    async def stream():
        # Frame 1 — the routing decision, before any token is generated.
        # This is what the judge reads off the screen.
        yield json.dumps({"type": "routing", **decision.to_dict()}) + "\n"

        # Frame 2 — swap telemetry. Makes a model change legible.
        swap = await ollama.ensure_loaded(spec.ollama_tag)
        yield json.dumps({"type": "swap", **swap}) + "\n"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": req.message},
        ]
        try:
            async for frame in ollama.stream_chat(
                spec.ollama_tag, messages, spec.options
            ):
                if "token" in frame:
                    yield json.dumps({"type": "token", "text": frame["token"]}) + "\n"
                elif "stats" in frame:
                    yield json.dumps({"type": "stats", **frame["stats"]}) + "\n"
        except Exception as exc:
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"

        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
