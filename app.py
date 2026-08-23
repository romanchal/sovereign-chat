"""
Sovereign AI Workbench — Phase 0 + 1 + 2 skeleton.
SIH26117 · MRPL

Run:
    uvicorn app:app --reload --host 127.0.0.1 --port 8000
Then open http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import egress
import ingest as ingest_mod
from audit import AuditLogger
from ollama_client import OllamaClient
from registry import Registry
from router import Router
from retriever import Retriever, build_grounded_prompt
from sandbox import SandboxExecutor, get_tools_schema
from store import ChunkRow, Store, UPLOADS_DIR

BASE = Path(__file__).parent

app = FastAPI(title="Sovereign AI Workbench", version="0.2.0")

registry = Registry()
router = Router(registry)
ollama = OllamaClient()
sandbox = SandboxExecutor()
audit = AuditLogger()
store = Store()

EMBED_MODEL_ID = "embed"
EMBED_BATCH = 32
RAG_TOP_K = 5

_embed_spec = registry.get(EMBED_MODEL_ID) if registry.has(EMBED_MODEL_ID) else None
retriever = Retriever(store, ollama, _embed_spec.ollama_tag) if _embed_spec else None

SYSTEM_PROMPT = (
    "You are an on-premise assistant for MRPL. "
    "Answer precisely; when unsure, say so plainly rather than guessing. "
    "The application (not you) makes access decisions — do not invent your own "
    "confidentiality policy or refuse to answer legitimate questions about data "
    "the application has already shown you."
)

CODER_SYSTEM_PROMPT = (
    "You are an on-premise coding agent. You have exactly one tool: `run_python`, "
    "which executes code in an air-gapped Docker sandbox with no network. "
    "When the user asks for code that produces a result, WRITE the code and CALL "
    "the tool to run it. Read the output. If it errors, fix and re-run. "
    "When you are done, reply in plain text with the final answer and a brief "
    "explanation. Never fabricate execution results."
)

AGENT_MAX_ITER = 3
FENCE_RE = re.compile(r"```python\s*\n(.*?)\n```", re.DOTALL)


class ChatRequest(BaseModel):
    message: str
    has_image: bool = False
    grounded: bool | None = None       # None = auto (RAG if any docs ingested)
    doc_ids: list[str] | None = None   # explicit document selection


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(BASE / "static" / "index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "ollama": await ollama.health(),
        "registry": registry.as_list(),
        "loaded": await ollama.loaded_models(),
        "sandbox": {"available": sandbox.available, "error": sandbox.error},
    }


@app.get("/api/egress")
async def egress_status() -> dict[str, Any]:
    return egress.snapshot()


@app.get("/api/loaded")
async def loaded() -> dict[str, Any]:
    return {"loaded": await ollama.loaded_models()}


@app.post("/api/warm")
async def warm() -> dict[str, Any]:
    tags = [m.ollama_tag for m in registry.models.values()]
    return {"warmed": await ollama.warm_all(tags)}


@app.post("/api/reload-registry")
async def reload_registry() -> dict[str, Any]:
    registry.reload()
    return {"registry": registry.as_list()}


@app.get("/api/documents")
async def list_documents() -> dict[str, Any]:
    return {"documents": store.list_documents(), "chunk_count": store.chunk_count()}


@app.post("/api/ingest")
async def ingest_file(file: UploadFile = File(...)) -> dict[str, Any]:
    """Read a file, chunk it, embed with nomic-embed-text, persist to LanceDB."""
    if not registry.has(EMBED_MODEL_ID):
        raise HTTPException(500, f"embed model '{EMBED_MODEL_ID}' not in registry")
    embed_spec = registry.get(EMBED_MODEL_ID)

    filename = Path(file.filename or "unnamed").name
    ext = Path(filename).suffix.lower()
    if ext not in ingest_mod.SUPPORTED_EXTS:
        raise HTTPException(415, f"unsupported file type: {ext or 'unknown'}")

    content = await file.read()
    if not content:
        raise HTTPException(400, "empty file")

    doc_id = store.doc_id_for(Path(filename), content)
    stored = UPLOADS_DIR / f"{doc_id}_{filename}"
    stored.write_bytes(content)

    if store.has_doc(doc_id):
        return {
            "doc_id": doc_id, "filename": filename, "reused": True,
            "chunks": 0, "pages": 0,
        }

    try:
        chunks, mime, pages = ingest_mod.chunks_for(stored)
    except Exception as exc:
        raise HTTPException(422, f"extraction failed: {exc}")

    if not chunks:
        raise HTTPException(422, "no extractable text in file")

    # Embed in batches to keep the request finite even for large PDFs.
    rows: list[ChunkRow] = []
    for i in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[i : i + EMBED_BATCH]
        vectors = await ollama.embed(embed_spec.ollama_tag, [c.text for c in batch])
        if len(vectors) != len(batch):
            raise HTTPException(500, "embedding count mismatch from ollama")
        for c, v in zip(batch, vectors):
            rows.append(ChunkRow(
                id=store.chunk_id(doc_id, c.ordinal),
                doc_id=doc_id, ordinal=c.ordinal, page=c.page,
                text=c.text, vector=v,
            ))

    store.add_chunks(rows)
    store.add_document(
        doc_id=doc_id, filename=filename, mime=mime,
        size_bytes=len(content), pages=pages, chunk_count=len(rows),
        stored_path=stored,
    )
    audit.log(embed_spec.ollama_tag, filename, "ingest", {
        "doc_id": doc_id, "chunks": len(rows), "pages": pages,
    })
    return {
        "doc_id": doc_id, "filename": filename, "reused": False,
        "chunks": len(rows), "pages": pages, "mime": mime,
    }


def _extract_code(msg: dict[str, Any]) -> tuple[str | None, str, bool]:
    """Return (code, tool_label, used_native_tool_call)."""
    calls = msg.get("tool_calls") or []
    if calls:
        call = calls[0]
        fn = (call.get("function") or {})
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        code = args.get("code")
        if code:
            return code, "run_python", True
    content = msg.get("content") or ""
    match = FENCE_RE.search(content)
    if match:
        return match.group(1).strip(), "run_python (fence)", False
    return None, "", False


async def _run_agent_loop(
    tag: str,
    options: dict[str, Any],
    messages: list[dict[str, Any]],
    user_prompt: str,
):
    """
    Yield NDJSON frames for the ReAct loop.

    Yields a final frame `{"type":"final_text","text":"..."}` when the model
    produced its final answer inside the loop — the caller then knows NOT to
    run another stream_chat pass (that would duplicate history and cause the
    model to return an almost-empty response).

    Yields `{"type":"needs_stream": true}` when the loop exits with no final
    text yet — the caller should stream_chat to produce one.
    """
    for iteration in range(AGENT_MAX_ITER):
        yield json.dumps({"type": "plan", "iteration": iteration + 1}) + "\n"

        msg = await ollama.chat_once(
            tag, messages,
            tools=get_tools_schema() if sandbox.available else None,
            options=options,
        )
        code, tool_label, used_native = _extract_code(msg)
        content = (msg.get("content") or "").strip()

        if not code:
            # Model chose to answer directly. Stream that text as tokens so the
            # UI renders it, then signal the caller to skip stream_chat.
            if content:
                yield json.dumps({"type": "token", "text": content}) + "\n"
            audit.log(tag, user_prompt, "final-planned", {"content": content})
            yield json.dumps({"type": "final_text", "done": True}) + "\n"
            return

        yield json.dumps({
            "type": "tool_call", "tool": tool_label, "code": code,
        }) + "\n"

        output = sandbox.run_python(code)
        audit.log(tag, user_prompt, "tool_call", {
            "tool": tool_label, "code": code, "output": output,
        })

        yield json.dumps({"type": "tool_output", "output": output}) + "\n"

        if used_native:
            messages.append(msg)
            messages.append({"role": "tool", "content": str(output)})
        else:
            messages.append({"role": "assistant", "content": msg.get("content", "")})
            messages.append({
                "role": "user",
                "content": (
                    "The sandbox ran your code. Output:\n"
                    f"{output}\n"
                    "Now give the final answer to the original question, in plain text. "
                    "Do not call the tool again."
                ),
            })

    yield json.dumps({"type": "needs_stream"}) + "\n"


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    decision = router.classify(req.message, has_image=req.has_image)
    spec = registry.get(decision.model_id)
    is_coder = decision.model_id == "coder" and sandbox.available

    # RAG decision: explicit grounded=True/False overrides; otherwise auto —
    # ground when we have any indexed chunks AND we're not in the coder path.
    use_rag = False
    if retriever is not None and not is_coder:
        if req.grounded is True:
            use_rag = True
        elif req.grounded is None and store.chunk_count() > 0:
            use_rag = True

    async def stream():
        yield json.dumps({"type": "routing", **decision.to_dict()}) + "\n"

        # Retrieve BEFORE the model swap so the citations frame lands early.
        retrieved: list[dict[str, Any]] = []
        any_relevant = False
        if use_rag:
            try:
                result = await retriever.retrieve(
                    req.message, k=RAG_TOP_K, allowed_doc_ids=req.doc_ids,
                )
                retrieved = result.hits
                any_relevant = result.any_relevant
                yield json.dumps({
                    "type": "citations",
                    "grounded": True,
                    "hits": [{
                        "doc_id": h["doc_id"], "filename": h["filename"],
                        "page": h["page"], "score": round(h["score"], 3),
                    } for h in retrieved],
                    "any_relevant": any_relevant,
                }) + "\n"
                audit.log(spec.ollama_tag, req.message, "retrieval", {
                    "k": RAG_TOP_K,
                    "returned": len(retrieved),
                    "any_relevant": any_relevant,
                    "doc_ids": req.doc_ids,
                })
            except Exception as exc:
                audit.log(spec.ollama_tag, req.message, "retrieval-error", {"error": str(exc)})
                yield json.dumps({"type": "error", "message": f"retrieval failed: {exc}"}) + "\n"

        swap = await ollama.ensure_loaded(spec.ollama_tag)
        yield json.dumps({"type": "swap", **swap}) + "\n"

        if is_coder:
            system = CODER_SYSTEM_PROMPT
        elif use_rag:
            system = build_grounded_prompt(retrieved)
        else:
            system = SYSTEM_PROMPT
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": req.message},
        ]
        audit.log(spec.ollama_tag, req.message, "route", {
            **decision.to_dict(), "use_rag": use_rag, "chunk_count": len(retrieved),
        })

        try:
            skip_stream = False
            if is_coder:
                async for frame in _run_agent_loop(
                    spec.ollama_tag, spec.options, messages, req.message,
                ):
                    # Peek at control frames without leaking them to the client.
                    try:
                        parsed = json.loads(frame)
                    except ValueError:
                        parsed = {}
                    ftype = parsed.get("type")
                    if ftype == "final_text":
                        skip_stream = True
                        continue
                    if ftype == "needs_stream":
                        continue
                    yield frame

            if not skip_stream:
                final_text = ""
                async for frame in ollama.stream_chat(
                    spec.ollama_tag, messages, spec.options
                ):
                    if "token" in frame:
                        final_text += frame["token"]
                        yield json.dumps({"type": "token", "text": frame["token"]}) + "\n"
                    elif "stats" in frame:
                        yield json.dumps({"type": "stats", **frame["stats"]}) + "\n"
                audit.log(spec.ollama_tag, req.message, "final", {"text": final_text})
        except Exception as exc:
            audit.log(spec.ollama_tag, req.message, "error", {"message": str(exc)})
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"

        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
