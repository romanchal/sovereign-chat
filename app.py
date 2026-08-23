"""
Sovereign AI Workbench — Phase 0 + 1 + 2 skeleton.
SIH26117 · MRPL

Run:
    uvicorn app:app --reload --host 127.0.0.1 --port 8000
Then open http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


def _load_dotenv(path: Path) -> list[str]:
    """Tiny .env loader. Zero deps. utf-8-sig strips a Notepad BOM. Env vars
    already set in the process take precedence. Returns loaded keys."""
    if not path.exists():
        print(f"[dotenv] no .env at {path}")
        return []
    loaded: list[str] = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip().lstrip("﻿")
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val
            loaded.append(key)
    print(f"[dotenv] loaded {len(loaded)} key(s) from {path.name}: {loaded}")
    return loaded


_load_dotenv(Path(__file__).parent / ".env")

from fastapi import Depends, FastAPI, Form, HTTPException, Request, Response, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import egress
import ingest as ingest_mod
from audit import AuditLogger
from auth import (
    ACCESS_LEVELS, AuthContext, DEPARTMENTS, ROLES, SESSION_COOKIE,
    User, UserStore, allowed_doc_ids, current_user, issue_cookie,
    permits_doc, require_admin,
)
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

# Auth bootstrap. Fail loudly if the user store is empty and no bootstrap env
# vars are set — otherwise the app would silently reject every login.
users = UserStore()
_created = users.bootstrap_admin_from_env()
if _created:
    print(f"[auth] bootstrapped admin: {_created.email}")
if not users.all():
    raise RuntimeError(
        "no users configured. Create .env from .env.example with "
        "BOOTSTRAP_ADMIN_EMAIL / BOOTSTRAP_ADMIN_PASSWORD / BOOTSTRAP_ADMIN_NAME, "
        "then restart. (Or export those env vars in the current shell.)"
    )
AuthContext.store = users

EMBED_MODEL_ID = "embed"
VISION_MODEL_ID = "vision"
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
async def index(request: Request):
    # Guard the SPA at the edge so unauthenticated visitors never see it.
    if not request.cookies.get(SESSION_COOKIE):
        return RedirectResponse("/login", status_code=302)
    from auth import verify_cookie
    if not verify_cookie(request.cookies.get(SESSION_COOKIE, "")):
        return RedirectResponse("/login", status_code=302)
    return FileResponse(BASE / "static" / "index.html")


@app.get("/login")
async def login_page() -> FileResponse:
    return FileResponse(BASE / "static" / "login.html")


class LoginPayload(BaseModel):
    email: str
    password: str


@app.post("/api/login")
async def api_login(payload: LoginPayload, response: Response) -> dict[str, Any]:
    u = users.authenticate(payload.email, payload.password)
    if not u:
        audit.log("auth", payload.email, "login-fail", None)
        raise HTTPException(401, "invalid email or password")
    token = issue_cookie(u.email)
    response.set_cookie(
        SESSION_COOKIE, token,
        httponly=True, samesite="lax", max_age=12 * 3600, path="/",
    )
    audit.log("auth", u.email, "login-ok", {"role": u.role, "department": u.department})
    return {"email": u.email, "name": u.name, "role": u.role, "department": u.department}


@app.post("/api/logout")
async def api_logout(response: Response) -> dict[str, Any]:
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"ok": True}


@app.get("/api/me")
async def api_me(user: User = Depends(current_user)) -> dict[str, Any]:
    return {
        "email": user.email, "name": user.name,
        "role": user.role, "department": user.department,
        "is_admin": user.is_admin,
        "roles": list(ROLES), "departments": list(DEPARTMENTS),
        "access_levels": list(ACCESS_LEVELS),
    }


@app.get("/api/health")
async def health(user: User = Depends(current_user)) -> dict[str, Any]:
    return {
        "ollama": await ollama.health(),
        "registry": registry.as_list(),
        "loaded": await ollama.loaded_models(),
        "sandbox": {"available": sandbox.available, "error": sandbox.error},
    }


@app.get("/api/egress")
async def egress_status(user: User = Depends(current_user)) -> dict[str, Any]:
    return egress.snapshot()


@app.get("/api/loaded")
async def loaded(user: User = Depends(current_user)) -> dict[str, Any]:
    return {"loaded": await ollama.loaded_models()}


@app.post("/api/warm")
async def warm(user: User = Depends(current_user)) -> dict[str, Any]:
    tags = [m.ollama_tag for m in registry.models.values()]
    return {"warmed": await ollama.warm_all(tags)}


@app.post("/api/reload-registry")
async def reload_registry(user: User = Depends(require_admin)) -> dict[str, Any]:
    registry.reload()
    return {"registry": registry.as_list()}


@app.delete("/api/documents/{doc_id}")
async def delete_document(
    doc_id: str, user: User = Depends(current_user),
) -> dict[str, Any]:
    docs = store.list_documents()
    meta = next((d for d in docs if d["doc_id"] == doc_id), None)
    if not meta:
        raise HTTPException(404, "no such document")
    if not permits_doc(user, meta):
        raise HTTPException(403, "not authorized for this document")
    if not (user.is_admin or (meta.get("uploaded_by") or "") == user.email):
        raise HTTPException(403, "only the uploader or an admin can delete")

    # Remove the physical file (best-effort; the index removal is authoritative).
    for f in UPLOADS_DIR.glob(f"{doc_id}_*"):
        try:
            f.unlink()
        except Exception:
            pass
    store.delete_document(doc_id)
    audit.log("store", user.email, "document-deleted", {
        "doc_id": doc_id, "filename": meta.get("filename"),
    })
    return {"deleted": True, "doc_id": doc_id}


# ────────────────────────────────────────────────────────── admin endpoints

@app.get("/api/admin/users")
async def admin_list_users(user: User = Depends(require_admin)) -> dict[str, Any]:
    return {"users": [{
        "email": u.email, "name": u.name, "role": u.role,
        "department": u.department, "disabled": u.disabled,
    } for u in users.all()]}


@app.get("/api/documents")
async def list_documents(user: User = Depends(current_user)) -> dict[str, Any]:
    """List only documents the current user is allowed to see."""
    all_docs = store.list_documents()
    visible = [d for d in all_docs if permits_doc(user, d)]
    return {
        "documents": visible,
        "chunk_count": store.chunk_count(),
        "total_documents": len(all_docs),
    }


@app.post("/api/ingest")
async def ingest_file(
    file: UploadFile = File(...),
    department: str = Form("general"),
    access_level: str = Form("all"),
    user: User = Depends(current_user),
) -> dict[str, Any]:
    """Read a file, chunk it, embed with nomic-embed-text, persist to LanceDB."""
    if not registry.has(EMBED_MODEL_ID):
        raise HTTPException(500, f"embed model '{EMBED_MODEL_ID}' not in registry")
    embed_spec = registry.get(EMBED_MODEL_ID)

    department = (department or "general").lower()
    access_level = (access_level or "all").lower()
    if department not in DEPARTMENTS:
        raise HTTPException(400, f"unknown department: {department}")
    if access_level not in ACCESS_LEVELS:
        raise HTTPException(400, f"unknown access_level: {access_level}")
    # Employees can only upload to their own department (or 'general').
    if not user.is_admin and department not in (user.department, "general"):
        raise HTTPException(403, "cannot upload into another department")

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
        store.set_meta(doc_id, department, access_level, user.email)
        return {
            "doc_id": doc_id, "filename": filename, "reused": True,
            "chunks": 0, "pages": 0,
            "department": department, "access_level": access_level,
        }

    try:
        vision_needed = ingest_mod.needs_vision(stored)
        if vision_needed:
            if not registry.has(VISION_MODEL_ID):
                raise HTTPException(
                    422,
                    "this file needs the vision model (scanned/image), but "
                    "'vision' is not in the registry. Run: ollama pull qwen2.5vl:7b",
                )
            vl_spec = registry.get(VISION_MODEL_ID)
            await ollama.ensure_loaded(vl_spec.ollama_tag)

            async def _extract(b64: str) -> str:
                return await ollama.vl_extract_text(vl_spec.ollama_tag, b64, vl_spec.options)

            chunks, mime, pages = await ingest_mod.chunks_via_vision(stored, _extract)
        else:
            chunks, mime, pages = ingest_mod.chunks_for(stored)
    except HTTPException:
        raise
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
    store.set_meta(doc_id, department, access_level, user.email)
    audit.log(embed_spec.ollama_tag, filename, "ingest", {
        "doc_id": doc_id, "chunks": len(rows), "pages": pages,
        "department": department, "access_level": access_level,
        "uploaded_by": user.email, "vision": vision_needed,
    })
    return {
        "doc_id": doc_id, "filename": filename, "reused": False,
        "chunks": len(rows), "pages": pages, "mime": mime,
        "department": department, "access_level": access_level,
        "vision": vision_needed,
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
async def chat(req: ChatRequest, user: User = Depends(current_user)) -> StreamingResponse:
    decision = router.classify(req.message, has_image=req.has_image)
    spec = registry.get(decision.model_id)
    is_coder = decision.model_id == "coder" and sandbox.available

    # Compute the doc_id allowlist BEFORE retrieval — this is the RBAC gate.
    all_docs = store.list_documents()
    permitted_ids = set(allowed_doc_ids(user, all_docs))
    if req.doc_ids:
        requested = set(req.doc_ids)
        denied = requested - permitted_ids
        if denied:
            raise HTTPException(403, f"not authorized for doc(s): {sorted(denied)}")
        scope_ids: list[str] | None = sorted(requested)
    else:
        scope_ids = sorted(permitted_ids) if permitted_ids else []

    use_rag = False
    if retriever is not None and not is_coder:
        if req.grounded is True:
            use_rag = True
        elif req.grounded is None and store.chunk_count() > 0 and scope_ids:
            use_rag = True

    async def stream():
        yield json.dumps({"type": "routing", **decision.to_dict()}) + "\n"

        # Retrieve BEFORE the model swap so the citations frame lands early.
        retrieved: list[dict[str, Any]] = []
        any_relevant = False
        if use_rag:
            try:
                result = await retriever.retrieve(
                    req.message, k=RAG_TOP_K, allowed_doc_ids=scope_ids,
                )
                retrieved = result.hits
                any_relevant = result.any_relevant
                yield json.dumps({
                    "type": "citations",
                    "grounded": True,
                    "scope": len(scope_ids) if scope_ids is not None else -1,
                    "hits": [{
                        "doc_id": h["doc_id"], "filename": h["filename"],
                        "page": h["page"], "score": round(h["score"], 3),
                    } for h in retrieved],
                    "any_relevant": any_relevant,
                }) + "\n"
                audit.log(spec.ollama_tag, req.message, "retrieval", {
                    "user": user.email,
                    "k": RAG_TOP_K,
                    "returned": len(retrieved),
                    "any_relevant": any_relevant,
                    "requested_doc_ids": req.doc_ids,
                    "scope_ids": scope_ids,
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
