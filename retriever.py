"""
RAG retrieval — query embedding + LanceDB vector search + prompt assembly.

Phase 3b baseline: vector-only. BM25 hybrid + BGE reranker land as follow-ups
without changing this module's caller contract.

Authorization boundary: `allowed_doc_ids` is a hard filter applied BEFORE
retrieval (LanceDB `where` prefilter). If the caller passes `[]` we return
no chunks — the LLM never sees content the user is not allowed to see. If
the caller passes `None` we search everything (used until RBAC ships).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Take-home limits. Small so the model isn't drowned in context and so the
# routing/citation frames stay legible in the UI.
DEFAULT_TOP_K = 5
# LanceDB returns L2 distance on nomic-embed-text's normalized vectors.
# Empirically anything above ~1.3 is off-topic; keep as a soft cut only for
# the "not found" prompt hint, never to hide chunks the caller asked for.
LOOSE_RELEVANCE_MAX_DISTANCE = 1.3


@dataclass
class RetrievalResult:
    hits: list[dict[str, Any]]     # from Store.search
    any_relevant: bool             # at least one hit under the distance floor


class Retriever:
    def __init__(self, store, ollama, embed_tag: str) -> None:
        self.store = store
        self.ollama = ollama
        self.embed_tag = embed_tag

    async def retrieve(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        allowed_doc_ids: list[str] | None = None,
    ) -> RetrievalResult:
        if not (query or "").strip():
            return RetrievalResult(hits=[], any_relevant=False)
        vectors = await self.ollama.embed(self.embed_tag, [query])
        if not vectors:
            return RetrievalResult(hits=[], any_relevant=False)
        hits = self.store.search(vectors[0], k=k, allowed_doc_ids=allowed_doc_ids)
        any_rel = any(h["score"] <= LOOSE_RELEVANCE_MAX_DISTANCE for h in hits)
        return RetrievalResult(hits=hits, any_relevant=any_rel)


def build_grounded_prompt(hits: list[dict[str, Any]]) -> str:
    """
    Compose the system prompt that turns the LLM into a document-grounded
    answerer. Callers append it to the model's system message. The 'AUTHORIZED
    CONTEXT' framing is deliberate: it tells the model the application has
    already made the access decision, so the model does not invent its own.
    """
    header = (
        "You are an on-premise assistant for MRPL. "
        "Answer using the AUTHORIZED CONTEXT below. "
        "The application has verified that the current user is authorized to see "
        "every excerpt shown — do not refuse on confidentiality grounds and do "
        "not invent an access policy of your own. "
        "Cite sources inline as [filename p.N]. "
        "If the answer is not contained in the excerpts, reply exactly: "
        "'Not found in the authorized documents.'"
    )
    if not hits:
        return header + "\n\nAUTHORIZED CONTEXT\n==================\n(no excerpts retrieved)\n=================="
    body_parts = ["AUTHORIZED CONTEXT", "=================="]
    for h in hits:
        tag = f"[{h['filename']} p.{h['page']}]" if h["page"] else f"[{h['filename']}]"
        body_parts.append(tag)
        body_parts.append(h["text"].strip())
        body_parts.append("")
    body_parts.append("==================")
    return header + "\n\n" + "\n".join(body_parts)
