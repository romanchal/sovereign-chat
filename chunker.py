"""
Recursive text chunker with page-aware splits.

Hand-rolled instead of pulling langchain-text-splitters, because that
package drags in tenacity, tiktoken, pyyaml pins, and a handful of
transitive deps we do not need. Twenty lines here beats fifty MB of wheels.

A chunk is roughly `target` characters. We try to cut on paragraph, then
sentence, then whitespace, never mid-word. Overlap is preserved so a
concept split across the cut is still retrievable from both chunks.
"""
from __future__ import annotations

from dataclasses import dataclass

# Rough approximation: 1 token ~ 4 chars for English/technical prose.
DEFAULT_TARGET_CHARS = 800 * 4     # ~800 tokens
DEFAULT_OVERLAP_CHARS = 120 * 4    # ~120 tokens

SPLIT_ORDER = ("\n\n", "\n", ". ", " ")


@dataclass
class Chunk:
    text: str
    page: int          # 1-indexed; 0 if source has no page concept
    ordinal: int       # position of the chunk inside its source


def _split_once(text: str, target: int) -> tuple[str, str]:
    """Take a prefix of `text` up to ~target chars, on a natural boundary."""
    if len(text) <= target:
        return text, ""
    window = text[: int(target * 1.1)]
    for sep in SPLIT_ORDER:
        idx = window.rfind(sep, target // 2)
        if idx > 0:
            cut = idx + len(sep)
            return text[:cut], text[cut:]
    # Fallback: hard cut at target (very rare — no whitespace found).
    return text[:target], text[target:]


def chunk_text(
    text: str,
    page: int = 0,
    target: int = DEFAULT_TARGET_CHARS,
    overlap: int = DEFAULT_OVERLAP_CHARS,
    start_ordinal: int = 0,
) -> list[Chunk]:
    """Split `text` into overlapping chunks, tagged with `page`."""
    text = (text or "").strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    ordinal = start_ordinal
    remaining = text
    while remaining:
        piece, rest = _split_once(remaining, target)
        piece = piece.strip()
        if piece:
            chunks.append(Chunk(text=piece, page=page, ordinal=ordinal))
            ordinal += 1
        if not rest:
            break
        # Carry the tail of `piece` forward as overlap so cross-cut context survives.
        tail = piece[-overlap:] if overlap and len(piece) > overlap else ""
        remaining = (tail + rest).strip()
    return chunks


def chunk_pages(pages: list[str]) -> list[Chunk]:
    """Chunk a list of page texts, preserving 1-indexed page numbers."""
    out: list[Chunk] = []
    ordinal = 0
    for i, page_text in enumerate(pages, start=1):
        page_chunks = chunk_text(page_text, page=i, start_ordinal=ordinal)
        out.extend(page_chunks)
        ordinal += len(page_chunks)
    return out
