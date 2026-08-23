"""
LanceDB-backed vector + metadata store for RAG chunks.

Everything lives on disk under data/index/. Sovereign by construction:
no server process, no network, no cloud bucket. Deleting data/index/
resets the corpus.

Retrieval lives in Phase 3b; this module just writes and lists for now.
"""
from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa

BASE = Path(__file__).parent
INDEX_DIR = BASE / "data" / "index"
UPLOADS_DIR = BASE / "data" / "uploads"
META_DB_PATH = BASE / "data" / "doc_meta.sqlite"
CHUNKS_TABLE = "chunks"
DOCS_TABLE = "documents"


@dataclass
class ChunkRow:
    id: str
    doc_id: str
    ordinal: int
    page: int
    text: str
    vector: list[float] = field(default_factory=list)


def _ensure_dirs() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def _chunks_schema(dim: int) -> pa.Schema:
    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("doc_id", pa.string()),
        pa.field("ordinal", pa.int32()),
        pa.field("page", pa.int32()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), list_size=dim)),
    ])


def _docs_schema() -> pa.Schema:
    return pa.schema([
        pa.field("doc_id", pa.string()),
        pa.field("filename", pa.string()),
        pa.field("mime", pa.string()),
        pa.field("bytes", pa.int64()),
        pa.field("pages", pa.int32()),
        pa.field("chunks", pa.int32()),
        pa.field("ingested_at", pa.float64()),
        pa.field("path", pa.string()),
    ])


class Store:
    def __init__(self) -> None:
        _ensure_dirs()
        self.db = lancedb.connect(str(INDEX_DIR))
        self._chunks_dim: int | None = None
        self._chunks = None
        if CHUNKS_TABLE in self.db.table_names():
            self._chunks = self.db.open_table(CHUNKS_TABLE)
            # Infer dim from existing schema so a restart matches ingested data.
            for f in self._chunks.schema:
                if f.name == "vector" and isinstance(f.type, pa.FixedSizeListType):
                    self._chunks_dim = f.type.list_size
                    break
        if DOCS_TABLE in self.db.table_names():
            self._docs = self.db.open_table(DOCS_TABLE)
        else:
            self._docs = self.db.create_table(DOCS_TABLE, schema=_docs_schema())

        # Sidecar SQLite for permission metadata. Kept out of LanceDB to avoid
        # its schema-evolution rough edges when we add new fields later.
        self._meta_conn = sqlite3.connect(str(META_DB_PATH), check_same_thread=False)
        self._meta_conn.execute("""
            CREATE TABLE IF NOT EXISTS doc_meta (
                doc_id TEXT PRIMARY KEY,
                department TEXT NOT NULL DEFAULT 'general',
                access_level TEXT NOT NULL DEFAULT 'all',
                uploaded_by TEXT
            )
        """)
        self._meta_conn.commit()

    # ------------------------------------------------------------------ ids

    @staticmethod
    def doc_id_for(path: Path, content: bytes) -> str:
        """Stable id: sha256 of file content, first 16 hex chars."""
        return hashlib.sha256(content).hexdigest()[:16]

    @staticmethod
    def chunk_id(doc_id: str, ordinal: int) -> str:
        return f"{doc_id}:{ordinal}"

    # ------------------------------------------------------------------ writes

    def _ensure_chunks_table(self, dim: int) -> None:
        if self._chunks is not None and self._chunks_dim == dim:
            return
        if self._chunks is not None and self._chunks_dim != dim:
            raise ValueError(
                f"vector dim mismatch: table has {self._chunks_dim}, ingest gave {dim}"
            )
        self._chunks = self.db.create_table(CHUNKS_TABLE, schema=_chunks_schema(dim))
        self._chunks_dim = dim

    def has_doc(self, doc_id: str) -> bool:
        try:
            hit = self._docs.search().where(f"doc_id = '{doc_id}'").limit(1).to_list()
            return bool(hit)
        except Exception:
            return False

    def add_document(
        self,
        doc_id: str,
        filename: str,
        mime: str,
        size_bytes: int,
        pages: int,
        chunk_count: int,
        stored_path: Path,
    ) -> None:
        self._docs.add([{
            "doc_id": doc_id,
            "filename": filename,
            "mime": mime,
            "bytes": size_bytes,
            "pages": pages,
            "chunks": chunk_count,
            "ingested_at": time.time(),
            "path": str(stored_path),
        }])

    def add_chunks(self, rows: list[ChunkRow]) -> None:
        if not rows:
            return
        dim = len(rows[0].vector)
        if dim == 0:
            raise ValueError("empty vector in chunk row")
        self._ensure_chunks_table(dim)
        self._chunks.add([{
            "id": r.id,
            "doc_id": r.doc_id,
            "ordinal": r.ordinal,
            "page": r.page,
            "text": r.text,
            "vector": r.vector,
        } for r in rows])

    # ------------------------------------------------------------------ reads

    def set_meta(
        self, doc_id: str, department: str, access_level: str, uploaded_by: str | None,
    ) -> None:
        self._meta_conn.execute(
            "INSERT OR REPLACE INTO doc_meta (doc_id, department, access_level, uploaded_by)"
            " VALUES (?,?,?,?)",
            (doc_id, department, access_level, uploaded_by),
        )
        self._meta_conn.commit()

    def _all_meta(self) -> dict[str, dict[str, Any]]:
        cur = self._meta_conn.execute(
            "SELECT doc_id, department, access_level, uploaded_by FROM doc_meta"
        )
        return {r[0]: {
            "department": r[1], "access_level": r[2], "uploaded_by": r[3],
        } for r in cur.fetchall()}

    def list_documents(self) -> list[dict[str, Any]]:
        try:
            rows = self._docs.search().limit(1000).to_list()
        except Exception:
            return []
        meta_by_id = self._all_meta()
        rows.sort(key=lambda r: r.get("ingested_at", 0), reverse=True)
        out = []
        for r in rows:
            m = meta_by_id.get(r["doc_id"], {})
            out.append({
                "doc_id": r["doc_id"],
                "filename": r["filename"],
                "mime": r["mime"],
                "bytes": r["bytes"],
                "pages": r["pages"],
                "chunks": r["chunks"],
                "ingested_at": r["ingested_at"],
                "department": m.get("department", "general"),
                "access_level": m.get("access_level", "all"),
                "uploaded_by": m.get("uploaded_by"),
            })
        return out

    def doc_lookup(self, doc_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch document metadata for a set of doc_ids (single query)."""
        if not doc_ids:
            return {}
        try:
            uniq = list(set(doc_ids))
            clause = " OR ".join(f"doc_id = '{d}'" for d in uniq)
            rows = self._docs.search().where(clause).limit(len(uniq)).to_list()
        except Exception:
            return {}
        return {r["doc_id"]: r for r in rows}

    def search(
        self,
        vector: list[float],
        k: int = 5,
        allowed_doc_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Vector search chunks. Optionally restrict to a doc_id allowlist."""
        if self._chunks is None or not vector:
            return []
        q = self._chunks.search(vector).limit(k)
        if allowed_doc_ids is not None:
            if not allowed_doc_ids:
                return []
            clause = " OR ".join(f"doc_id = '{d}'" for d in allowed_doc_ids)
            q = q.where(clause, prefilter=True)
        try:
            hits = q.to_list()
        except Exception:
            return []
        docs = self.doc_lookup([h["doc_id"] for h in hits])
        out: list[dict[str, Any]] = []
        for h in hits:
            meta = docs.get(h["doc_id"], {})
            out.append({
                "chunk_id": h.get("id"),
                "doc_id": h.get("doc_id"),
                "filename": meta.get("filename", "unknown"),
                "page": int(h.get("page") or 0),
                "text": h.get("text") or "",
                "score": float(h.get("_distance", 0.0)),
            })
        return out

    def chunk_count(self) -> int:
        if self._chunks is None:
            return 0
        try:
            return self._chunks.count_rows()
        except Exception:
            return 0
