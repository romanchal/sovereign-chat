"""
File ingest — read any supported file, return (chunks, mime, page_count).

Dispatch is by extension. Each extractor returns a list[str] where each
element is one "page" of text; formats without pages (docx, xlsx, pptx,
txt) collapse to a single page or one page per slide/sheet.

OCR + vision extractors land in Phase 3c. This module handles the native-
text formats first so the rest of the pipeline (chunk → embed → store)
can be wired end to end today.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from chunker import Chunk, chunk_pages, chunk_text

SUPPORTED_EXTS = {".pdf", ".txt", ".md", ".docx", ".xlsx", ".pptx"}


def _extract_pdf(path: Path) -> list[str]:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages: list[str] = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return pages


def _extract_txt(path: Path) -> list[str]:
    return [path.read_text(encoding="utf-8", errors="replace")]


def _extract_docx(path: Path) -> list[str]:
    from docx import Document
    doc = Document(str(path))
    parts: list[str] = []
    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text)
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return ["\n".join(parts)]


def _extract_xlsx(path: Path) -> list[str]:
    from openpyxl import load_workbook
    wb = load_workbook(str(path), data_only=True, read_only=True)
    sheets: list[str] = []
    for sheet in wb.worksheets:
        rows: list[str] = [f"# Sheet: {sheet.title}"]
        for row in sheet.iter_rows(values_only=True):
            cells = ["" if v is None else str(v) for v in row]
            if any(c.strip() for c in cells):
                rows.append(" | ".join(cells))
        sheets.append("\n".join(rows))
    return sheets  # one "page" per sheet


def _extract_pptx(path: Path) -> list[str]:
    from pptx import Presentation
    prs = Presentation(str(path))
    slides: list[str] = []
    for i, slide in enumerate(prs.slides, start=1):
        parts = [f"# Slide {i}"]
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                parts.append(shape.text)
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text
            if notes.strip():
                parts.append("Notes: " + notes)
        slides.append("\n".join(parts))
    return slides  # one "page" per slide


EXTRACTORS: dict[str, Callable[[Path], list[str]]] = {
    ".pdf": _extract_pdf,
    ".txt": _extract_txt,
    ".md": _extract_txt,
    ".docx": _extract_docx,
    ".xlsx": _extract_xlsx,
    ".pptx": _extract_pptx,
}


def extract(path: Path) -> tuple[list[str], str]:
    """Return (list_of_page_texts, mime_hint)."""
    ext = path.suffix.lower()
    if ext not in EXTRACTORS:
        raise ValueError(f"unsupported file type: {ext}")
    return EXTRACTORS[ext](path), ext.lstrip(".")


def chunks_for(path: Path) -> tuple[list[Chunk], str, int]:
    """Return (chunks, mime_hint, page_count) for a file."""
    pages, mime = extract(path)
    if len(pages) <= 1:
        # No native pagination; keep page=0 so callers can render "n/a".
        text = pages[0] if pages else ""
        return chunk_text(text, page=0), mime, 1 if text.strip() else 0
    return chunk_pages(pages), mime, len(pages)
