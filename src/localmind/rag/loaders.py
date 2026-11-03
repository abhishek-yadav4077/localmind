"""Turn uploaded bytes into LangChain Documents (one per PDF page / text file)."""

from __future__ import annotations

import io
from pathlib import PurePath

from langchain_core.documents import Document

from localmind.exceptions import LocalMindError

SUPPORTED = {".pdf", ".txt", ".md"}


class UnsupportedDocumentError(LocalMindError):
    """File type not supported or no extractable text."""


def load_bytes(filename: str, data: bytes) -> list[Document]:
    """Parse a PDF/TXT/MD upload. PDF pages keep their 1-based page number for citations."""
    suffix = PurePath(filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise UnsupportedDocumentError(
            f"Unsupported file type '{suffix}'",
            hint=f"Upload one of: {', '.join(sorted(SUPPORTED))}",
        )
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        docs = [
            Document(page_content=text, metadata={"source": filename, "page": i + 1})
            for i, page in enumerate(reader.pages)
            if (text := (page.extract_text() or "").strip())
        ]
    else:
        text = data.decode("utf-8", errors="replace").strip()
        docs = (
            [Document(page_content=text, metadata={"source": filename, "page": None})]
            if text
            else []
        )
    if not docs:
        raise UnsupportedDocumentError(
            f"No text found in '{filename}'",
            hint="Scanned PDFs are images; run OCR first (e.g. ocrmypdf) and upload the result",
        )
    return docs
