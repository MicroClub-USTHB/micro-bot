"""Parsed-document store: ``DATA_DIR/documents/<doc_id>.json`` (ingest-pipeline Req 7.2, 7.3).

Keeps each indexed ``ParsedDocument`` so a reindex (new embedder or chunking settings) can
rebuild the points without the original file or URL: no re-parse, re-OCR or re-fetch.
Writes are atomic (temp file, fsync, rename), so a crash never leaves a truncated JSON.
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path

import structlog
from pydantic import ValidationError

from mcclub_rag.ingest.models import StoredDocument, Visibility

log = structlog.get_logger(__name__)


class DocStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, doc_id: str) -> Path:
        return self.root / f"{doc_id}.json"

    def _write(self, stored: StoredDocument) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        target = self._path(stored.doc_id)
        tmp = target.with_name(target.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(stored.model_dump_json())
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)

    def _read(self, doc_id: str) -> StoredDocument | None:
        path = self._path(doc_id)
        if not path.is_file():
            return None
        return StoredDocument.model_validate_json(path.read_bytes())

    def _read_all(self) -> tuple[list[StoredDocument], list[str]]:
        stored: list[StoredDocument] = []
        failed: list[str] = []
        if not self.root.is_dir():
            return stored, failed
        for path in sorted(self.root.glob("*.json")):
            try:
                stored.append(StoredDocument.model_validate_json(path.read_bytes()))
            except (OSError, ValidationError):
                log.warning("stored_document_unreadable", doc_id=path.stem)
                failed.append(path.stem)
        return stored, failed

    async def save(self, stored: StoredDocument) -> None:
        await asyncio.to_thread(self._write, stored)

    async def load(self, doc_id: str) -> StoredDocument | None:
        return await asyncio.to_thread(self._read, doc_id)

    async def load_all(self) -> tuple[list[StoredDocument], list[str]]:
        """Every readable document, sorted by ``doc_id``, plus the ids of unreadable files."""
        return await asyncio.to_thread(self._read_all)

    async def delete(self, doc_id: str) -> None:
        await asyncio.to_thread(self._path(doc_id).unlink, missing_ok=True)

    async def set_expiry(self, doc_id: str, expires_at: datetime | None) -> None:
        stored = await self.load(doc_id)
        if stored is not None:
            await self.save(stored.model_copy(update={"expires_at": expires_at}))

    async def set_visibility(self, doc_id: str, visibility: Visibility) -> None:
        stored = await self.load(doc_id)
        if stored is None:
            return
        document = stored.document.model_copy(update={"visibility": visibility})
        await self.save(stored.model_copy(update={"document": document}))
