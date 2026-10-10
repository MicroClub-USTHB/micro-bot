"""Token counting with the embedder's own tokenizer (chunking Req 6).

Chunk sizes are measured in embedder tokens, not characters: Arabic runs at about 2.9
characters per granite token against 4.6 for English prose, so character budgets would make
Arabic chunks much larger than French or English ones.

granite's ``tokenizer.json`` ships with truncation at 32,768 tokens and padding enabled.
Both are switched off here, otherwise long texts under-count and batch counts over-count.
"""

import hashlib
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from tokenizers import Tokenizer

from mcclub_rag.ingest.errors import TokenizerConfigError
from mcclub_rag.ingest.settings import IngestSettings

_INSTRUCTION = "uv run python scripts/download_models.py --only tokenizer --dest {directory}"


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...

    def count_many(self, texts: Sequence[str]) -> list[int]: ...

    def cut(self, text: str, max_tokens: int) -> tuple[str, str]:
        """Split ``text`` so the head holds at most ``max_tokens``; ``head + tail == text``."""
        ...

    @property
    def fingerprint(self) -> str: ...


class HFTokenCounter:
    def __init__(self, path: Path) -> None:
        try:
            tokenizer = Tokenizer.from_file(str(path))
        except Exception as exc:  # tokenizers raises plain Exception for bad files
            raise TokenizerConfigError(
                f"tokenizer at {path} is unreadable ({exc}). "
                f"Run: {_INSTRUCTION.format(directory=path.parent)}"
            ) from exc
        tokenizer.no_truncation()
        tokenizer.no_padding()
        self._tokenizer = tokenizer
        self._fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def count_many(self, texts: Sequence[str]) -> list[int]:
        if not texts:
            return []
        encodings = self._tokenizer.encode_batch(list(texts), add_special_tokens=False)
        return [len(encoding.ids) for encoding in encodings]

    def cut(self, text: str, max_tokens: int) -> tuple[str, str]:
        offsets = self._tokenizer.encode(text, add_special_tokens=False).offsets
        if len(offsets) <= max_tokens:
            return text, ""
        limit = max_tokens
        while True:
            end = max((stop for _, stop in offsets[: max(limit, 1)]), default=0)
            end = max(end, 1)  # always make progress
            head = text[:end]
            # Re-tokenizing the head can merge differently at the edge; shrink until it fits.
            if limit <= 1 or self.count(head) <= max_tokens:
                return head, text[end:]
            limit -= 1


def verify_tokenizer(settings: IngestSettings) -> Path:
    path = settings.chunk_tokenizer_path
    if not path.is_file() or path.stat().st_size == 0:
        raise TokenizerConfigError(
            f"embedder tokenizer missing or empty at {path}. "
            f"Run: {_INSTRUCTION.format(directory=path.parent)}"
        )
    return path


@lru_cache
def _counter_for(path: Path) -> HFTokenCounter:
    return HFTokenCounter(path)


def get_token_counter(settings: IngestSettings) -> HFTokenCounter:
    """One loaded tokenizer per path, reused across documents."""
    return _counter_for(verify_tokenizer(settings))
