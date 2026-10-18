"""Reuse a grounded answer for the same first question on the same corpus: 0 LLM calls.

Club questions repeat ("how do I join?"), and every free provider has a daily cap. Only a
grounded LLM answer to a message without history is kept; the key holds the corpus version, so
any document change (upload, replace, delete, expiry) makes old entries unreachable.
"""

from collections import OrderedDict
from threading import Lock

from micro_bot.schemas import Source


def cache_key(message: str, language: str, corpus_version: str) -> tuple[str, str, str]:
    return (" ".join(message.casefold().split()), language, corpus_version)


class AnswerCache:
    def __init__(self, size: int) -> None:
        self.size = size
        self._items: OrderedDict[tuple[str, str, str], tuple[str, list[Source]]] = OrderedDict()
        self._lock = Lock()

    def get(self, key: tuple[str, str, str]) -> tuple[str, list[Source]] | None:
        with self._lock:
            hit = self._items.get(key)
            if hit is not None:
                self._items.move_to_end(key)
            return hit

    def put(self, key: tuple[str, str, str], answer: str, sources: list[Source]) -> None:
        if self.size <= 0:
            return
        with self._lock:
            self._items[key] = (answer, list(sources))
            self._items.move_to_end(key)
            while len(self._items) > self.size:
                self._items.popitem(last=False)
