"""Document identity: topic -> ``doc_id`` -> point ids (ingest-pipeline Req 2).

Pure and deterministic. ``NAMESPACE`` must never change: it would re-key every document,
so a re-upload would no longer replace its earlier version.
"""

import hashlib
import unicodedata
from pathlib import PurePath
from urllib.parse import urlsplit
from uuid import UUID, uuid5

from mcclub_rag.ingest.errors import InvalidInput

NAMESPACE = UUID("b558a996-46fd-4c6d-b008-fc40258e1b86")
MAX_TOPIC = 200
_HASH_CHARS = 16
_DEFAULT_PORTS = {"http": 80, "https": 443}


def clean_topic(topic: str) -> str:
    cleaned = unicodedata.normalize("NFC", topic.strip())
    if not cleaned:
        raise InvalidInput("topic must not be empty")
    if len(cleaned) > MAX_TOPIC:
        raise InvalidInput(f"topic longer than {MAX_TOPIC} characters")
    return cleaned


def _fit(topic: str) -> str:
    """Shorten an over-long *default* topic to a readable prefix plus a hash of the whole.

    Saved web pages often carry percent-encoded URLs as filenames (234 characters in
    data/parsed). The hash keeps the topic unique and stable, so re-uploads still replace.
    """
    topic = unicodedata.normalize("NFC", topic.strip())
    if len(topic) <= MAX_TOPIC:
        return topic
    digest = hashlib.sha256(topic.encode()).hexdigest()[:_HASH_CHARS]
    return f"{topic[: MAX_TOPIC - _HASH_CHARS - 1]}~{digest}"


def file_topic(filename: str) -> str:
    """Default topic of an upload: the filename without its last extension."""
    return clean_topic(_fit(PurePath(filename).stem))


def url_topic(url: str) -> str:
    """Default topic of a web page: the URL without credentials, query, fragment or default port."""
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    host = parts.hostname or ""
    port = f":{parts.port}" if parts.port and parts.port != _DEFAULT_PORTS.get(scheme) else ""
    path = parts.path.rstrip("/") or "/"
    return clean_topic(_fit(f"{scheme}://{host}{port}{path}"))


def doc_id_for(topic: str) -> str:
    return str(uuid5(NAMESPACE, topic))


def point_id(doc_id: str, content_hash: str, index: int) -> str:
    """Qdrant point id; includes the hash so two versions never overwrite each other."""
    return str(uuid5(UUID(doc_id), f"{content_hash}:{index}"))
