"""Drivers: the same requests against a running app (HTTP) or an in-process app (Req 8).

Both speak the public API through httpx; only the transport and the corpus setup differ. Each
chat request carries a fresh unknown ``conversation_id`` (``eval-<uuid>``): the chat route then
starts a new conversation with no history, without any API change (Req 5.3). Follow-up cases
need history in the server's store, which only the in-process driver can seed; the HTTP driver
reports them as unsupported.
"""

import asyncio
import os
import secrets
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx

from eval.e2e.backends import PipelineDocuments, Ready
from eval.e2e.cases import Case, CorpusDoc
from eval.e2e.score import Observation
from mcclub_rag.config import Settings
from mcclub_rag.conversation import Turn
from mcclub_rag.ingest.pipeline import IngestPipeline
from mcclub_rag.main import ApplicationState, RealLanguageDetector, create_app
from mcclub_rag.rate_limit import SlidingWindowRateLimiter
from mcclub_rag.retrieval.backend import HybridRetriever
from mcclub_rag.schemas import Retrieval


@dataclass(frozen=True)
class Source:
    """One corpus document to load: its manifest entry and the file holding its text."""

    doc: CorpusDoc
    path: Path


class _Api:
    """Requests through the public API with the role's headers."""

    supports_history = False

    def __init__(self, client: httpx.AsyncClient, member_token: str, admin_key: str) -> None:
        self.client = client
        self.member_token = member_token
        self.admin_key = admin_key

    def headers(self, role: str | None = None, admin: bool = False) -> dict[str, str]:
        headers = {}
        if role == "member":
            headers["Authorization"] = f"Bearer {self.member_token}"
        if admin:
            headers["X-Admin-Key"] = self.admin_key
        return headers

    async def request(self, method: str, path: str, **kwargs) -> httpx.Response:
        return await self.client.request(method, path, **kwargs)

    async def _conversation_id(self, case: Case) -> str:
        return f"eval-{uuid.uuid4()}"

    async def _wait(self) -> None:
        return None

    async def chat(self, case: Case, message: str | None = None) -> Observation:
        await self._wait()
        payload = {
            "message": case.question if message is None else message,
            "conversation_id": await self._conversation_id(case),
        }
        start = time.perf_counter()
        try:
            response = await self.client.post(
                "/api/v1/chat", json=payload, headers=self.headers(case.role)
            )
        except httpx.HTTPError as exc:
            return Observation(None, None, _ms(start), f"{type(exc).__name__}: {exc}")
        try:
            body = response.json()
        except ValueError:
            body = response.text
        return Observation(response.status_code, body, _ms(start))

    async def upload(self, source: Source, topic: str) -> dict:
        files = {"file": (source.path.name, source.path.read_bytes())}
        data = {"visibility": source.doc.visibility, "topic": topic}
        response = await self.client.post(
            "/api/v1/documents", files=files, data=data, headers=self.headers(admin=True)
        )
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        await self.client.aclose()


class _Captured:
    """The app's retrieval backend, keeping the last result for the scorer."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.last: Retrieval | None = None

    async def retrieve(self, query, role, history, *, rewrite=None) -> Retrieval:
        self.last = await self.inner.retrieve(query, role, history, rewrite=rewrite)
        return self.last


class InProcessDriver(_Api):
    """The app built in this process on the given pipeline, retriever and generator."""

    supports_history = True

    def __init__(
        self,
        pipeline: IngestPipeline,
        retriever: HybridRetriever,
        generator: object,
        data_dir: Path,
        state_overrides: dict | None = None,
    ) -> None:
        member, admin = secrets.token_hex(16), secrets.token_hex(16)
        settings = Settings(  # auth reads the app's settings: no environment needed (I20)
            member_token=member, admin_api_key=admin, data_dir=data_dir, llm_daily_limit=10_000
        )
        self.pipeline = pipeline
        self.retriever = retriever
        self.documents = PipelineDocuments(pipeline)
        fields = {
            "settings": settings,
            "documents": self.documents,
            "document_probe": Ready(self.documents),
            "provider_probe": Ready(),
            "retrieval": _Captured(retriever),
            "language_detector": RealLanguageDetector(),
            "generator": generator,
            "rate_limiter": SlidingWindowRateLimiter(100_000, 60),
        }
        self.state = ApplicationState(**(fields | (state_overrides or {})))
        self.app = create_app(settings=settings, state=self.state)
        # The app's JSON 500 handler must answer, not re-raise into the test client.
        transport = httpx.ASGITransport(app=self.app, raise_app_exceptions=False)
        client = httpx.AsyncClient(transport=transport, base_url="http://eval", timeout=120)
        super().__init__(client, member, admin)

    async def close(self) -> None:
        await super().close()

    async def chat(self, case: Case, message: str | None = None) -> Observation:
        captured = self.state.retrieval
        if isinstance(captured, _Captured):
            captured.last = None  # a blank message retrieves nothing
        observation = await super().chat(case, message)
        if isinstance(captured, _Captured):
            observation.retrieval = captured.last
        return observation

    async def setup(self, sources: Sequence[Source]) -> dict[str, str]:
        """Ingest each source with its manifest title and topic; doc_id -> topic."""
        topics = {}
        for source in sources:
            doc = source.doc
            result = await self.pipeline.ingest_file(
                source.path, source.path.name, doc.visibility, title=doc.title, topic=doc.topic
            )
            topics[result.doc_id] = doc.topic
        return topics

    async def _conversation_id(self, case: Case) -> str:
        if not case.history:
            return await super()._conversation_id(case)
        store = self.state.conversations
        conversation = store.create(case.lang)
        for turn in case.history:
            store.save(conversation.conversation_id, Turn(**turn.model_dump()), [])
        return conversation.conversation_id

    async def rewrite(self, question: str, history: Sequence[Turn]):
        """The app's query rewrite (None when the run has no ``--rewrite``)."""
        rewriter = self.state.rewriter
        if rewriter is None:
            return None
        previous = history[-1].user_message if history else None
        return (await rewriter.rewrite(self.state.generator, question, previous)).rewrite

    async def retrieve(self, case: Case) -> tuple[Retrieval, float]:
        history = [Turn(**t.model_dump()) for t in case.history]
        start = time.perf_counter()
        rewrite = await self.rewrite(case.question, history)
        retrieval = await self.retriever.retrieve(
            case.question, case.role, history, rewrite=rewrite
        )
        return retrieval, _ms(start)


class HttpDriver(_Api):
    """A running app (compose or uvicorn). Tokens come from MEMBER_TOKEN / ADMIN_API_KEY.

    Chat requests are paced to a **maximum offered rate** of ``requests_per_window`` per
    ``window_s``, shared by every caller of this driver: ``_wait`` reserves the next slot under a
    lock, so concurrent load tasks cannot read the same slot and burst. The achieved rate is still
    bounded by latency and by saturation on the server.
    """

    def __init__(self, base_url: str, *, requests_per_window: int = 20, window_s: float = 60):
        client = httpx.AsyncClient(base_url=base_url, timeout=120)
        member = os.environ.get("MEMBER_TOKEN", "")
        admin = os.environ.get("ADMIN_API_KEY", "")
        super().__init__(client, member, admin)
        self.interval = window_s / requests_per_window
        self._pace = asyncio.Lock()
        self._next = 0.0
        self.uploaded: list[str] = []

    async def _wait(self) -> None:
        """Pace chat requests under the server's per-IP limit (Req 5.4), one slot at a time."""
        async with self._pace:
            now = time.monotonic()
            if now < self._next:
                await asyncio.sleep(self._next - now)
            self._next = max(now, self._next) + self.interval

    async def setup(self, sources: Sequence[Source]) -> dict[str, str]:
        """Upload through the API under ``eval-`` topics (titles come from the files)."""
        topics = {}
        for source in sources:
            topic = source.doc.topic
            topic = topic if topic.startswith("eval-") else f"eval-{topic}"
            body = await self.upload(source, topic)
            topics[body["doc_id"]] = source.doc.topic
            self.uploaded.append(body["doc_id"])
        return topics

    async def cleanup(self) -> int:
        deleted = 0
        for doc_id in self.uploaded:
            response = await self.request(
                "DELETE", f"/api/v1/documents/{doc_id}", headers=self.headers(admin=True)
            )
            deleted += response.status_code == 200
        return deleted


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
