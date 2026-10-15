from dataclasses import dataclass, field
from pathlib import Path

from fastapi.testclient import TestClient

from mcclub_rag.config import Settings
from mcclub_rag.llm.client import GenerationResult, GenerationUsage
from mcclub_rag.main import (
    ApplicationState,
    FakeDocumentBackend,
    FakeGenerationService,
    FakeHealthProbe,
    FakeLanguageDetector,
    create_app,
)
from mcclub_rag.schemas import Chunk, Retrieval, Role


@dataclass
class RoleAwareRetrieval:
    public_chunks: list[Chunk]
    member_chunks: list[Chunk]
    roles: list[Role] = field(default_factory=list)

    async def retrieve(self, query: str, role: Role, history: object, rewrite=None) -> Retrieval:
        self.roles.append(role)
        chunks = self.member_chunks if role == "member" else self.public_chunks
        return Retrieval(chunks=chunks, top_score=0.9, decision="answer")


def test_public_role_is_passed_to_the_boundary_and_never_returns_member_chunks(
    tmp_path: Path,
) -> None:
    public_chunk = Chunk(
        chunk_id="public-chunk",
        doc_id="public-doc",
        title="Public guide",
        text="The public workshop is Saturday.",
        score=0.9,
    )
    member_chunk = Chunk(
        chunk_id="member-chunk",
        doc_id="member-doc",
        title="Members guide",
        text="Member-only meeting is Sunday.",
        score=0.9,
    )
    retrieval = RoleAwareRetrieval([public_chunk], [member_chunk])
    state = ApplicationState(
        settings=Settings(data_dir=tmp_path, member_token="member-secret"),
        documents=FakeDocumentBackend(),
        document_probe=FakeHealthProbe(),
        provider_probe=FakeHealthProbe(),
        retrieval=retrieval,
        language_detector=FakeLanguageDetector(),
        generator=FakeGenerationService(
            GenerationResult("The public workshop is Saturday. [public-doc]", GenerationUsage(1))
        ),
    )

    with TestClient(create_app(settings=state.settings, state=state)) as client:
        response = client.post("/api/v1/chat", json={"message": "When is the workshop?"})

    assert response.status_code == 200
    assert retrieval.roles == ["public"]
    assert response.json()["sources"] == [
        {
            "doc_id": "public-doc",
            "title": "Public guide",
            "snippet": "The public workshop is Saturday.",
        }
    ]
    assert "member-doc" not in response.text
