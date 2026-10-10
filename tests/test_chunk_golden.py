import json
import time

import pytest

from mcclub_rag.ingest.chunk import chunk_document
from mcclub_rag.ingest.models import Section
from mcclub_rag.ingest.settings import IngestSettings
from tests.chunk_helpers import MCOLI_GOLDEN, doc, mcoli_chunk_summary


@pytest.fixture
def settings(tessdata_dir, tokenizer_path):
    return IngestSettings(tessdata_prefix=tessdata_dir, chunk_tokenizer_path=tokenizer_path)


async def test_mcoli_docs_page_matches_golden(settings):
    summary = await mcoli_chunk_summary(settings)
    assert summary == json.loads(MCOLI_GOLDEN.read_text())
    assert all(tokens <= settings.chunk_max_tokens for _, _, tokens in summary)


PARAGRAPHS = [
    "أعلن النادي عن فتح باب التسجيل في الهاكاثون السنوي. يمكن للطلبة المشاركة في فرق من ثلاثة.",
    "Le club organise un atelier de découverte chaque jeudi. L'inscription est gratuite.",
    "Members can borrow the club laptops for projects. Ask the HR lead before Friday.",
]


def test_large_mixed_document_chunks_quickly(tokenizer_path):
    settings = IngestSettings(chunk_tokenizer_path=tokenizer_path)
    sections = tuple(
        Section(
            text=f"## Section {i}\n" + "\n\n".join([PARAGRAPHS[i % 3]] * (1 + i % 7)),
            heading=f"Section {i}",
            page=1 + i // 3,
            language=("ar", "fr", "en")[i % 3],
        )
        for i in range(300)
    )
    chunk_document(doc(sections[0]), settings=settings)  # load the tokenizer outside the timing
    start = time.perf_counter()
    chunks = chunk_document(doc(*sections, language="mixed"), settings=settings)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"chunking took {elapsed:.2f}s"
    assert all(c.token_count <= settings.chunk_max_tokens for c in chunks)
    assert {c.language for c in chunks} == {"ar", "fr", "en"}
