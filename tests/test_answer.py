from mcclub_rag.answer import build, extractive_fallback
from mcclub_rag.schemas import Chunk


def make_chunk(
    doc_id: str = "doc-1", title: str = "First document", text: str = "First chunk text"
) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}-chunk",
        doc_id=doc_id,
        title=title,
        text=text,
        score=0.9,
    )


def test_build_keeps_only_valid_cited_sources() -> None:
    chunks = [make_chunk(), make_chunk("doc-2", "Second document", "Second chunk text")]

    answer, sources, grounded = build("The answer is here [doc-2] [doc-1].", chunks)

    assert answer == "The answer is here [doc-2] [doc-1]."
    assert grounded is True
    assert [source.model_dump() for source in sources] == [
        {"doc_id": "doc-2", "title": "Second document", "snippet": "Second chunk text"},
        {"doc_id": "doc-1", "title": "First document", "snippet": "First chunk text"},
    ]


def test_build_removes_hallucinated_citations_and_is_not_grounded() -> None:
    chunks = [make_chunk()]

    answer, sources, grounded = build("Supported [doc-1], invented [not-retrieved].", chunks)

    assert answer == "Supported [doc-1], invented ."
    assert [source.doc_id for source in sources] == ["doc-1"]
    assert grounded is False


def test_build_requires_at_least_one_valid_citation() -> None:
    _, sources, grounded = build("An uncited answer.", [make_chunk()])

    assert sources == []
    assert grounded is False


def test_build_rejects_empty_citations_and_explicit_refusals() -> None:
    chunks = [make_chunk()]

    assert build("Answer [] [doc-1]", chunks)[2] is False
    assert build("Answer [doc-1]", chunks, refused=True)[2] is False


def test_build_recognizes_i_dont_know_in_supported_languages() -> None:
    chunks = [make_chunk()]

    assert build("I don't know [doc-1]", chunks)[2] is False
    assert build("Je ne sais pas [doc-1]", chunks)[2] is False
    assert build("لا أعرف [doc-1]", chunks)[2] is False


def test_build_limits_source_snippets_to_500_characters() -> None:
    long_text = "x" * 501

    _, sources, grounded = build("Answer [doc-1]", [make_chunk(text=long_text)])

    assert grounded is True
    assert len(sources[0].snippet) == 500
    assert sources[0].snippet == long_text[:500]


def test_extractive_fallback_returns_verbatim_top_two_chunks() -> None:
    chunks = [
        make_chunk("doc-1", "First document", "Top chunk one."),
        make_chunk("doc-2", "Second document", "Top chunk two."),
        make_chunk("doc-3", "Third document", "This chunk is not included."),
    ]

    answer, sources, grounded = extractive_fallback(chunks)

    assert answer == "Top chunk one.\n\nTop chunk two."
    assert [source.doc_id for source in sources] == ["doc-1", "doc-2"]
    assert grounded is True


def test_extractive_fallback_without_chunks_is_not_grounded() -> None:
    assert extractive_fallback([]) == ("", [], False)
