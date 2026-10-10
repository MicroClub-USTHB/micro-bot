import pytest

from mcclub_rag.ingest import errors as e


def test_every_error_has_a_unique_code():
    codes = [cls.code for cls in e.ALL_ERRORS]
    assert len(codes) == len(set(codes))
    assert all(code and code == code.lower() for code in codes)


def test_all_errors_lists_every_subclass():
    def subclasses(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from subclasses(sub)

    assert set(subclasses(e.IngestError)) == set(e.ALL_ERRORS)


@pytest.mark.parametrize(
    ("child", "parent"),
    [
        (e.ParseTimeout, e.ParseError),
        (e.UrlNotAllowed, e.FetchError),
        (e.FileTooLarge, e.IngestError),
        (e.OcrConfigError, e.IngestError),
        (e.TokenizerConfigError, e.IngestError),
        (e.IndexUnavailable, e.IngestError),
        (e.IndexStale, e.IngestError),
        (e.EmbeddingError, e.IngestError),
        (e.EmbedderConfigError, e.IngestError),
        (e.RerankError, e.IngestError),
        (e.RerankerConfigError, e.IngestError),
    ],
)
def test_hierarchy(child, parent):
    assert issubclass(child, parent)


def test_message_and_code_exposed():
    err = e.UnsupportedFileType("unsupported file type: application/zip")
    assert err.code == "unsupported_file_type"
    assert err.message == "unsupported file type: application/zip"
    assert str(err) == err.message


def test_expected_codes():
    assert {cls.__name__: cls.code for cls in e.ALL_ERRORS} == {
        "InvalidInput": "invalid_input",
        "FileTooLarge": "file_too_large",
        "UnsupportedFileType": "unsupported_file_type",
        "ParseError": "parse_error",
        "ParseTimeout": "parse_timeout",
        "EmptyDocument": "empty_document",
        "FetchError": "fetch_error",
        "UrlNotAllowed": "url_not_allowed",
        "OcrConfigError": "ocr_config_error",
        "TokenizerConfigError": "tokenizer_config_error",
        "IndexUnavailable": "index_unavailable",
        "IndexStale": "index_stale",
        "EmbeddingError": "embedding_error",
        "EmbedderConfigError": "embedder_config_error",
        "RerankError": "rerank_error",
        "RerankerConfigError": "reranker_config_error",
    }
