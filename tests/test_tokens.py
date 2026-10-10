import hashlib

import pytest

from mcclub_rag.ingest.errors import TokenizerConfigError
from mcclub_rag.ingest.settings import IngestSettings
from mcclub_rag.ingest.tokens import HFTokenCounter, get_token_counter, verify_tokenizer

SAMPLES = {
    "ar": "أعلن النادي عن موعد الهاكاثون؟ التسجيل مفتوح حتى يوم الجمعة؛ لا تتأخروا. " * 20,
    "fr": "Le délai d'inscription au hackathon est fixé à vendredi, 18 h. " * 20,
    "code": "--sidebar-primary: oklch(0.205 0 0); .dark { --chart-1: oklch(0.488 0.243 264); }\n"
    * 20,
}


def test_missing_tokenizer_raises_with_instruction(tmp_path):
    settings = IngestSettings(chunk_tokenizer_path=tmp_path / "nope" / "tokenizer.json")
    with pytest.raises(TokenizerConfigError, match="--only tokenizer"):
        verify_tokenizer(settings)


def test_empty_tokenizer_file_is_rejected(tmp_path):
    path = tmp_path / "tokenizer.json"
    path.write_bytes(b"")
    with pytest.raises(TokenizerConfigError):
        verify_tokenizer(IngestSettings(chunk_tokenizer_path=path))


def test_corrupt_tokenizer_file_is_rejected(tmp_path):
    path = tmp_path / "tokenizer.json"
    path.write_text("{not json")
    with pytest.raises(TokenizerConfigError, match="unreadable"):
        HFTokenCounter(path)


@pytest.fixture(scope="module")
def counter(tokenizer_path):
    return HFTokenCounter(tokenizer_path)


def test_embedded_truncation_is_disabled(counter):
    # granite's tokenizer.json ships with truncation at 32,768 tokens
    text = SAMPLES["ar"] * 80  # about 45k tokens
    assert counter.count(text) > 32_768


def test_batch_counts_match_single_counts(counter):
    # padding is enabled in the shipped file; batch counts would otherwise be inflated
    texts = ["court", SAMPLES["fr"], SAMPLES["ar"], ""]
    assert counter.count_many(texts) == [counter.count(t) for t in texts]
    assert counter.count("") == 0


@pytest.mark.parametrize("kind", sorted(SAMPLES))
@pytest.mark.parametrize("limit", [1, 17, 100])
def test_cut_is_lossless_and_within_limit(counter, kind, limit):
    text = SAMPLES[kind]
    head, tail = counter.cut(text, limit)
    assert head + tail == text
    assert head
    assert counter.count(head) <= limit


def test_cut_of_short_text_returns_everything(counter):
    assert counter.cut("bonjour", 50) == ("bonjour", "")


def test_fingerprint_is_file_sha256(counter, tokenizer_path):
    assert counter.fingerprint == hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()


def test_get_token_counter_is_cached(tokenizer_path):
    settings = IngestSettings(chunk_tokenizer_path=tokenizer_path)
    assert get_token_counter(settings) is get_token_counter(settings)
