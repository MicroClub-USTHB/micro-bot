import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from onnxruntime.capi import onnxruntime_pybind11_state as ort_state
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from mcclub_rag.ingest.chunk import chunk_document
from mcclub_rag.ingest.errors import RerankerConfigError, RerankError
from mcclub_rag.ingest.settings import IngestSettings
from mcclub_rag.retrieval.registry import RerankerSpec
from mcclub_rag.retrieval.rerank import OnnxReranker, get_reranker, passage
from mcclub_rag.retrieval.settings import RetrievalSettings
from tests.chunk_helpers import FakeCounter, doc, md, words

SETTINGS = IngestSettings(
    chunk_target_tokens=30, chunk_max_tokens=40, chunk_min_tokens=8, chunk_max_header_tokens=8
)


def test_passage_is_what_the_chunk_was_embedded_with():
    document = doc(
        *md("# Statuts", f"## Adhésion\n\n{words(20)}", f"## Cotisation\n\n{words(25, 'dinar')}"),
        title="Règlement",
    )
    chunks = chunk_document(document, settings=SETTINGS, counter=FakeCounter())
    assert len(chunks) >= 2
    for chunk in chunks:
        assert passage(document.title, chunk.heading_path, chunk.text) == chunk.embed_text


def test_consecutive_repeats_collapse_case_insensitively():
    assert passage("Theming", ["theming", "Overview"], "body") == "Theming > Overview\n\nbody"


def test_blank_entries_are_skipped():
    assert passage("", [" ", "Section"], "body") == "Section\n\nbody"
    assert passage("", [], "body") == "body"


def test_header_off_is_the_text_alone():
    assert passage("Doc", ["Section"], "body", header=False) == "body"


VOCAB = {"[UNK]": 0, "<s>": 1, "</s>": 2, "<pad>": 3, "q": 4, "w": 5, "x": 6}
ONNX = b"fake onnx bytes"


def _write_model(directory):
    """A tiny pair tokenizer with padding switched on (like gte's) and fake ONNX bytes."""
    tok = Tokenizer(models.WordLevel(VOCAB, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> </s> $B </s>",
        special_tokens=[("<s>", 1), ("</s>", 2)],
    )
    tok.enable_padding(length=32, pad_id=3, pad_token="<pad>")
    (directory / "onnx").mkdir(parents=True)
    tok.save(str(directory / "tokenizer.json"))
    (directory / "onnx" / "model_int8.onnx").write_bytes(ONNX)
    return RerankerSpec(
        repo="org/fake-reranker",
        revision="r1",
        onnx_file="onnx/model_int8.onnx",
        onnx_sha256=hashlib.sha256(ONNX).hexdigest(),
        tokenizer_sha256=hashlib.sha256((directory / "tokenizer.json").read_bytes()).hexdigest(),
    )


class FakeSession:
    """Logit = (number of 'w' tokens) - 2; records every feed."""

    def __init__(self, inputs=("input_ids", "attention_mask"), error=None):
        self.names = inputs
        self.feeds = []
        self.error = error

    def get_inputs(self):
        return [SimpleNamespace(name=n) for n in self.names]

    def run(self, outputs, feed):
        if self.error:
            raise self.error
        self.feeds.append(feed)
        ids = feed["input_ids"]
        return [np.array([[float((ids == VOCAB["w"]).sum()) - 2.0]], dtype=np.float32)]


@pytest.fixture
def model(tmp_path):
    directory = tmp_path / "fake-reranker"
    return _write_model(directory), directory


def test_one_run_per_pair_with_batch_size_one(model):
    spec, directory = model
    session = FakeSession()
    scores = OnnxReranker(spec, directory, session=session).score("q", ["w", "w w w", "x"])
    assert len(session.feeds) == 3
    assert all(f["input_ids"].shape[0] == 1 for f in session.feeds)
    assert all(f["input_ids"].shape[1] < 32 for f in session.feeds)  # padding switched off
    assert all(0 < s < 1 for s in scores)
    assert scores[1] > scores[0] > scores[2]  # monotonic in the logit


def test_inputs_are_filtered_to_the_declared_ones(model):
    spec, directory = model
    two = FakeSession()
    OnnxReranker(spec, directory, session=two).score("q", ["w"])
    assert set(two.feeds[0]) == {"input_ids", "attention_mask"}
    three = FakeSession(inputs=("input_ids", "attention_mask", "token_type_ids"))
    OnnxReranker(spec, directory, session=three).score("q", ["w"])
    assert not three.feeds[0]["token_type_ids"].any()


def test_truncation_cuts_the_passage_and_keeps_the_query(model):
    spec, directory = model
    session = FakeSession()
    OnnxReranker(spec, directory, max_length=10, session=session).score("q q q", ["w " * 50])
    ids = session.feeds[0]["input_ids"][0].tolist()
    assert len(ids) == 10 and ids[:5] == [1, 4, 4, 4, 2]


def test_a_query_longer_than_max_length_still_scores(model):
    spec, directory = model
    session = FakeSession()
    OnnxReranker(spec, directory, max_length=10, session=session).score("q " * 40, ["w w"])
    assert session.feeds[0]["input_ids"].shape == (1, 10)


def test_same_pair_same_score(model):
    spec, directory = model
    reranker = OnnxReranker(spec, directory, session=FakeSession())
    assert reranker.score("q", ["w w"]) == reranker.score("q", ["x", "w w"])[1:]


def test_runtime_failure_becomes_rerank_error(model):
    spec, directory = model
    reranker = OnnxReranker(spec, directory, session=FakeSession(error=ort_state.Fail("boom")))
    with pytest.raises(RerankError, match="reranker failed: Fail"):
        reranker.score("q", ["w"])


def test_bad_files_fail_with_download_command(model, tmp_path):
    spec, directory = model
    (directory / "onnx" / "model_int8.onnx").write_bytes(b"tampered")
    with pytest.raises(RerankerConfigError, match="wrong sha256.*--only reranker"):
        OnnxReranker(spec, directory, session=FakeSession())
    with pytest.raises(RerankerConfigError, match="missing"):
        OnnxReranker(spec, tmp_path / "nowhere", session=FakeSession())


def test_unknown_model_lists_the_known_ones(tmp_path):
    settings = RetrievalSettings(rerank_model="org/unknown", models_dir=tmp_path)
    with pytest.raises(RerankerConfigError, match="known: Alibaba-NLP/gte"):
        get_reranker(settings)


def test_fingerprint_names_repo_revision_and_hash(model):
    spec, directory = model
    fingerprint = OnnxReranker(spec, directory, session=FakeSession()).fingerprint
    assert fingerprint == f"org/fake-reranker@r1:{spec.onnx_sha256}"


def test_passage_drops_link_targets_like_the_embedded_text():
    document = doc(*md(f"# Campus\n\nPar Niemeyer (https://fr.wikipedia.org/wiki/N). {words(10)}"))
    (chunk,) = chunk_document(document, settings=SETTINGS, counter=FakeCounter())
    assert "wikipedia" not in passage(document.title, chunk.heading_path, chunk.text)
    assert passage(document.title, chunk.heading_path, chunk.text) == chunk.embed_text


def test_spec_max_length_caps_the_setting(model):
    from dataclasses import replace

    spec, directory = model
    session = FakeSession()
    capped = replace(spec, max_length=10)
    OnnxReranker(capped, directory, max_length=640, session=session).score("q", ["w " * 50])
    assert session.feeds[0]["input_ids"].shape == (1, 10)


def test_token_type_ids_follow_the_tokenizer(model, tmp_path):
    spec, directory = model
    tok = Tokenizer.from_file(str(directory / "tokenizer.json"))
    tok.post_processor = processors.TemplateProcessing(  # BERT style: passage is segment 1
        single="<s> $A </s>",
        pair="<s> $A </s> $B:1 </s>:1",
        special_tokens=[("<s>", 1), ("</s>", 2)],
    )
    tok.save(str(directory / "tokenizer.json"))
    from dataclasses import replace

    sha = hashlib.sha256((directory / "tokenizer.json").read_bytes()).hexdigest()
    session = FakeSession(inputs=("input_ids", "attention_mask", "token_type_ids"))
    OnnxReranker(replace(spec, tokenizer_sha256=sha), directory, session=session).score("q", ["w"])
    assert session.feeds[0]["token_type_ids"].tolist() == [[0, 0, 0, 1, 1]]
