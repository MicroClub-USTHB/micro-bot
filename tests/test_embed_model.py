"""Real default embedder (bge-m3) and BM25 stopwords (``embedder`` marker)."""

import os
import time

import numpy as np
import pytest

from mcclub_rag.retrieval.embed import Bm25SparseEncoder, OnnxDenseEncoder
from mcclub_rag.retrieval.registry import DEFAULT_EMBEDDER, EMBEDDERS

SPEC = EMBEDDERS[DEFAULT_EMBEDDER]

PAIRS = [
    "Le club organise un hackathon en novembre.",
    "The club is organizing a hackathon in November.",
    "ينظم النادي هاكاثون في نوفمبر.",
    "La recette de la tarte aux pommes demande du beurre.",
]


@pytest.fixture(scope="module")
def dense(onnx_path) -> OnnxDenseEncoder:
    return OnnxDenseEncoder(SPEC, onnx_path.parents[1], threads=4, max_length=512, dim=SPEC.dim)


def test_vectors_have_the_spec_dim_and_unit_norm(dense):
    for vector in dense.embed_documents(PAIRS):
        assert len(vector) == SPEC.dim
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)


def test_vector_does_not_depend_on_its_neighbours(dense):
    alone = dense.embed_documents([PAIRS[0]])[0]
    in_list = dense.embed_documents([PAIRS[3], PAIRS[0], "court"])[1]
    assert np.abs(np.asarray(alone) - np.asarray(in_list)).max() < 1e-4
    assert dense.embed_query(PAIRS[0]) == alone


def test_translations_score_above_unrelated_text(dense):
    fr, en, ar, unrelated = (np.asarray(v) for v in dense.embed_documents(PAIRS))
    assert fr @ en > fr @ unrelated
    assert fr @ ar > fr @ unrelated


def test_dense_and_sparse_throughput(dense, real_bm25_dir):
    sparse = Bm25SparseEncoder(real_bm25_dir)
    paragraph = "Le club organise des ateliers de programmation chaque semaine pour les membres. "
    chunks = [f"{i} " + paragraph * 22 for i in range(32)]  # about 300 tokens each
    if (os.cpu_count() or 1) < 4:
        pytest.skip("throughput target is defined on 4 CPU threads")
    start = time.perf_counter()
    dense.embed_documents(chunks)
    sparse.embed_documents(chunks)
    # About 5 s on the dev machine, 11 s on a 4-vCPU GitHub runner (Req 9.5).
    assert time.perf_counter() - start < 20
