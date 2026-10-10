from datetime import UTC, datetime, timedelta

import pytest
from qdrant_client import models as qm

from mcclub_rag.retrieval.search import Candidate, dedupe, visibility_filter

T0 = datetime(2026, 10, 10, tzinfo=UTC)


def cand(doc="d", index=0, rank=0, day=0, **kw) -> Candidate:
    fields = dict(
        point_id=f"{doc}-{index}-{rank}",
        doc_id=doc,
        content_hash="h",
        index=index,
        title="T",
        heading_path=(),
        text="x",
        visibility="public",
        ingested_at=T0 + timedelta(days=day),
        fused_score=1.0 / (rank + 1),
        fused_rank=rank,
    )
    return Candidate(**(fields | kw))


def _match(flt: qm.Filter):
    return flt.must[0].match


def test_visibility_filter_per_role():
    assert _match(visibility_filter("public")) == qm.MatchValue(value="public")
    assert _match(visibility_filter("member")) == qm.MatchAny(any=["public", "members"])
    with pytest.raises(ValueError, match="unknown role"):
        visibility_filter("admin")


def test_dedupe_keeps_newest_version_in_fused_order():
    old = cand("a", 0, rank=0, day=0, point_id="old")
    new = cand("a", 0, rank=2, day=1, point_id="new")
    other = cand("b", 0, rank=1)
    assert [c.point_id for c in dedupe([old, other, new])] == ["b-0-1", "new"]


def test_dedupe_tie_goes_to_better_rank():
    first, second = cand("a", 0, rank=0), cand("a", 0, rank=3)
    assert dedupe([second, first]) == [first]
