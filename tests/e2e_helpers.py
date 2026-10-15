"""Shared e2e fixtures: an in-process driver on fakes with one public and one members doc."""

from eval.e2e.cases import CorpusDoc
from eval.e2e.drivers import InProcessDriver, Source
from eval.e2e.generator import FaultyGenerator
from tests.pipeline_helpers import make_pipeline
from tests.retrieval_helpers import make_retriever

DOCS = {
    "club-notice": (
        "public",
        None,
        "# Python workshop\n\nThe Python workshop runs every Saturday at 10:00 in room 12.",
    ),
    "club-minutes": (
        "members",
        "CANARY-0001",
        "# Board minutes\n\nThe winter retreat is in Tikjda in December. CANARY-0001",
    ),
}


class RecordingRetriever:
    def __init__(self, inner):
        self.inner = inner
        self.histories = []

    async def retrieve(self, query, role, history):
        self.histories.append(list(history))
        return await self.inner.retrieve(query, role, history)


async def make_driver(tmp_path, generator=None):
    pipe = make_pipeline(tmp_path)
    await pipe.start()
    retriever = RecordingRetriever(make_retriever(pipe, gate_threshold=0.5))
    drv = InProcessDriver(
        pipe, retriever, generator or FaultyGenerator("timeout"), tmp_path / "app"
    )
    sources, corpus = [], {}
    for topic, (visibility, canary, text) in DOCS.items():
        path = tmp_path / f"{topic}.md"
        path.write_text(text, encoding="utf-8")
        doc = CorpusDoc(
            file=path.name, topic=topic, title=topic, visibility=visibility, canary=canary
        )
        sources.append(Source(doc, path))
        corpus[topic] = doc
    drv.topics = await drv.setup(sources)
    drv.corpus = corpus
    drv.pipe = pipe
    return drv
