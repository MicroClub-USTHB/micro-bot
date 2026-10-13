"""Builders for a run: the corpus sources on disk and the in-process app with real models."""

from pathlib import Path

from eval.e2e.drivers import InProcessDriver, Source
from eval.e2e.generator import Budget, OllamaClient, RecordingGenerator, ThreadedClient
from eval.run_eval import document_text

REPO = Path(__file__).resolve().parents[2]
E2E = REPO / "eval" / "e2e"


def sources(corpus: dict, work: Path) -> list[Source]:
    """Every manifest document whose file is here; parse.py dumps reduced to their text."""
    out = []
    for doc in corpus.values():
        adversarial = E2E / "corpus" / doc.file
        parsed = REPO / "data" / "parsed" / doc.file
        if adversarial.is_file():
            out.append(Source(doc, adversarial))
        elif parsed.is_file():
            path = work / doc.file
            path.write_text(document_text(parsed.read_text(encoding="utf-8")), encoding="utf-8")
            out.append(Source(doc, path))
    return out


async def inprocess(args, work: Path):
    from qdrant_client import AsyncQdrantClient

    from mcclub_rag.config import get_settings
    from mcclub_rag.ingest.docstore import DocStore
    from mcclub_rag.ingest.pipeline import IngestPipeline
    from mcclub_rag.ingest.settings import IngestSettings
    from mcclub_rag.llm.client import OpenRouterClient
    from mcclub_rag.retrieval.backend import HybridRetriever
    from mcclub_rag.retrieval.embed import get_dense_encoder, get_sparse_encoder
    from mcclub_rag.retrieval.settings import RetrievalSettings

    settings = IngestSettings(data_dir=work / "data")
    client = AsyncQdrantClient(url=args.qdrant_url) if args.qdrant_url else None
    pipeline = IngestPipeline(
        settings,
        dense=get_dense_encoder(settings),
        sparse=get_sparse_encoder(settings),
        client=client or AsyncQdrantClient(location=":memory:"),
        docstore=DocStore(settings.data_dir / "documents"),
    )
    await pipeline.start()
    retrieval = RetrievalSettings(qdrant_collection=settings.qdrant_collection)
    retriever = HybridRetriever.from_pipeline(pipeline, retrieval)  # RERANK_WORKERS sessions
    budget = Budget(args.max_llm_calls)
    if args.llm == "ollama":
        # Local stand-in for output tests; recordings are keyed apart from OpenRouter's.
        inner = ThreadedClient(OllamaClient(args.ollama_model))
        model = f"ollama/{args.ollama_model}"
    else:
        key = get_settings().openrouter_api_key
        inner = ThreadedClient(OpenRouterClient()) if key else None
        model = get_settings().openrouter_model
    generator = RecordingGenerator(inner, E2E / "recordings", args.mode, budget, model)
    driver = InProcessDriver(pipeline, retriever, generator, work / "app")
    info = {
        "gate_threshold": retrieval.gate_threshold,
        "rerank_model": retrieval.rerank_model,
        "llm_configured": inner is not None,
        "mode": args.mode,
        "model": model,
    }
    return driver, budget, generator, info, pipeline
