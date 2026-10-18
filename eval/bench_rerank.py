"""Reranker CPU latency and int8 batch dependence on corpus passages (specs/retrieval/design.md).

    MODELS_DIR=$PWD/models uv run python eval/bench_rerank.py --model BAAI/bge-reranker-v2-m3
    uv run python eval/bench_rerank.py --threads 8 --tokens 500 --batched

Cuts ``--count`` passages of about ``--tokens`` reranker tokens from the corpus files, scores them
against an Arabic, a French and an English query one pair per run, and with ``--batched`` also in
padded batches of 8 and of all passages, printing the time and the largest score change.
"""

import argparse
import os
import resource
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from micro_bot.retrieval.registry import DEFAULT_RERANKER, RERANKERS

QUERIES = {
    "ar": "ما هي شروط الترشح لمنصب نائب الرئيس في النادي؟",
    "fr": "Comment personnaliser le thème des composants de l'interface ?",
    "en": "How do I change the theme colors of the UI library?",
}


def passages(corpus: Path, tok: Tokenizer, target: int, count: int) -> list[str]:
    """Consecutive word windows of at most ``target`` tokens from every file."""
    out: list[str] = []
    for path in sorted(corpus.glob("*.md")):
        words = path.read_text(encoding="utf-8").split()
        start = 0
        while start < len(words) and len(out) < count:
            lo, hi = 1, len(words) - start
            while lo < hi:  # binary search for the largest window that fits
                mid = (lo + hi + 1) // 2
                fits = len(tok.encode(" ".join(words[start : start + mid])).ids) <= target
                lo, hi = (mid, hi) if fits else (lo, mid - 1)
            if lo < 50:
                break
            out.append(" ".join(words[start : start + lo]))
            start += lo
    return out[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_RERANKER, choices=sorted(RERANKERS))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=640)
    parser.add_argument("--tokens", type=int, default=384)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--corpus", type=Path, default=Path("data/parsed"))
    parser.add_argument("--batched", action="store_true")
    args = parser.parse_args()

    spec = RERANKERS[args.model]
    directory = Path(os.environ.get("MODELS_DIR", "models")) / "reranker" / spec.slug
    tok = Tokenizer.from_file(str(directory / "tokenizer.json"))
    tok.no_padding()
    texts = passages(args.corpus, tok, args.tokens, args.count)
    lengths = [len(tok.encode(t).ids) for t in texts]
    tok.enable_truncation(args.max_length, strategy="only_second")

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    started = time.perf_counter()
    session = ort.InferenceSession(str(directory / spec.onnx_file), options)
    load_s = time.perf_counter() - started
    names = {i.name for i in session.get_inputs()}
    pad_id = tok.token_to_id("<pad>") or 1

    def run(pairs: list[tuple[str, str]]) -> np.ndarray:
        encs = [tok.encode(q, p) for q, p in pairs]
        ids = np.full((len(encs), max(len(e.ids) for e in encs)), pad_id, dtype=np.int64)
        mask = np.zeros_like(ids)
        for i, e in enumerate(encs):
            ids[i, : len(e.ids)], mask[i, : len(e.ids)] = e.ids, 1
        feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
        return session.run(None, {k: v for k, v in feed.items() if k in names})[0].reshape(-1)

    run([(QUERIES["en"], texts[0])])  # warm-up
    print(f"{spec.repo} threads={args.threads} load={load_s:.1f}s inputs={sorted(names)}")
    print(f"{len(texts)} passages, tokens min {min(lengths)} median {statistics.median(lengths)}")
    for lang, query in QUERIES.items():
        pairs = [(query, t) for t in texts]
        started = time.perf_counter()
        alone = np.array([run([pair])[0] for pair in pairs])
        line = f" {lang}: batch 1 {time.perf_counter() - started:.2f}s"
        if args.batched:
            for size in (8, len(pairs)):
                started = time.perf_counter()
                batched = np.concatenate(
                    [run(pairs[i : i + size]) for i in range(0, len(pairs), size)]
                )
                elapsed = time.perf_counter() - started
                drift = np.abs(alone - batched).max()
                line += f" | batch {size} {elapsed:.2f}s, max drift {drift:.3f}"
        print(line + f" | logits {alone.min():.2f}..{alone.max():.2f}")
    print(f"max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")


if __name__ == "__main__":
    main()
