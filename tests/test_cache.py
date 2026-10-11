from mcclub_rag.cache import ExactMatchCache, normalize


def test_case_punctuation_and_greeting_variants_hit_the_same_entry() -> None:
    cache = ExactMatchCache(lambda: 1)
    response = {"answer": "Registration closes Friday."}

    cache.set("Hello, when does registration close?", "public", response)

    assert cache.get("  HELLO!!! When does registration close??  ", "public") is response


def test_workshop_a_and_workshop_b_do_not_share_entries() -> None:
    cache = ExactMatchCache(lambda: 1)
    response = {"answer": "Workshop A is Saturday."}

    cache.set("When is workshop A?", "public", response)

    assert cache.get("When is workshop B?", "public") is None


def test_history_skips_cache_reads_and_writes() -> None:
    cache = ExactMatchCache(lambda: 1)
    response = {"answer": "Registration closes Friday."}

    cache.set("When does registration close?", "public", response)

    assert cache.get("When does registration close?", "public", history=["prior turn"]) is None
    cache.set("What should I bring?", "public", {"answer": "An ID."}, history=["prior turn"])
    assert cache.get("What should I bring?", "public") is None


def test_corpus_version_bump_misses_existing_entries() -> None:
    version = 1
    cache = ExactMatchCache(lambda: version)
    response = {"answer": "Workshop A is Saturday."}

    cache.set("When is workshop A?", "public", response)
    version = 2

    assert cache.get("When is workshop A?", "public") is None


def test_normalize_uses_the_local_arabic_fallback_safely() -> None:
    assert normalize("السَّلَامُ عَلَيْكُمْ، مَتَى الوِرْشَة؟") == "متي الورشه"
