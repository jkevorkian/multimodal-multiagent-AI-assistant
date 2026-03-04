import asyncio

from app.storage.fallback_vector_store import FallbackVectorStore


class _StoreStub:
    def __init__(self, fail_upsert: bool = False, search_results: list[dict] | None = None, keyword_results: list[dict] | None = None) -> None:
        self.fail_upsert = fail_upsert
        self.search_results = search_results or []
        self.keyword_results = keyword_results or []
        self.upsert_calls = 0

    async def upsert(self, ids: list[str], vectors: list[list[float]], metadata: list[dict]) -> None:  # noqa: ARG002
        self.upsert_calls += 1
        if self.fail_upsert:
            raise RuntimeError("upsert failed")

    async def search(self, vector: list[float], top_k: int) -> list[dict]:  # noqa: ARG002
        return self.search_results[:top_k]

    async def keyword_search(self, query: str, top_k: int) -> list[dict]:  # noqa: ARG002
        return self.keyword_results[:top_k]


def test_fallback_upsert_writes_secondary_when_primary_fails() -> None:
    primary = _StoreStub(fail_upsert=True)
    secondary = _StoreStub()
    store = FallbackVectorStore(primary=primary, secondary=secondary, mirror_writes=False)

    asyncio.run(store.upsert(ids=["1"], vectors=[[0.1, 0.2]], metadata=[{"source": "x"}]))
    assert primary.upsert_calls == 1
    assert secondary.upsert_calls == 1


def test_fallback_search_uses_secondary_if_primary_empty() -> None:
    primary = _StoreStub(search_results=[])
    secondary = _StoreStub(search_results=[{"id": "secondary", "metadata": {"source": "doc"}, "score": 0.4}])
    store = FallbackVectorStore(primary=primary, secondary=secondary)

    results = asyncio.run(store.search(vector=[0.2, 0.3], top_k=3))
    assert results
    assert results[0]["id"] == "secondary"


def test_fallback_keyword_prefers_secondary() -> None:
    primary = _StoreStub(keyword_results=[{"id": "primary", "metadata": {"source": "p"}, "score": 0.2}])
    secondary = _StoreStub(keyword_results=[{"id": "secondary", "metadata": {"source": "s"}, "score": 0.3}])
    store = FallbackVectorStore(primary=primary, secondary=secondary)

    results = asyncio.run(store.keyword_search(query="policy", top_k=2))
    assert results
    assert results[0]["id"] == "secondary"
