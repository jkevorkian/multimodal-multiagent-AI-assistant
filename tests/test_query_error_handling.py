import asyncio

from fastapi.testclient import TestClient

from app.core.dependencies import create_test_container, get_container
from app.main import create_app


class _FailingLLM:
    async def generate(self, prompt: str, context: list[str] | None = None) -> str:  # noqa: ARG002
        raise RuntimeError("simulated llm failure")


class _EmptyLLM:
    async def generate(self, prompt: str, context: list[str] | None = None) -> str:  # noqa: ARG002
        return ""


def test_query_returns_json_when_llm_fails() -> None:
    app = create_app()
    container = create_test_container()
    container.llm = _FailingLLM()  # type: ignore[assignment]
    app.dependency_overrides[get_container] = lambda: container

    with TestClient(app) as client:
        response = client.post("/query", json={"query": "What is the status?"})
        payload = response.json()
        assert response.status_code == 200
        assert "answer" in payload
        assert "failed" in payload["answer"].lower() or "try again" in payload["answer"].lower()
        assert "trace" in payload

    app.dependency_overrides.clear()


def test_query_filters_unreadable_snippets_before_generation() -> None:
    app = create_app()
    container = create_test_container()
    gibberish = (bytes(range(256)) * 6).decode("latin-1", errors="ignore")
    vector = asyncio.run(container.embeddings.embed_text("placeholder vector seed"))
    asyncio.run(
        container.vector_store.upsert(
            ids=["bad-1"],
            vectors=[vector],
            metadata=[
                {
                    "source": "file:///tmp/binary.pdf",
                    "chunk_id": 0,
                    "offset": 0,
                    "snippet": gibberish,
                }
            ],
        )
    )
    app.dependency_overrides[get_container] = lambda: container

    with TestClient(app) as client:
        response = client.post("/query", json={"query": "What does it say?"})
        payload = response.json()
        assert response.status_code == 200
        assert "indexed context yet" in payload["answer"].lower()

    app.dependency_overrides.clear()


def test_query_returns_fallback_when_llm_returns_empty_string() -> None:
    app = create_app()
    container = create_test_container()
    container.llm = _EmptyLLM()  # type: ignore[assignment]
    app.dependency_overrides[get_container] = lambda: container

    with TestClient(app) as client:
        response = client.post("/query", json={"query": "What is the status?"})
        payload = response.json()
        assert response.status_code == 200
        assert payload["answer"]
        assert "empty" in payload["answer"].lower()

    app.dependency_overrides.clear()
