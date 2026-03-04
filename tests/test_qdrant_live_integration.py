from __future__ import annotations

from urllib.request import urlopen

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.dependencies import get_container
from app.main import create_app


def _qdrant_is_alive(url: str) -> bool:
    try:
        with urlopen(f"{url}/collections", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def test_qdrant_primary_store_live_path(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("qdrant_client")
    qdrant_url = "http://localhost:6333"
    if not _qdrant_is_alive(qdrant_url):
        pytest.skip("Qdrant service is not running on localhost:6333")

    monkeypatch.setattr(settings, "rag_vector_store_provider", "qdrant")
    monkeypatch.setattr(settings, "qdrant_url", qdrant_url)
    monkeypatch.setattr(settings, "qdrant_collection_name", "rag_chunks")
    monkeypatch.setattr(settings, "rag_vector_store_mirror_writes", True)
    monkeypatch.setattr(settings, "rag_embedding_provider", "deterministic")
    monkeypatch.setattr(settings, "rag_embedding_dimensions", 64)

    get_container.cache_clear()
    app = create_app()

    try:
        with TestClient(app) as client:
            source_path = tmp_path / "qdrant_source.txt"
            source_path.write_text(
                "Qdrant is running as primary vector store while fallback remains available.",
                encoding="utf-8",
            )

            ingest = client.post(
                "/ingest/documents",
                json={"sources": [str(source_path)], "source_type": "text"},
            )
            ingest_payload = ingest.json()
            assert ingest.status_code == 200
            assert ingest_payload["accepted_sources"] == 1
            assert ingest_payload["indexed_chunks"] > 0

            query = client.post("/query", json={"query": "Which vector store is primary?", "top_k": 3})
            query_payload = query.json()
            assert query.status_code == 200
            assert query_payload["citations"]

            container = get_container()
            assert container.vector_store_provider == "qdrant+pgvector"
    finally:
        get_container.cache_clear()
