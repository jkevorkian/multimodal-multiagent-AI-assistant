from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.core.dependencies import create_test_container, get_container
from app.main import app

client = TestClient(app)


def test_chat_scoped_retrieval_isolated_between_sessions(tmp_path: Path) -> None:
    container = create_test_container()
    app.dependency_overrides[get_container] = lambda: container
    try:
        chat1 = client.post("/chat/sessions", json={"title": "Chat A"}).json()["chat_id"]
        chat2 = client.post("/chat/sessions", json={"title": "Chat B"}).json()["chat_id"]

        source_a = tmp_path / "apple.txt"
        source_a.write_text("Chat A context: apples are red fruits.", encoding="utf-8")
        source_b = tmp_path / "banana.txt"
        source_b.write_text("Chat B context: bananas are yellow fruits.", encoding="utf-8")

        ingest_a = client.post(
            f"/chat/sessions/{chat1}/messages",
            json={
                "message": "Index source A",
                "mode": "rag",
                "sources": [source_a.as_uri()],
                "source_type": "text",
            },
        )
        assert ingest_a.status_code == 200
        ingest_b = client.post(
            f"/chat/sessions/{chat2}/messages",
            json={
                "message": "Index source B",
                "mode": "rag",
                "sources": [source_b.as_uri()],
                "source_type": "text",
            },
        )
        assert ingest_b.status_code == 200

        ask_a = client.post(
            f"/chat/sessions/{chat1}/messages",
            json={"message": "Which fruit is in this chat?", "mode": "rag", "top_k": 3},
        )
        ask_b = client.post(
            f"/chat/sessions/{chat2}/messages",
            json={"message": "Which fruit is in this chat?", "mode": "rag", "top_k": 3},
        )
        assert ask_a.status_code == 200
        assert ask_b.status_code == 200
        citations_a = ask_a.json()["citations"]
        citations_b = ask_b.json()["citations"]
        assert any("apple.txt" in item for item in citations_a)
        assert all("banana.txt" not in item for item in citations_a)
        assert any("banana.txt" in item for item in citations_b)
        assert all("apple.txt" not in item for item in citations_b)
    finally:
        app.dependency_overrides.clear()
