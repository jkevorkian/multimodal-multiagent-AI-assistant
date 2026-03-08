from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.core.dependencies import create_test_container, get_container
from app.main import app

client = TestClient(app)


def test_chat_sessions_persist_messages_and_can_be_continued(tmp_path: Path) -> None:
    container = create_test_container()
    app.dependency_overrides[get_container] = lambda: container
    try:
        create_response = client.post("/chat/sessions", json={"title": "M24 Session"})
        assert create_response.status_code == 200
        chat_id = create_response.json()["chat_id"]

        first = client.post(
            f"/chat/sessions/{chat_id}/messages",
            json={"message": "Hello there", "mode": "rag", "top_k": 3},
        )
        assert first.status_code == 200

        note = tmp_path / "chat_note.txt"
        note.write_text("Session notes mention retention is 45 days.", encoding="utf-8")
        second = client.post(
            f"/chat/sessions/{chat_id}/messages",
            json={
                "message": "Use this source and summarize retention.",
                "mode": "rag",
                "sources": [note.as_uri()],
                "source_type": "text",
                "top_k": 3,
            },
        )
        assert second.status_code == 200
        assert second.json()["accepted_sources"] == 1

        history_response = client.get(f"/chat/sessions/{chat_id}/messages")
        assert history_response.status_code == 200
        payload = history_response.json()
        assert payload["chat_id"] == chat_id
        assert len(payload["messages"]) >= 4
        assert any(item.get("role") == "assistant" for item in payload["messages"])
        assert payload["files"]
    finally:
        app.dependency_overrides.clear()
