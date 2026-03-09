from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.dependencies import create_test_container, get_container
from app.main import app

client = TestClient(app)


def _send_rag(chat_id: str, message: str, top_k: int = 3) -> dict:
    response = client.post(
        f"/chat/sessions/{chat_id}/messages",
        json={"message": message, "mode": "rag", "top_k": top_k},
    )
    assert response.status_code == 200
    return response.json()


def test_chat_memory_summary_is_compacted_and_persisted() -> None:
    container = create_test_container()
    app.dependency_overrides[get_container] = lambda: container
    try:
        chat_id = client.post("/chat/sessions", json={"title": "Memory Summary"}).json()["chat_id"]
        for turn in range(12):
            _send_rag(chat_id, f"Turn {turn}: track this conversation memory marker {turn}.")

        memory = container.chat_store.get_memory(chat_id)
        assert memory is not None
        summary_text = str(memory.get("summary_text", ""))
        assert summary_text.startswith("Conversation memory:")
        assert "Turn" in summary_text
    finally:
        app.dependency_overrides.clear()


def test_chat_semantic_memory_recovers_older_fact_beyond_recent_turns() -> None:
    container = create_test_container()
    app.dependency_overrides[get_container] = lambda: container
    try:
        chat_id = client.post("/chat/sessions", json={"title": "Semantic Memory"}).json()["chat_id"]
        _send_rag(chat_id, "Please remember this fact: the launch code is ORBIT-9173.")
        for turn in range(10):
            _send_rag(chat_id, f"Filler turn {turn}: unrelated housekeeping update.")

        final_payload = _send_rag(chat_id, "What launch code did I ask you to remember?")
        assistant_text = str(final_payload["assistant_message"]["content"])
        assert "orbit-9173" in assistant_text.lower()
        assert any("#memory" in str(citation) for citation in final_payload.get("citations", []))
    finally:
        app.dependency_overrides.clear()
