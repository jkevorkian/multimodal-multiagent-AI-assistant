from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from app.core.dependencies import create_test_container, get_container
from app.main import app

client = TestClient(app)


def _parse_sse_events(payload: str) -> list[dict]:
    events: list[dict] = []
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line.startswith("data: "):
            continue
        events.append(json.loads(line[6:]))
    return events


def test_chat_run_events_include_transcript_details(tmp_path: Path) -> None:
    container = create_test_container()
    app.dependency_overrides[get_container] = lambda: container
    run_id = f"chat-run-{uuid4().hex}"
    try:
        chat_id = client.post("/chat/sessions", json={"title": "Agentic Chat"}).json()["chat_id"]
        source = tmp_path / "events.txt"
        source.write_text("Runtime transcript source context for retrieval preview.", encoding="utf-8")

        send = client.post(
            f"/chat/sessions/{chat_id}/messages",
            json={
                "message": "Search this source with tools",
                "mode": "agentic",
                "run_id": run_id,
                "tools": ["stub_tool"],
                "sources": [source.as_uri()],
                "source_type": "text",
            },
        )
        assert send.status_code == 200
        assert send.json()["run_id"] == run_id

        events_response = client.get(
            f"/chat/sessions/{chat_id}/runs/{run_id}/events",
            params={"follow": "false"},
        )
        assert events_response.status_code == 200
        events = _parse_sse_events(events_response.text)
        assert events
        assert any(item.get("metadata", {}).get("event_family") == "stage" for item in events)
        assert any(item.get("metadata", {}).get("retrieval_preview") for item in events if isinstance(item.get("metadata"), dict))
    finally:
        app.dependency_overrides.clear()
