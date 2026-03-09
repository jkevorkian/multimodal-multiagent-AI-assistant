from __future__ import annotations

import json
from uuid import uuid4

from fastapi.testclient import TestClient

from app.core.config import settings
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


def test_run_status_endpoint_reports_latest_snapshot() -> None:
    run_id = f"run-{uuid4().hex}"
    run_response = client.post(
        "/agents/run",
        json={"query": "summarize available context", "run_id": run_id},
    )
    assert run_response.status_code == 200
    run_payload = run_response.json()
    assert run_payload["run_id"] == run_id

    status_response = client.get(f"/runs/{run_id}/status")
    assert status_response.status_code == 200
    status = status_response.json()
    assert status["run_id"] == run_id
    assert status["state"] in {"completed", "guardrail_exit"}
    assert status["last_event_type"] in {"run.completed", "run.guardrail_triggered"}
    assert status["last_sequence_number"] >= 1


def test_sse_events_are_ordered_and_replayable() -> None:
    run_id = f"run-{uuid4().hex}"
    run_response = client.post(
        "/agents/run",
        json={"query": "search latest policy notes", "run_id": run_id},
    )
    assert run_response.status_code == 200

    events_response = client.get(f"/runs/{run_id}/events", params={"follow": "false"})
    assert events_response.status_code == 200
    assert events_response.headers["content-type"].startswith("text/event-stream")
    all_events = _parse_sse_events(events_response.text)
    assert all_events

    sequences = [int(item["sequence_number"]) for item in all_events]
    assert sequences == sorted(sequences)
    assert len(set(sequences)) == len(sequences)
    event_types = {item["event_type"] for item in all_events}
    assert "run.started" in event_types
    assert "agent.step.started" in event_types

    pivot = sequences[len(sequences) // 2]
    replay_response = client.get(
        f"/runs/{run_id}/events",
        params={"follow": "false", "after_sequence": pivot},
    )
    assert replay_response.status_code == 200
    replay_events = _parse_sse_events(replay_response.text)
    assert replay_events
    assert all(int(item["sequence_number"]) > pivot for item in replay_events)


def test_revision_event_is_not_emitted_for_plain_insufficient_evidence_by_default() -> None:
    container = create_test_container()
    app.dependency_overrides[get_container] = lambda: container
    run_id = f"run-{uuid4().hex}"
    try:
        run_response = client.post(
            "/agents/run",
            json={"query": "answer from missing corpus", "run_id": run_id, "tools": ["nonexistent_tool"]},
        )
        assert run_response.status_code == 200

        events_response = client.get(f"/runs/{run_id}/events", params={"follow": "false"})
        assert events_response.status_code == 200
        all_events = _parse_sse_events(events_response.text)
        assert all(item.get("event_type") != "agent.revision.requested" for item in all_events)
    finally:
        app.dependency_overrides.clear()


def test_guardrail_status_when_max_steps_is_tight() -> None:
    original_max_steps = settings.agent_max_steps
    try:
        settings.agent_max_steps = 1
        run_id = f"run-{uuid4().hex}"
        run_response = client.post(
            "/agents/run",
            json={"query": "force partial execution", "run_id": run_id},
        )
        assert run_response.status_code == 200

        status_response = client.get(f"/runs/{run_id}/status")
        assert status_response.status_code == 200
        status = status_response.json()
        assert status["state"] == "guardrail_exit"
        assert status["last_event_type"] == "run.guardrail_triggered"
    finally:
        settings.agent_max_steps = original_max_steps
