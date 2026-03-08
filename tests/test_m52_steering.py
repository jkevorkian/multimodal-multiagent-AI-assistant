from __future__ import annotations

from fastapi.testclient import TestClient

from app.contracts.steering import SteeringRequest
from app.core.dependencies import create_test_container, get_container
from app.core.steering import apply_tool_policy
from app.main import app

client = TestClient(app)


def test_tool_policy_allow_deny_require_resolution() -> None:
    available = ["stub_tool", "web_search", "url_fetch"]
    steering = SteeringRequest.model_validate(
        {
            "tools": {
                "allow": ["stub_tool", "url_fetch"],
                "deny": ["url_fetch"],
                "require": ["web_search"],
            }
        }
    )
    selected, notes = apply_tool_policy(
        available_tools=available,
        requested_tools=["stub_tool", "web_search"],
        steering=steering,
    )
    assert selected == ["stub_tool", "web_search"]
    assert any("tool_allow_applied" in note for note in notes)
    assert any("tool_deny_applied" in note for note in notes)
    assert any("tool_require_applied" in note for note in notes)


def test_query_strict_grounded_can_abstain_when_citations_are_missing() -> None:
    container = create_test_container()
    app.dependency_overrides[get_container] = lambda: container
    try:
        response = client.post(
            "/query",
            json={
                "query": "What is the policy?",
                "top_k": 3,
                "steering": {
                    "profile": "strict-grounded",
                    "grounding": {"min_citations": 2, "abstain_on_insufficient_citations": True},
                },
            },
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"].startswith("I do not have enough grounded citations")
    assert payload["steering_applied"]["profile"] == "strict-grounded"


def test_query_creative_profile_applies_style_prefix(tmp_path) -> None:
    container = create_test_container()
    app.dependency_overrides[get_container] = lambda: container
    try:
        sample = tmp_path / "m52_creative.txt"
        sample.write_text("This is a grounding snippet for creative profile test.", encoding="utf-8")
        ingest = client.post("/ingest/documents", json={"sources": [sample.as_uri()], "source_type": "text"})
        assert ingest.status_code == 200
        response = client.post(
            "/query",
            json={"query": "Summarize this.", "top_k": 2, "steering": {"profile": "creative"}},
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"].startswith("Creative framing:")
    assert payload["steering_applied"]["profile"] == "creative"


def test_agents_route_reports_steering_application() -> None:
    container = create_test_container()
    app.dependency_overrides[get_container] = lambda: container
    try:
        response = client.post(
            "/agents/run",
            json={
                "query": "Search the indexed context.",
                "steering": {
                    "profile": "balanced",
                    "tools": {"allow": ["stub_tool"], "require": ["stub_tool"]},
                },
            },
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert payload["steering_applied"]["profile"] == "balanced"
    assert any("tool_require_applied" in note for note in payload["steering_applied"]["notes"])
