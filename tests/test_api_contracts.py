from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_required_routes_registered() -> None:
    expected_paths = {
        "/health",
        "/ingest/documents",
        "/query",
        "/agents/run",
        "/vision/analyze",
        "/video/analyze",
        "/metrics",
    }
    routes = {route.path for route in app.routes}
    assert expected_paths.issubset(routes)


def test_health_contract() -> None:
    response = client.get("/health")
    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert "service" in payload
    assert "version" in payload


def test_ingest_contract() -> None:
    response = client.post("/ingest/documents", json={"sources": ["file://sample.pdf"], "source_type": "pdf"})
    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "accepted"
    assert payload["accepted_sources"] == 1
    assert "trace" in payload


def test_query_contract() -> None:
    response = client.post("/query", json={"query": "What is this project?"})
    payload = response.json()
    assert response.status_code == 200
    assert "answer" in payload
    assert "citations" in payload
    assert "confidence" in payload
    assert "trace" in payload


def test_agents_contract() -> None:
    response = client.post("/agents/run", json={"query": "Analyze this", "tools": ["web_search"]})
    payload = response.json()
    assert response.status_code == 200
    assert "steps" in payload
    assert "tool_calls" in payload
    assert "trace" in payload


def test_vision_contract() -> None:
    response = client.post("/vision/analyze", json={"image_uri": "s3://bucket/image.png"})
    payload = response.json()
    assert response.status_code == 200
    assert "summary" in payload
    assert "findings" in payload
    assert "trace" in payload


def test_video_contract() -> None:
    response = client.post("/video/analyze", json={"video_uri": "s3://bucket/video.mp4"})
    payload = response.json()
    assert response.status_code == 200
    assert "summary" in payload
    assert "key_events" in payload
    assert "processed_frames" in payload
    assert "trace" in payload


def test_metrics_contract() -> None:
    response = client.get("/metrics")
    payload = response.json()
    assert response.status_code == 200
    assert set(payload.keys()) == {"accuracy", "p95_latency_ms", "avg_cost_usd", "last_updated"}


def test_trace_headers_added() -> None:
    response = client.get("/health")
    assert response.headers.get("x-request-id")
    assert response.headers.get("x-trace-id")

