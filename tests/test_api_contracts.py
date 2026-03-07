from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from app.core.dependencies import create_test_container, get_container
from app.main import app


client = TestClient(app)


def test_required_routes_registered() -> None:
    expected_paths = {
        "/health",
        "/ingest/documents",
        "/query",
        "/agents/run",
        "/agents/tools",
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
    with TemporaryDirectory() as temp_dir:
        pdf_path = Path(temp_dir) / "sample.pdf"
        pdf_path.write_text("M1 contract fixture content.", encoding="utf-8")
        response = client.post("/ingest/documents", json={"sources": [pdf_path.as_uri()], "source_type": "pdf"})
        payload = response.json()
        assert response.status_code == 200
        assert payload["status"] == "accepted"
        assert payload["accepted_sources"] == 1
        assert payload["indexed_chunks"] > 0
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


def test_agents_tools_contract() -> None:
    response = client.get("/agents/tools")
    payload = response.json()
    assert response.status_code == 200
    assert "tools" in payload
    assert isinstance(payload["tools"], list)
    assert payload["tools"]
    first = payload["tools"][0]
    assert "name" in first
    assert "description" in first


def test_agents_contract_defaults_to_all_tools_when_omitted() -> None:
    response = client.post("/agents/run", json={"query": "Analyze this"})
    payload = response.json()
    assert response.status_code == 200
    assert "tool_calls" in payload
    assert isinstance(payload["tool_calls"], list)


def test_vision_contract() -> None:
    response = client.post("/vision/analyze", json={"image_uri": "s3://bucket/image.png"})
    payload = response.json()
    assert response.status_code == 200
    assert "summary" in payload
    assert "findings" in payload
    assert "trace" in payload


def test_video_contract(monkeypatch, tmp_path) -> None:
    class _Encoded:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        def tobytes(self) -> bytes:
            return self._payload

    class _Frame:
        shape = (480, 640, 3)

    class _Capture:
        def __init__(self, path: str) -> None:
            self.path = path

        def isOpened(self) -> bool:
            return True

        def get(self, prop: int) -> float:
            if prop == _FakeCV2.CAP_PROP_FPS:
                return 10.0
            if prop == _FakeCV2.CAP_PROP_FRAME_COUNT:
                return 100.0
            return 0.0

        def set(self, prop: int, value: float) -> bool:  # noqa: ARG002
            return True

        def read(self):
            return True, _Frame()

        def release(self) -> None:
            return None

    class _FakeCV2:
        CAP_PROP_FPS = 1
        CAP_PROP_FRAME_COUNT = 2
        CAP_PROP_POS_MSEC = 3
        INTER_AREA = 1

        def VideoCapture(self, path: str) -> _Capture:
            return _Capture(path)

        def imencode(self, ext: str, frame: _Frame):  # noqa: ARG002
            return True, _Encoded(b"jpeg-bytes")

        def resize(self, frame: _Frame, size: tuple[int, int], interpolation: int):  # noqa: ARG002
            return frame

    monkeypatch.setattr("app.video.frame_sampler._load_cv2", lambda: _FakeCV2())
    video_path = tmp_path / "contract_video.mp4"
    video_path.write_bytes(b"video")

    container = create_test_container()
    app.dependency_overrides[get_container] = lambda: container
    try:
        response = client.post("/video/analyze", json={"video_uri": video_path.as_uri()})
    finally:
        app.dependency_overrides.clear()
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
