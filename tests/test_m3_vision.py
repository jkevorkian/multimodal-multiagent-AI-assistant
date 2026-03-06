import asyncio
import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.dependencies import create_test_container, get_container
from app.main import create_app
from app.vision.adapter import VisionAnalysis
from app.vision.fusion import VisionFusion
from app.vision.preprocess import VisionPreprocessor

_ONE_BY_ONE_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z6kQAAAAASUVORK5CYII="


def _write_fixture_png(path: Path) -> None:
    path.write_bytes(base64.b64decode(_ONE_BY_ONE_PNG_BASE64))


def test_vision_preprocessor_extracts_deterministic_fixture_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "fixture.png"
    _write_fixture_png(image_path)
    processed = asyncio.run(VisionPreprocessor().preprocess(str(image_path)))
    assert processed.width == 1
    assert processed.height == 1
    assert processed.byte_size is not None and processed.byte_size > 0
    assert processed.content_hash is not None and len(processed.content_hash) == 64


def test_vision_preprocessor_rejects_invalid_type_and_size(tmp_path: Path) -> None:
    bad_path = tmp_path / "fixture.txt"
    bad_path.write_text("not an image", encoding="utf-8")
    preprocessor = VisionPreprocessor(max_source_bytes=10)

    with pytest.raises(ValueError):
        asyncio.run(preprocessor.preprocess(str(bad_path)))

    big_png = tmp_path / "big.png"
    big_png.write_bytes(base64.b64decode(_ONE_BY_ONE_PNG_BASE64) + b"X" * 100)
    with pytest.raises(ValueError):
        asyncio.run(preprocessor.preprocess(str(big_png)))


def test_vision_fusion_emits_evidence_grounded_findings() -> None:
    processed = asyncio.run(VisionPreprocessor().preprocess("https://example.com/sample.png"))
    analysis = VisionAnalysis(
        processed_image=processed,
        prompt="find objects",
        summary="A traffic sign is visible; a vehicle appears in the foreground.",
    )
    summary, findings, confidence = VisionFusion().compose(analysis)
    assert "traffic sign" in summary.lower()
    assert findings
    assert all("[evidence:" in finding for finding in findings)
    assert 0.1 <= confidence <= 0.95


def test_vision_preprocessor_resolves_webpage_url_to_image_data_uri(monkeypatch) -> None:
    class _FakeResponse:
        def __init__(self, payload: bytes, content_type: str) -> None:
            self._payload = payload
            self.headers = {"Content-Type": content_type}

        def read(self, max_bytes: int | None = None) -> bytes:
            if max_bytes is None:
                return self._payload
            return self._payload[:max_bytes]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    html = b'<html><head><meta property="og:image" content="https://cdn.example.com/dog.jpg"/></head></html>'
    image_bytes = base64.b64decode(_ONE_BY_ONE_PNG_BASE64)
    calls = {"count": 0}

    def _fake_urlopen(request, timeout: int = 10):  # noqa: ARG001
        calls["count"] += 1
        url = str(getattr(request, "full_url", request))
        if "dog-breed" in url:
            return _FakeResponse(html, "text/html; charset=utf-8")
        if "cdn.example.com/dog.jpg" in url:
            return _FakeResponse(image_bytes, "image/jpeg")
        raise AssertionError(f"unexpected URL in test: {url}")

    monkeypatch.setattr("app.vision.preprocess.urlopen", _fake_urlopen)

    processed = asyncio.run(VisionPreprocessor(max_source_bytes=1_000_000).preprocess("https://site.example/dog-breed"))
    assert processed.inference_uri.startswith("data:image/jpeg;base64,")
    assert processed.mime_type == "image/jpeg"
    assert processed.byte_size == len(image_bytes)
    assert processed.content_hash is not None
    assert calls["count"] >= 2


def test_vision_preprocessor_rejects_webpage_without_discoverable_image(monkeypatch) -> None:
    class _FakeResponse:
        def __init__(self, payload: bytes, content_type: str) -> None:
            self._payload = payload
            self.headers = {"Content-Type": content_type}

        def read(self, max_bytes: int | None = None) -> bytes:
            if max_bytes is None:
                return self._payload
            return self._payload[:max_bytes]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def _fake_urlopen(request, timeout: int = 10):  # noqa: ARG001
        return _FakeResponse(b"<html><head></head><body>no image here</body></html>", "text/html; charset=utf-8")

    monkeypatch.setattr("app.vision.preprocess.urlopen", _fake_urlopen)

    with pytest.raises(ValueError):
        asyncio.run(VisionPreprocessor().preprocess("https://site.example/article-without-image"))


def test_vision_endpoint_returns_grounded_findings(tmp_path: Path) -> None:
    app = create_app()
    container = create_test_container()
    app.dependency_overrides[get_container] = lambda: container
    client = TestClient(app)

    image_path = tmp_path / "route_fixture.png"
    _write_fixture_png(image_path)
    response = client.post("/vision/analyze", json={"image_uri": str(image_path), "prompt": "describe scene"})
    payload = response.json()
    assert response.status_code == 200
    assert payload["summary"]
    assert payload["findings"]
    assert all("[evidence:" in item for item in payload["findings"])
    assert payload["confidence"] > 0

    invalid_response = client.post("/vision/analyze", json={"image_uri": "https://example.com/file.txt"})
    assert invalid_response.status_code == 422

    app.dependency_overrides.clear()
