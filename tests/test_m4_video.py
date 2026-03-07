import asyncio

from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.dependencies import create_test_container, get_container
from app.main import create_app
from app.video import TemporalAggregator, VideoAnalysisAdapter, VideoFrameSampler
from app.video.frame_sampler import FrameSamplingResult, SampledFrame


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


class _VideoClientStub:
    async def analyze_video(
        self,
        video_uri: str,
        prompt: str | None = None,  # noqa: ARG002
        sample_fps: float = 1.0,  # noqa: ARG002
        max_frames: int = 32,  # noqa: ARG002
    ) -> str:
        return "Car appears at intersection; light turns green; car exits frame."


class _VisionClientStub:
    async def analyze_image(self, image_uri: str, prompt: str | None = None) -> str:  # noqa: ARG002
        return "Frame shows a car entering and leaving an intersection."


class _FrameSamplerWithImagePayloads:
    def sample(self, video_uri: str, sample_fps: float = 1.0, max_frames: int = 32) -> FrameSamplingResult:  # noqa: ARG002
        return FrameSamplingResult(
            source=video_uri,
            estimated_duration_sec=2.0,
            sample_fps=sample_fps,
            requested_max_frames=max_frames,
            processed_frames=2,
            budget_limited=False,
            frames=[
                SampledFrame(
                    index=0,
                    timestamp_sec=0.0,
                    cue="frame_000",
                    image_uri="data:image/jpeg;base64,ZmFrZQ==",
                ),
                SampledFrame(
                    index=1,
                    timestamp_sec=1.0,
                    cue="frame_001",
                    image_uri="data:image/jpeg;base64,ZmFrZTI=",
                ),
            ],
            extraction_mode="decoded",
        )


class _URLResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self, size: int = -1) -> bytes:  # noqa: ARG002
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ARG002
        return None


def test_frame_sampler_requires_decode_backend(monkeypatch, tmp_path) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake-video")
    monkeypatch.setattr("app.video.frame_sampler._load_cv2", lambda: None)
    sampler = VideoFrameSampler()
    try:
        sampler.sample(video_uri=str(video_path), sample_fps=1.0, max_frames=4)
    except ValueError as exc:
        assert "opencv decode backend unavailable" in str(exc)
    else:
        raise AssertionError("Expected sampler to fail when cv2 is unavailable")


def test_frame_sampler_respects_sampling_and_latency_budget(monkeypatch, tmp_path) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake-video")
    monkeypatch.setattr("app.video.frame_sampler._load_cv2", lambda: _FakeCV2())

    sampler = VideoFrameSampler(latency_budget_ms=50, estimated_ms_per_frame=10.0, max_window_sec=30.0)
    sampled = sampler.sample(video_uri=str(video_path), sample_fps=2.0, max_frames=20)
    assert sampled.processed_frames == 5
    assert sampled.budget_limited is True
    assert sampled.frames
    assert sampled.frames[0].timestamp_sec == 0.0
    assert sampled.frames[1].timestamp_sec == 0.5
    assert sampled.frames[-1].timestamp_sec == 2.0
    assert sampled.extraction_mode == "decoded"


def test_frame_sampler_decodes_remote_video_url(monkeypatch) -> None:
    monkeypatch.setattr("app.video.frame_sampler._load_cv2", lambda: _FakeCV2())
    monkeypatch.setattr("app.video.frame_sampler.urlopen", lambda request, timeout: _URLResponse(b"remote-video-bytes"))

    sampler = VideoFrameSampler(latency_budget_ms=100, estimated_ms_per_frame=10.0, max_window_sec=30.0)
    sampled = sampler.sample(video_uri="https://example.com/sample.mp4", sample_fps=2.0, max_frames=8)

    assert sampled.extraction_mode == "decoded"
    assert sampled.processed_frames == 8
    assert sampled.frames
    assert sampled.frames[0].image_uri is not None
    assert sampled.frames[0].image_uri.startswith("data:image/jpeg;base64,")


def test_temporal_aggregator_outputs_timeline_ordered_events() -> None:
    sampling = FrameSamplingResult(
        source="https://example.com/scene.mp4",
        estimated_duration_sec=3.0,
        sample_fps=1.0,
        requested_max_frames=4,
        processed_frames=3,
        budget_limited=False,
        frames=[
            SampledFrame(index=0, timestamp_sec=0.0, cue="frame_000"),
            SampledFrame(index=1, timestamp_sec=1.0, cue="frame_001"),
            SampledFrame(index=2, timestamp_sec=2.0, cue="frame_002"),
        ],
        extraction_mode="decoded",
    )
    aggregation = TemporalAggregator(max_key_events=5).aggregate(
        video_uri="https://example.com/scene.mp4",
        prompt="Summarize events",
        raw_summary="First event; second event; third event.",
        sampling=sampling,
    )
    assert aggregation.key_events
    assert all("[t=" in event for event in aggregation.key_events)
    assert all("[source=https://example.com/scene.mp4]" in event for event in aggregation.key_events)
    timestamps = [float(event.split("[t=")[1].split("s]")[0]) for event in aggregation.key_events]
    assert timestamps == sorted(timestamps)


def test_video_adapter_composes_client_sampler_aggregator() -> None:
    adapter = VideoAnalysisAdapter(
        video_client=_VideoClientStub(),  # type: ignore[arg-type]
        vision_client=_VisionClientStub(),  # type: ignore[arg-type]
        frame_sampler=_FrameSamplerWithImagePayloads(),  # type: ignore[arg-type]
        temporal_aggregator=TemporalAggregator(max_key_events=5),
        require_frame_level_findings=True,
    )
    result = asyncio.run(
        adapter.analyze(
            video_uri="file:///tmp/traffic.mp4",
            prompt="Describe timeline",
            sample_fps=1.0,
            max_frames=10,
        )
    )
    assert result.summary
    assert result.key_events
    assert result.processed_frames > 0
    assert 0.1 <= result.confidence <= 0.95


def test_video_adapter_uses_frame_level_vision_findings_when_available() -> None:
    adapter = VideoAnalysisAdapter(
        video_client=_VideoClientStub(),  # type: ignore[arg-type]
        vision_client=_VisionClientStub(),  # type: ignore[arg-type]
        frame_sampler=_FrameSamplerWithImagePayloads(),  # type: ignore[arg-type]
        temporal_aggregator=TemporalAggregator(max_key_events=5),
    )
    result = asyncio.run(
        adapter.analyze(
            video_uri="file:///tmp/demo.mp4",
            prompt="Focus on traffic state transitions.",
            sample_fps=1.0,
            max_frames=6,
        )
    )
    assert result.key_events
    assert any("Frame shows a car entering and leaving an intersection." in event for event in result.key_events)
    assert result.processed_frames == 2


def test_video_endpoint_returns_timeline_events_and_budget_limited_frames(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.video.frame_sampler._load_cv2", lambda: _FakeCV2())

    local_video = tmp_path / "route_demo.mp4"
    local_video.write_bytes(b"route-video")

    app = create_app()
    container = create_test_container()
    app.dependency_overrides[get_container] = lambda: container
    client = TestClient(app)

    original_budget = settings.multimodal_video_latency_budget_ms
    original_estimate = settings.multimodal_video_estimated_ms_per_frame
    original_key_events = settings.multimodal_video_max_key_events
    try:
        settings.multimodal_video_latency_budget_ms = 24
        settings.multimodal_video_estimated_ms_per_frame = 12.0
        settings.multimodal_video_max_key_events = 5
        response = client.post(
            "/video/analyze",
            json={
                "video_uri": local_video.as_uri(),
                "prompt": "Describe key events.",
                "sample_fps": 2.0,
                "max_frames": 40,
            },
        )
    finally:
        settings.multimodal_video_latency_budget_ms = original_budget
        settings.multimodal_video_estimated_ms_per_frame = original_estimate
        settings.multimodal_video_max_key_events = original_key_events
        app.dependency_overrides.clear()

    payload = response.json()
    assert response.status_code == 200
    assert payload["summary"]
    assert payload["key_events"]
    assert all("[t=" in event for event in payload["key_events"])
    assert all(f"[source={local_video.as_uri()}]" in event for event in payload["key_events"])
    assert payload["processed_frames"] == 2
