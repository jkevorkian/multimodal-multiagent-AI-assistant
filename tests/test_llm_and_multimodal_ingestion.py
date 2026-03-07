import asyncio

import pytest

from app.llm.clients import build_llm_client
from app.multimodal.clients import OpenAIVisionClient, build_multimodal_clients
from app.rag.embeddings import build_embedding_client
from app.rag.ingestion import DocumentIngestionService


class _VectorStoreRecorder:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    async def upsert(self, ids: list[str], vectors: list[list[float]], metadata: list[dict]) -> None:
        for row_id, vector, item_metadata in zip(ids, vectors, metadata, strict=True):
            self.rows.append({"id": row_id, "vector": vector, "metadata": item_metadata})

    async def search(self, vector: list[float], top_k: int) -> list[dict]:  # noqa: ARG002
        return []

    async def keyword_search(self, query: str, top_k: int) -> list[dict]:  # noqa: ARG002
        return []


class _VisionStub:
    async def analyze_image(self, image_uri: str, prompt: str | None = None) -> str:  # noqa: ARG002
        return f"vision-summary::{image_uri}"


class _VideoStub:
    async def analyze_video(
        self,
        video_uri: str,
        prompt: str | None = None,  # noqa: ARG002
        sample_fps: float = 1.0,  # noqa: ARG002
        max_frames: int = 32,  # noqa: ARG002
    ) -> str:
        return f"video-summary::{video_uri}"


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


def test_llm_auto_provider_falls_back_without_api_key() -> None:
    selection = build_llm_client(provider="auto", api_key=None)
    assert selection.provider_name == "heuristic"
    answer = asyncio.run(selection.client.generate("What is the status?", ["context line"]))
    assert "Grounded answer" in answer


def test_openai_compatible_endpoints_can_use_base_url_without_api_key() -> None:
    llm_selection = build_llm_client(
        provider="openai",
        model_name="llama3.2:3b",
        api_key=None,
        base_url="http://localhost:11434/v1",
    )
    assert llm_selection.provider_name == "openai"
    assert getattr(llm_selection.client, "_base_url", None) == "http://localhost:11434/v1"

    embedding_selection = build_embedding_client(
        provider="openai",
        model_name="nomic-embed-text",
        api_key=None,
        base_url="http://localhost:11434/v1",
    )
    assert embedding_selection.provider_name == "openai"
    assert getattr(embedding_selection.client, "_base_url", None) == "http://localhost:11434/v1"

    multimodal_selection = build_multimodal_clients(
        provider="openai",
        vision_model="qwen2.5vl:7b",
        api_key=None,
        base_url="http://localhost:11434/v1",
    )
    assert multimodal_selection.provider_name == "openai+heuristic-video"
    assert getattr(multimodal_selection.vision, "_base_url", None) == "http://localhost:11434/v1"


def test_openai_embedding_provider_requires_api_key_or_base_url() -> None:
    with pytest.raises(RuntimeError):
        build_embedding_client(provider="openai", api_key=None, base_url=None)


def test_openai_vision_client_serializes_file_uri_as_data_url(tmp_path) -> None:
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nplaceholder")
    client = OpenAIVisionClient(model="qwen3-vl:4b", base_url="http://localhost:11434/v1")
    encoded = client._prepare_image_url(image_path.as_uri())
    assert encoded.startswith("data:image/png;base64,")


def test_image_and_video_are_ingested_with_modality_metadata(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.video.frame_sampler._load_cv2", lambda: _FakeCV2())
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"fake-video")

    store = _VectorStoreRecorder()
    embeddings = build_embedding_client(provider="deterministic", deterministic_dimensions=16).client
    ingestion = DocumentIngestionService(
        embedding_client=embeddings,
        vector_store=store,
        vision_client=_VisionStub(),  # type: ignore[arg-type]
        video_client=_VideoStub(),  # type: ignore[arg-type]
        chunk_size=120,
        chunk_overlap=20,
    )

    summary = asyncio.run(
        ingestion.ingest(
            sources=["https://assets.example.com/diagram.png", video_path.as_uri()],
            source_type="mixed",
        )
    )
    assert summary.accepted_sources == 2
    assert summary.indexed_chunks >= 2
    assert any(row["metadata"].get("modality") == "image" for row in store.rows)
    assert any(row["metadata"].get("modality") == "video" for row in store.rows)
    assert any("vision-summary" in row["metadata"].get("snippet", "") for row in store.rows)
    assert any("Key events:" in row["metadata"].get("snippet", "") for row in store.rows)
    assert any("[t=" in row["metadata"].get("snippet", "") for row in store.rows)
