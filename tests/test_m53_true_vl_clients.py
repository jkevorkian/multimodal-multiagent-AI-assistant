from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.interfaces.multimodal_embedding import MultimodalEmbeddingInput
from app.rag.multimodal_embeddings import Qwen3VLNativeEmbeddingClient, build_multimodal_embedding_client
from app.rag.reranker import Qwen3VLNativeReranker, build_reranker


def test_qwen3_vl_embedding_provider_requires_base_url() -> None:
    with pytest.raises(RuntimeError):
        build_multimodal_embedding_client(
            provider="qwen3_vl",
            model_name="Qwen/Qwen3-VL-Embedding-2B",
            base_url=None,
        )


def test_qwen3_vl_embedding_client_sends_multimodal_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "data": [
                    {"embedding": [0.1, 0.2]},
                    {"embedding": [0.3, 0.4]},
                    {"embedding": [0.5, 0.6]},
                ]
            }

    class _Client:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ARG002
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ARG002
            return False

        async def post(self, url: str, json: dict[str, Any], headers: dict[str, str]):  # noqa: A002
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return _Response()

    monkeypatch.setattr("app.rag.multimodal_embeddings.httpx.AsyncClient", _Client)

    client = Qwen3VLNativeEmbeddingClient(
        model="Qwen/Qwen3-VL-Embedding-2B",
        base_url="http://localhost:8001/v1",
    )
    vectors = asyncio.run(
        client.embed(
            [
                MultimodalEmbeddingInput(modality="text", text="hello world"),
                MultimodalEmbeddingInput(modality="image", text="an image", source_uri="https://example.com/demo.png"),
                MultimodalEmbeddingInput(modality="video", text="a video", source_uri="https://example.com/demo.mp4"),
            ]
        )
    )

    assert len(vectors) == 3
    assert captured["url"] == "http://localhost:8001/v1/embeddings"
    payload = captured["json"]
    assert payload["model"] == "Qwen/Qwen3-VL-Embedding-2B"
    assert payload["input"][0] == "hello world"
    image_parts = payload["input"][1]
    video_parts = payload["input"][2]
    assert any(part.get("type") == "image_url" for part in image_parts)
    assert any(part.get("type") == "video_url" for part in video_parts)


def test_qwen3_vl_embedding_client_deduplicates_v1_path(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"data": [{"embedding": [0.1, 0.2]}]}

    class _Client:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ARG002
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ARG002
            return False

        async def post(self, url: str, json: dict[str, Any], headers: dict[str, str]):  # noqa: A002, ARG002
            captured["url"] = url
            return _Response()

    monkeypatch.setattr("app.rag.multimodal_embeddings.httpx.AsyncClient", _Client)

    client = Qwen3VLNativeEmbeddingClient(
        model="Qwen/Qwen3-VL-Embedding-2B",
        base_url="http://localhost:8001/v1",
        endpoint_path="/v1/embeddings",
    )
    _ = asyncio.run(client.embed([MultimodalEmbeddingInput(modality="text", text="hello")]))

    assert captured["url"] == "http://localhost:8001/v1/embeddings"


def test_qwen3_vl_reranker_provider_requires_base_url() -> None:
    with pytest.raises(RuntimeError):
        build_reranker(
            enabled=True,
            provider="qwen3_vl",
            cross_encoder_model="Qwen/Qwen3-VL-Reranker-2B",
            openai_api_key=None,
            openai_base_url=None,
        )


def test_qwen3_vl_reranker_uses_scores_from_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"results": [{"index": 1, "score": 0.95}, {"index": 0, "score": 0.2}]}

    class _Client:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ARG002
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ARG002
            return False

        async def post(self, url: str, json: dict[str, Any], headers: dict[str, str]):  # noqa: A002
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return _Response()

    monkeypatch.setattr("app.rag.reranker.httpx.AsyncClient", _Client)

    reranker = Qwen3VLNativeReranker(
        model_name="Qwen/Qwen3-VL-Reranker-2B",
        base_url="http://localhost:8001/v1",
    )
    ranked = asyncio.run(
        reranker.rerank(
            query="aurora dashboard",
            candidates=[
                {"source": "a.txt", "snippet": "maintenance report", "modality": "text", "score": 0.9},
                {
                    "source": "https://example.com/aurora.png",
                    "snippet": "aurora visual dashboard",
                    "modality": "image",
                    "score": 0.4,
                },
            ],
            top_k=2,
        )
    )

    assert ranked
    assert ranked[0]["source"] == "https://example.com/aurora.png"
    assert captured["url"] == "http://localhost:8001/v1/rerank"
    payload = captured["json"]
    assert payload["model"] == "Qwen/Qwen3-VL-Reranker-2B"
    assert payload["query"][0]["type"] == "text"
    assert any(part.get("type") == "image_url" for part in payload["documents"][1])
