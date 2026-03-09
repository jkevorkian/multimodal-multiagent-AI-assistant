from __future__ import annotations

import base64
import importlib.util
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import httpx

from app.interfaces.multimodal_embedding import MultimodalEmbeddingClient, MultimodalEmbeddingInput
from app.rag.embeddings import DeterministicEmbeddingClient


@dataclass(frozen=True)
class MultimodalEmbeddingSelection:
    client: MultimodalEmbeddingClient
    provider_name: str
    dimensions: int | None


class DeterministicMultimodalEmbeddingClient:
    def __init__(self, dimensions: int = 64) -> None:
        self._text_embedder = DeterministicEmbeddingClient(dimensions=dimensions)
        self.dimensions = dimensions

    async def embed(self, inputs: list[MultimodalEmbeddingInput]) -> list[list[float]]:
        if not inputs:
            return []
        serialized = [self._serialize(item) for item in inputs]
        return await self._text_embedder.embed_texts(serialized)

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed([MultimodalEmbeddingInput(modality="text", text=text)]))[0]

    @staticmethod
    def _serialize(item: MultimodalEmbeddingInput) -> str:
        source = item.source_uri or ""
        return f"[modality={item.modality}] [source={source}] {item.text}".strip()


class OpenAIMultimodalEmbeddingClient:
    """Text-serialized fallback for OpenAI-compatible embedding endpoints."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self.dimensions = dimensions
        self._client: Any | None = None

    async def embed(self, inputs: list[MultimodalEmbeddingInput]) -> list[list[float]]:
        if not inputs:
            return []
        payload: dict[str, Any] = {
            "model": self._model,
            "input": [self._serialize(item) for item in inputs],
        }
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        response = await self._get_client().embeddings.create(**payload)
        ordered = sorted(response.data, key=lambda row: row.index)
        return [list(row.embedding) for row in ordered]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed([MultimodalEmbeddingInput(modality="text", text=text)]))[0]

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import AsyncOpenAI
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("openai package is required for multimodal embeddings") from exc
        self._client = AsyncOpenAI(api_key=self._api_key or "local-placeholder-key", base_url=self._base_url)
        return self._client

    @staticmethod
    def _serialize(item: MultimodalEmbeddingInput) -> str:
        source = item.source_uri or ""
        return f"[modality={item.modality}] [source={source}] {item.text}".strip()


class Qwen3VLNativeEmbeddingClient:
    """True multimodal embedding client using image/video-aware request payloads."""

    def __init__(
        self,
        model: str,
        base_url: str,
        endpoint_path: str = "/embeddings",
        api_key: str | None = None,
        timeout_sec: float = 60.0,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._endpoint_path = endpoint_path if endpoint_path.startswith("/") else f"/{endpoint_path}"
        self._api_key = api_key
        self._timeout_sec = timeout_sec
        self.dimensions: int | None = None

    async def embed(self, inputs: list[MultimodalEmbeddingInput]) -> list[list[float]]:
        if not inputs:
            return []
        payload = {
            "model": self._model,
            "input": [self._to_multimodal_payload(item) for item in inputs],
        }
        response_payload = await self._post_json(payload)
        vectors = self._extract_vectors(response_payload)
        if vectors and self.dimensions is None:
            self.dimensions = len(vectors[0])
        if len(vectors) != len(inputs):
            raise ValueError("multimodal embedding endpoint returned mismatched batch size")
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed([MultimodalEmbeddingInput(modality="text", text=text)]))[0]

    async def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request_url = self._build_request_url()
        async with httpx.AsyncClient(timeout=self._timeout_sec) as client:
            response = await client.post(request_url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("multimodal embedding endpoint returned a non-object payload")
        return data

    def _build_request_url(self) -> str:
        endpoint_path = self._endpoint_path
        if self._base_url.endswith("/v1") and endpoint_path.startswith("/v1/"):
            endpoint_path = endpoint_path[len("/v1") :]
        return f"{self._base_url}{endpoint_path}"

    def _to_multimodal_payload(self, item: MultimodalEmbeddingInput) -> Any:
        modality = item.modality.strip().lower()
        if modality == "text":
            return item.text

        media_uri = _resolve_media_uri(item.source_uri or "", modality=modality)
        content: list[dict[str, Any]] = []
        if item.text.strip():
            content.append({"type": "text", "text": item.text})
        if modality == "image":
            content.append({"type": "image_url", "image_url": {"url": media_uri}})
        elif modality == "video":
            content.append({"type": "video_url", "video_url": {"url": media_uri}})
        else:
            content.append({"type": "text", "text": item.text})
        return content

    @staticmethod
    def _extract_vectors(payload: dict[str, Any]) -> list[list[float]]:
        rows = payload.get("data")
        if isinstance(rows, list):
            vectors: list[list[float]] = []
            for row in rows:
                if isinstance(row, dict) and isinstance(row.get("embedding"), list):
                    vectors.append([float(value) for value in row["embedding"]])
            if vectors:
                return vectors

        rows = payload.get("embeddings")
        if isinstance(rows, list):
            vectors = []
            for row in rows:
                if isinstance(row, list):
                    vectors.append([float(value) for value in row])
            if vectors:
                return vectors

        raise ValueError("multimodal embedding response does not include embeddings in a supported format")


def build_multimodal_embedding_client(
    provider: str = "deterministic",
    model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
    api_key: str | None = None,
    base_url: str | None = None,
    deterministic_dimensions: int = 64,
    requested_dimensions: int | None = None,
    endpoint_path: str = "/embeddings",
) -> MultimodalEmbeddingSelection:
    normalized_provider = provider.strip().lower().replace("-", "_")
    openai_available = importlib.util.find_spec("openai") is not None
    qwen_provider_aliases = {"qwen3_vl", "qwen3_vl_embedding", "qwen_vl_embedding"}

    if normalized_provider in qwen_provider_aliases:
        if not base_url:
            raise RuntimeError("qwen3_vl multimodal embedding provider requires MMAA_RAG_MULTIMODAL_OPENAI_BASE_URL")
        client = Qwen3VLNativeEmbeddingClient(
            model=model_name,
            base_url=base_url,
            endpoint_path=endpoint_path,
            api_key=api_key,
        )
        return MultimodalEmbeddingSelection(client=client, provider_name="qwen3_vl", dimensions=client.dimensions)

    if normalized_provider == "openai":
        if not openai_available:
            raise RuntimeError("openai package is required for openai multimodal embedding provider")
        if not api_key and not base_url:
            raise RuntimeError(
                "multimodal embedding provider requires MMAA_RAG_MULTIMODAL_OPENAI_API_KEY "
                "or MMAA_RAG_MULTIMODAL_OPENAI_BASE_URL"
            )
        client = OpenAIMultimodalEmbeddingClient(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            dimensions=requested_dimensions,
        )
        return MultimodalEmbeddingSelection(client=client, provider_name="openai", dimensions=client.dimensions)

    if normalized_provider in {"auto", "neural"} and base_url and "qwen" in model_name.lower():
        client = Qwen3VLNativeEmbeddingClient(
            model=model_name,
            base_url=base_url,
            endpoint_path=endpoint_path,
            api_key=api_key,
        )
        return MultimodalEmbeddingSelection(client=client, provider_name="qwen3_vl", dimensions=client.dimensions)

    if normalized_provider in {"auto", "neural"} and openai_available and (api_key or base_url):
        client = OpenAIMultimodalEmbeddingClient(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            dimensions=requested_dimensions,
        )
        return MultimodalEmbeddingSelection(client=client, provider_name="openai", dimensions=client.dimensions)

    if normalized_provider in {"deterministic", "hash", "auto", "neural"}:
        client = DeterministicMultimodalEmbeddingClient(dimensions=deterministic_dimensions)
        return MultimodalEmbeddingSelection(client=client, provider_name="deterministic", dimensions=client.dimensions)

    raise ValueError(f"Unsupported multimodal embedding provider '{provider}'")


def _resolve_media_uri(uri: str, *, modality: str) -> str:
    stripped = uri.strip()
    if not stripped:
        return ""
    parsed = urlparse(stripped)
    if parsed.scheme in {"http", "https", "data"}:
        return stripped
    if parsed.scheme == "file":
        local_path = unquote(parsed.path)
        if local_path.startswith("/") and len(local_path) > 2 and local_path[2] == ":":
            local_path = local_path[1:]
        path = Path(local_path)
    else:
        path = Path(stripped)
    if not path.exists() or modality != "image":
        return str(path)

    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"
