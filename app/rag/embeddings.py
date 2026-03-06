from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import math
import re
from dataclasses import dataclass
from typing import Any

from app.interfaces.embedding import EmbeddingClient

_OPENAI_MODEL_DEFAULT_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


@dataclass(frozen=True)
class EmbeddingSelection:
    client: EmbeddingClient
    provider_name: str
    dimensions: int | None


class DeterministicEmbeddingClient:
    def __init__(self, dimensions: int = 64) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be > 0")
        self.dimensions = dimensions

    async def embed_text(self, text: str) -> list[float]:
        return (await self.embed_texts([text]))[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_tokens(re.findall(r"[a-z0-9]+", text.lower())) for text in texts]

    def _embed_tokens(self, tokens: list[str]) -> list[float]:
        if not tokens:
            return [0.0 for _ in range(self.dimensions)]

        vector = [0.0 for _ in range(self.dimensions)]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for index in range(0, len(digest), 2):
                slot = ((digest[index] << 8) | digest[index + 1]) % self.dimensions
                sign = 1.0 if digest[index] % 2 == 0 else -1.0
                magnitude = 0.5 + (digest[index + 1] / 255.0)
                vector[slot] += sign * magnitude

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return [0.0 for _ in range(self.dimensions)]
        return [value / norm for value in vector]


class OpenAIEmbeddingClient:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        dimensions: int | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self.dimensions = dimensions if dimensions and dimensions > 0 else _OPENAI_MODEL_DEFAULT_DIMENSIONS.get(model)
        self._client: Any | None = None

    async def embed_text(self, text: str) -> list[float]:
        return (await self.embed_texts([text]))[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        payload: dict[str, Any] = {"model": self._model, "input": texts}
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        response = await client.embeddings.create(**payload)
        ordered = sorted(response.data, key=lambda row: row.index)
        return [list(row.embedding) for row in ordered]

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import AsyncOpenAI
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("openai package is required for OpenAI embeddings") from exc
        # Some OpenAI-compatible local servers do not enforce API keys.
        self._client = AsyncOpenAI(api_key=self._api_key or "local-placeholder-key", base_url=self._base_url)
        return self._client


class SentenceTransformerEmbeddingClient:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        normalize_embeddings: bool = True,
    ) -> None:
        self._model_name = model_name
        self._normalize_embeddings = normalize_embeddings
        self._model: Any | None = None
        self.dimensions: int | None = None

    async def embed_text(self, text: str) -> list[float]:
        return (await self.embed_texts([text]))[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = await asyncio.to_thread(self._encode, texts)
        if embeddings and self.dimensions is None:
            self.dimensions = len(embeddings[0])
        return embeddings

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        vectors = model.encode(
            texts,
            normalize_embeddings=self._normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("sentence-transformers package is required for local neural embeddings") from exc
        self._model = SentenceTransformer(self._model_name)
        return self._model


def build_embedding_client(
    provider: str = "deterministic",
    model_name: str = "text-embedding-3-small",
    sentence_transformer_model: str = "BAAI/bge-small-en-v1.5",
    api_key: str | None = None,
    base_url: str | None = None,
    deterministic_dimensions: int = 64,
    requested_dimensions: int | None = None,
) -> EmbeddingSelection:
    normalized_provider = provider.strip().lower()

    if normalized_provider in {"sentence_transformers", "sentence-transformers"}:
        if importlib.util.find_spec("sentence_transformers") is None:
            raise RuntimeError("sentence-transformers package is required for sentence_transformers provider")
        client = SentenceTransformerEmbeddingClient(model_name=sentence_transformer_model)
        return EmbeddingSelection(client=client, provider_name="sentence_transformers", dimensions=requested_dimensions)

    if normalized_provider == "openai":
        if importlib.util.find_spec("openai") is None:
            raise RuntimeError("openai package is required for openai embedding provider")
        if not api_key and not base_url:
            raise RuntimeError(
                "embedding provider 'openai' requires MMAA_RAG_OPENAI_API_KEY or MMAA_RAG_OPENAI_BASE_URL"
            )
        client = OpenAIEmbeddingClient(
            model=model_name,
            api_key=api_key,
            dimensions=requested_dimensions,
            base_url=base_url,
        )
        return EmbeddingSelection(client=client, provider_name="openai", dimensions=client.dimensions)

    if normalized_provider in {"auto", "neural"}:
        if (api_key or base_url) and importlib.util.find_spec("openai") is not None:
            client = OpenAIEmbeddingClient(
                model=model_name,
                api_key=api_key,
                dimensions=requested_dimensions,
                base_url=base_url,
            )
            return EmbeddingSelection(client=client, provider_name="openai", dimensions=client.dimensions)

    if normalized_provider in {"deterministic", "hash", "auto", "neural"}:
        client = DeterministicEmbeddingClient(dimensions=deterministic_dimensions)
        return EmbeddingSelection(client=client, provider_name="deterministic", dimensions=client.dimensions)

    raise ValueError(f"Unsupported embedding provider '{provider}'")


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0

    size = min(len(left), len(right))
    numerator = sum(left[index] * right[index] for index in range(size))
    left_norm = math.sqrt(sum(value * value for value in left[:size]))
    right_norm = math.sqrt(sum(value * value for value in right[:size]))
    denominator = left_norm * right_norm
    if denominator == 0:
        return 0.0
    return numerator / denominator
