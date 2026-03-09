from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import httpx

from app.interfaces.reranker import Reranker


class NoOpReranker:
    async def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        _ = query
        ranked = sorted(candidates, key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return ranked[:top_k]


class LexicalOverlapReranker:
    async def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        rescored: list[dict] = []

        for item in candidates:
            snippet = str(item.get("snippet", ""))
            snippet_terms = set(re.findall(r"[a-z0-9]+", snippet.lower()))
            overlap = len(query_terms & snippet_terms)
            prior_score = float(item.get("score", 0.0))
            composite = (prior_score * 0.7) + (overlap * 0.3)
            rescored.append({**item, "score": composite})

        rescored.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
        return rescored[:top_k]


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: Any | None = None

    async def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        if not candidates:
            return []
        scores = await asyncio.to_thread(self._score, query, candidates)
        rescored = [{**item, "score": score} for item, score in zip(candidates, scores, strict=True)]
        rescored.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
        return rescored[:top_k]

    def _score(self, query: str, candidates: list[dict]) -> list[float]:
        model = self._get_model()
        pairs = [(query, str(item.get("snippet", ""))) for item in candidates]
        values = model.predict(pairs)
        return [float(value) for value in values]

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("sentence-transformers package is required for cross-encoder reranking") from exc
        self._model = CrossEncoder(self._model_name)
        return self._model


class Qwen3VLNativeReranker:
    """True multimodal reranker client using image/video-aware document payloads."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        endpoint_path: str = "/rerank",
        api_key: str | None = None,
        timeout_sec: float = 60.0,
        max_candidates: int = 40,
    ) -> None:
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._endpoint_path = endpoint_path if endpoint_path.startswith("/") else f"/{endpoint_path}"
        self._api_key = api_key
        self._timeout_sec = timeout_sec
        self._max_candidates = max(1, max_candidates)
        self._fallback = LexicalOverlapReranker()

    async def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        if not candidates:
            return []
        limited_candidates = list(candidates[: self._max_candidates])
        try:
            payload = {
                "model": self._model_name,
                "query": [{"type": "text", "text": query}],
                "documents": [self._candidate_to_document(item) for item in limited_candidates],
                "top_n": min(top_k, len(limited_candidates)),
            }
            response_payload = await self._post_json(payload)
            parsed = self._extract_ranking(response_payload, candidate_count=len(limited_candidates))
            if not parsed:
                return await self._fallback.rerank(query=query, candidates=limited_candidates, top_k=top_k)

            index_to_score = {index: score for index, score in parsed}
            rescored = []
            for idx, item in enumerate(limited_candidates):
                prior = float(item.get("score", 0.0))
                rerank_score = index_to_score.get(idx)
                if rerank_score is None:
                    rescored.append({**item, "score": prior * 0.5})
                else:
                    composite = (prior * 0.35) + (rerank_score * 0.65)
                    rescored.append({**item, "score": composite})
            rescored.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
            return rescored[:top_k]
        except Exception:
            return await self._fallback.rerank(query=query, candidates=limited_candidates, top_k=top_k)

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
            raise ValueError("qwen3_vl reranker endpoint returned a non-object payload")
        return data

    def _build_request_url(self) -> str:
        endpoint_path = self._endpoint_path
        if self._base_url.endswith("/v1") and endpoint_path.startswith("/v1/"):
            endpoint_path = endpoint_path[len("/v1") :]
        return f"{self._base_url}{endpoint_path}"

    @staticmethod
    def _candidate_to_document(item: dict[str, Any]) -> list[dict[str, Any]]:
        snippet = str(item.get("snippet", "")).strip()
        source = str(item.get("source", "")).strip()
        modality = str(item.get("modality", "text")).strip().lower()
        parts: list[dict[str, Any]] = []
        if snippet:
            parts.append({"type": "text", "text": snippet})
        if modality == "image" and source:
            parts.append({"type": "image_url", "image_url": {"url": _resolve_media_uri(source, modality="image")}})
        elif modality == "video" and source:
            parts.append({"type": "video_url", "video_url": {"url": _resolve_media_uri(source, modality="video")}})
        if not parts:
            parts.append({"type": "text", "text": source})
        return parts

    @staticmethod
    def _extract_ranking(payload: dict[str, Any], candidate_count: int) -> list[tuple[int, float]]:
        rows = payload.get("data")
        ranking = Qwen3VLNativeReranker._extract_rows(rows, candidate_count=candidate_count)
        if ranking:
            return ranking

        rows = payload.get("results")
        ranking = Qwen3VLNativeReranker._extract_rows(rows, candidate_count=candidate_count)
        if ranking:
            return ranking

        rows = payload.get("rankings")
        ranking = Qwen3VLNativeReranker._extract_rows(rows, candidate_count=candidate_count)
        if ranking:
            return ranking

        return []

    @staticmethod
    def _extract_rows(rows: Any, *, candidate_count: int) -> list[tuple[int, float]]:
        if not isinstance(rows, list):
            return []
        parsed: list[tuple[int, float]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            index = row.get("index", row.get("id"))
            score = row.get("relevance_score", row.get("score", 0.0))
            try:
                parsed_index = int(index)
                parsed_score = float(score)
            except (TypeError, ValueError):
                continue
            if 0 <= parsed_index < candidate_count:
                parsed.append((parsed_index, parsed_score))
        return Qwen3VLNativeReranker._dedupe_ranking(parsed)

    @staticmethod
    def _dedupe_ranking(items: list[tuple[int, float]]) -> list[tuple[int, float]]:
        seen: set[int] = set()
        unique: list[tuple[int, float]] = []
        for index, score in items:
            if index in seen:
                continue
            seen.add(index)
            unique.append((index, score))
        return unique


class OpenAILLMReranker:
    """LLM-based reranker for OpenAI-compatible endpoints (including local Ollama /v1)."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_candidates: int = 24,
    ) -> None:
        self._model_name = model_name
        self._api_key = api_key
        self._base_url = base_url
        self._max_candidates = max(1, max_candidates)
        self._client: Any | None = None
        self._fallback = LexicalOverlapReranker()

    async def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        if not candidates:
            return []
        limited_candidates = list(candidates[: self._max_candidates])
        try:
            prompt = self._build_prompt(query=query, candidates=limited_candidates)
            response = await self._get_client().chat.completions.create(
                model=self._model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a retrieval reranker. Return only JSON with ranked candidate indices and scores. "
                            "Never include prose."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            raw = self._extract_content(response)
            parsed = self._parse_ranked_indices(raw, candidate_count=len(limited_candidates))
            if not parsed:
                return await self._fallback.rerank(query=query, candidates=limited_candidates, top_k=top_k)

            index_to_score = {index: score for index, score in parsed}
            rescored = []
            for idx, item in enumerate(limited_candidates):
                prior = float(item.get("score", 0.0))
                rerank_score = index_to_score.get(idx)
                if rerank_score is None:
                    rescored.append({**item, "score": prior * 0.5})
                else:
                    composite = (prior * 0.35) + (rerank_score * 0.65)
                    rescored.append({**item, "score": composite})

            rescored.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
            return rescored[:top_k]
        except Exception:
            return await self._fallback.rerank(query=query, candidates=limited_candidates, top_k=top_k)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import AsyncOpenAI
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("openai package is required for openai reranking") from exc
        self._client = AsyncOpenAI(api_key=self._api_key or "local-placeholder-key", base_url=self._base_url)
        return self._client

    @staticmethod
    def _build_prompt(query: str, candidates: list[dict]) -> str:
        lines = [f"Query: {query}", "", "Candidates:"]
        for idx, item in enumerate(candidates):
            snippet = str(item.get("snippet", "")).replace("\n", " ").strip()
            source = str(item.get("source", "unknown"))
            lines.append(f"[{idx}] source={source} snippet={snippet[:700]}")
        lines.extend(
            [
                "",
                "Return JSON only in this format:",
                '{"ranking":[{"index":0,"score":1.0},{"index":1,"score":0.8}]}',
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _extract_content(response: Any) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        if message is None:
            return ""
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            fragments = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    fragments.append(str(item.get("text", "")))
                elif hasattr(item, "text"):
                    fragments.append(str(getattr(item, "text")))
                else:
                    fragments.append(str(item))
            return "\n".join(fragment for fragment in fragments if fragment).strip()
        return str(content).strip()

    @staticmethod
    def _parse_ranked_indices(raw: str, candidate_count: int) -> list[tuple[int, float]]:
        if not raw:
            return []
        try:
            payload = json.loads(raw)
        except Exception:
            payload = None

        ranking: list[tuple[int, float]] = []
        if isinstance(payload, dict) and isinstance(payload.get("ranking"), list):
            for position, item in enumerate(payload["ranking"]):
                if not isinstance(item, dict):
                    continue
                index = item.get("index")
                score = item.get("score", max(0.0, 1.0 - (position * 0.05)))
                try:
                    parsed_index = int(index)
                    parsed_score = float(score)
                except (TypeError, ValueError):
                    continue
                if 0 <= parsed_index < candidate_count:
                    ranking.append((parsed_index, parsed_score))
        elif isinstance(payload, list):
            for position, index in enumerate(payload):
                try:
                    parsed_index = int(index)
                except (TypeError, ValueError):
                    continue
                if 0 <= parsed_index < candidate_count:
                    ranking.append((parsed_index, max(0.0, 1.0 - (position * 0.05))))

        if ranking:
            return OpenAILLMReranker._dedupe_ranking(ranking)

        fallback_indices = []
        for token in re.findall(r"\d+", raw):
            try:
                parsed = int(token)
            except ValueError:
                continue
            if 0 <= parsed < candidate_count:
                fallback_indices.append(parsed)
        fallback_ranked = [
            (idx, max(0.0, 1.0 - (position * 0.05)))
            for position, idx in enumerate(fallback_indices)
        ]
        return OpenAILLMReranker._dedupe_ranking(fallback_ranked)

    @staticmethod
    def _dedupe_ranking(items: list[tuple[int, float]]) -> list[tuple[int, float]]:
        seen: set[int] = set()
        unique: list[tuple[int, float]] = []
        for index, score in items:
            if index in seen:
                continue
            seen.add(index)
            unique.append((index, score))
        return unique


@dataclass(frozen=True)
class RerankerSelection:
    reranker: Reranker
    provider_name: str


def build_reranker(
    enabled: bool = True,
    provider: str = "lexical",
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
    qwen3_vl_endpoint: str = "/rerank",
) -> RerankerSelection:
    if not enabled:
        return RerankerSelection(reranker=NoOpReranker(), provider_name="disabled")

    normalized_provider = provider.strip().lower().replace("-", "_")
    if normalized_provider in {"none", "disabled"}:
        return RerankerSelection(reranker=NoOpReranker(), provider_name="disabled")

    if normalized_provider == "cross_encoder":
        if importlib.util.find_spec("sentence_transformers") is None:
            raise RuntimeError("sentence-transformers package is required for cross_encoder reranker")
        return RerankerSelection(
            reranker=CrossEncoderReranker(model_name=cross_encoder_model),
            provider_name="cross_encoder",
        )

    if normalized_provider in {"qwen3_vl", "qwen3_vl_reranker"}:
        if not openai_base_url:
            raise RuntimeError("qwen3_vl reranker requires MMAA_RAG_RERANKER_OPENAI_BASE_URL")
        return RerankerSelection(
            reranker=Qwen3VLNativeReranker(
                model_name=cross_encoder_model,
                base_url=openai_base_url,
                endpoint_path=qwen3_vl_endpoint,
                api_key=openai_api_key,
            ),
            provider_name="qwen3_vl",
        )

    if normalized_provider == "openai":
        if importlib.util.find_spec("openai") is None:
            raise RuntimeError("openai package is required for openai reranker")
        if not openai_api_key and not openai_base_url:
            raise RuntimeError("openai reranker requires MMAA_RAG_RERANKER_OPENAI_API_KEY or MMAA_RAG_RERANKER_OPENAI_BASE_URL")
        return RerankerSelection(
            reranker=OpenAILLMReranker(
                model_name=cross_encoder_model,
                api_key=openai_api_key,
                base_url=openai_base_url,
            ),
            provider_name="openai",
        )

    if normalized_provider == "lexical":
        return RerankerSelection(reranker=LexicalOverlapReranker(), provider_name="lexical")

    if normalized_provider == "auto":
        return RerankerSelection(reranker=LexicalOverlapReranker(), provider_name="lexical")

    raise ValueError(f"Unsupported reranker provider '{provider}'")


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
