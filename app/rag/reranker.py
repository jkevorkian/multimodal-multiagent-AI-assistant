from __future__ import annotations

import asyncio
import importlib.util
import re
from dataclasses import dataclass
from typing import Any

from app.interfaces.reranker import Reranker

#TODO: implement a nicer strategy/adapter approach so the interfacae of a reranker is well defined and no necessity exists for the build_reranker funcion to have a lot of if statements. Does this make sense?

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


@dataclass(frozen=True)
class RerankerSelection:
    reranker: Reranker
    provider_name: str


def build_reranker(
    enabled: bool = True,
    provider: str = "lexical",
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> RerankerSelection:
    if not enabled:
        return RerankerSelection(reranker=NoOpReranker(), provider_name="disabled")

    normalized_provider = provider.strip().lower()
    if normalized_provider in {"none", "disabled"}:
        return RerankerSelection(reranker=NoOpReranker(), provider_name="disabled")

    if normalized_provider == "cross_encoder":
        if importlib.util.find_spec("sentence_transformers") is None:
            raise RuntimeError("sentence-transformers package is required for cross_encoder reranker")
        return RerankerSelection(reranker=CrossEncoderReranker(model_name=cross_encoder_model), provider_name="cross_encoder")

    if normalized_provider == "lexical":
        return RerankerSelection(reranker=LexicalOverlapReranker(), provider_name="lexical")

    if normalized_provider == "auto":
        return RerankerSelection(reranker=LexicalOverlapReranker(), provider_name="lexical")

    raise ValueError(f"Unsupported reranker provider '{provider}'")
