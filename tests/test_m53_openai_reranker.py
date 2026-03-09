from __future__ import annotations

import asyncio

import pytest

from app.rag.reranker import OpenAILLMReranker, build_reranker


def test_openai_reranker_requires_api_key_or_base_url() -> None:
    with pytest.raises(RuntimeError):
        build_reranker(
            enabled=True,
            provider="openai",
            cross_encoder_model="qwen3:4b",
            openai_api_key=None,
            openai_base_url=None,
        )


def test_openai_reranker_applies_llm_ranked_order(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Response:
        class _Choice:
            class _Message:
                content = '{"ranking":[{"index":1,"score":1.0},{"index":0,"score":0.2}]}'

            message = _Message()

        choices = [_Choice()]

    class _Completions:
        async def create(self, **kwargs):  # noqa: ARG002
            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    reranker = OpenAILLMReranker(model_name="qwen3:4b", base_url="http://localhost:11434/v1")
    monkeypatch.setattr(reranker, "_get_client", lambda: _Client())
    ranked = asyncio.run(
        reranker.rerank(
            query="Which candidate is about aurora?",
            candidates=[
                {"source": "doc-a", "snippet": "maintenance checklist", "score": 0.9},
                {"source": "doc-b", "snippet": "aurora dashboard summary", "score": 0.5},
            ],
            top_k=2,
        )
    )
    assert ranked
    assert ranked[0]["source"] == "doc-b"
