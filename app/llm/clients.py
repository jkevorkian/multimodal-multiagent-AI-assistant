from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

from app.interfaces.llm import LLMClient


@dataclass(frozen=True)
class LLMSelection:
    client: LLMClient
    provider_name: str


class GroundedHeuristicLLMClient:
    async def generate(self, prompt: str, context: list[str] | None = None) -> str:
        if not context:
            return "I do not have indexed context yet. Ingest documents first."
        top_context = " ".join(segment.strip() for segment in context[:3] if segment.strip())
        return f"Grounded answer for '{prompt}': {top_context}"


class OpenAILLMClient:
    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 500,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client: Any | None = None

    async def generate(self, prompt: str, context: list[str] | None = None) -> str:
        client = self._get_client()
        evidence = "\n".join(f"- {item}" for item in (context or []) if item.strip())
        system_prompt = (
            "You are an evidence-aware assistant.\n"
            "Prioritize provided evidence and cite concrete details from it when available.\n"
            "You may make careful, limited inferences when they are strongly implied by the evidence.\n"
            "When you infer beyond explicit evidence, state it as an inference and keep it conservative.\n"
            "Do not fabricate specific facts, entities, numbers, or events not supported by evidence.\n"
            "Unless the user explicitly asks for translation, answer in the same language as the user's question."
        )
        user_prompt = f"Question:\n{prompt}\n\nEvidence:\n{evidence if evidence else '(none)'}"
        response = await client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        return str(content).strip()

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import AsyncOpenAI
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("openai package is required for OpenAI LLM provider") from exc
        # Some OpenAI-compatible local servers do not enforce API keys.
        self._client = AsyncOpenAI(api_key=self._api_key or "local-placeholder-key", base_url=self._base_url)
        return self._client


def build_llm_client(
    provider: str = "auto",
    model_name: str = "gpt-4.1-mini",
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 500,
) -> LLMSelection:
    normalized_provider = provider.strip().lower()

    if normalized_provider == "openai":
        openai_available = importlib.util.find_spec("openai") is not None
        can_use_openai = (bool(api_key) or bool(base_url)) and openai_available
        if not can_use_openai:
            raise RuntimeError("llm provider 'openai' requires MMAA_LLM_API_KEY or MMAA_LLM_BASE_URL and openai package")
        return LLMSelection(
            client=OpenAILLMClient(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            provider_name="openai",
        )

    if normalized_provider in {"auto", "heuristic", "deterministic", "fallback"}:
        return LLMSelection(client=GroundedHeuristicLLMClient(), provider_name="heuristic")

    raise ValueError(f"Unsupported LLM provider '{provider}'")
