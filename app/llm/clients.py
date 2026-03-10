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
            "You may make careful inferences when implied by the evidence, but always state when an inference is being made.\n"
            "Do not fabricate specific facts, entities, numbers, or events not supported by evidence. But you can make inferences if you do detail it being an inference.\n"
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
        return self._extract_text(response)

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

    @staticmethod
    def _extract_text(response: Any) -> str:
        try:
            message = response.choices[0].message
        except Exception:
            return ""

        text = OpenAILLMClient._normalize_message_field(getattr(message, "content", None))
        if text:
            return text

        # Some OpenAI-compatible local providers (for example Qwen derivatives)
        # can place textual output in `reasoning` when `content` is empty.
        reasoning = OpenAILLMClient._normalize_message_field(getattr(message, "reasoning", None))
        if reasoning:
            return reasoning
        return ""

    @staticmethod
    def _normalize_message_field(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, str):
                    text = item.strip()
                    if text:
                        parts.append(text)
                    continue
                if isinstance(item, dict):
                    text_candidate = str(item.get("text", "")).strip()
                    if text_candidate:
                        parts.append(text_candidate)
                    continue
                text = str(getattr(item, "text", item)).strip()
                if text:
                    parts.append(text)
            return " ".join(parts).strip()
        return str(value).strip()


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
