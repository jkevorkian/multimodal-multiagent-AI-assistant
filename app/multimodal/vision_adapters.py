from __future__ import annotations

from typing import Any, Protocol


class VisionResponseAdapter(Protocol):
    def extract_text(self, response: Any) -> str:
        ...

    def is_remote_url_unsupported_error(self, exc: Exception) -> bool:
        ...


class OpenAICompatibleVisionAdapter:
    def extract_text(self, response: Any) -> str:
        try:
            message = response.choices[0].message
        except Exception:
            return ""
        content = getattr(message, "content", None)
        return self._normalize_content(content)

    def is_remote_url_unsupported_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return "image url" in message and "not currently supported" in message

    def _normalize_content(self, value: Any) -> str:
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
                text = str(item).strip()
                if text:
                    parts.append(text)
            return " ".join(parts).strip()
        return str(value).strip()


class QwenVisionAdapter(OpenAICompatibleVisionAdapter):
    def extract_text(self, response: Any) -> str:
        text = super().extract_text(response)
        if text:
            return text
        try:
            message = response.choices[0].message
        except Exception:
            return ""
        reasoning = getattr(message, "reasoning", None)
        return self._normalize_content(reasoning)


def build_vision_response_adapter(model_name: str) -> VisionResponseAdapter:
    normalized = model_name.strip().lower()
    if normalized.startswith("qwen"):
        return QwenVisionAdapter()
    return OpenAICompatibleVisionAdapter()

