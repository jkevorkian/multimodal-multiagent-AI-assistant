from __future__ import annotations

import base64
import importlib.util
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from app.interfaces.video import VideoClient
from app.interfaces.vision import VisionClient
from app.multimodal.vision_adapters import (
    VisionResponseAdapter,
    build_vision_response_adapter,
)


@dataclass(frozen=True)
class MultimodalClientSelection:
    vision: VisionClient
    video: VideoClient
    provider_name: str


class HeuristicVisionClient:
    async def analyze_image(self, image_uri: str, prompt: str | None = None) -> str:
        parsed = urlparse(image_uri)
        filename = Path(parsed.path).name or image_uri
        tags = re.findall(r"[a-z0-9]+", filename.lower())
        hint = ", ".join(tags[:5]) if tags else "unknown-subject"
        prompt_suffix = f" prompt={prompt}" if prompt else ""
        return f"Image analyzed ({hint}).{prompt_suffix}".strip()


class HeuristicVideoClient:
    async def analyze_video(
        self,
        video_uri: str,
        prompt: str | None = None,
        sample_fps: float = 1.0,
        max_frames: int = 32,
    ) -> str:
        parsed = urlparse(video_uri)
        filename = Path(parsed.path).name or video_uri
        tags = re.findall(r"[a-z0-9]+", filename.lower())
        hint = ", ".join(tags[:6]) if tags else "unknown-scene"
        prompt_suffix = f" prompt={prompt}" if prompt else ""
        return (
            f"Video analyzed ({hint}); sample_fps={sample_fps}; max_frames={max_frames}."
            f"{prompt_suffix}"
        ).strip()


class OpenAIVisionClient:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 300,
        remote_fetch_timeout_sec: float = 10.0,
        max_remote_image_bytes: int = 5_000_000,
        response_adapter: VisionResponseAdapter | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._max_tokens = max_tokens
        self._remote_fetch_timeout_sec = remote_fetch_timeout_sec
        self._max_remote_image_bytes = max_remote_image_bytes
        self._response_adapter = response_adapter or build_vision_response_adapter(model)
        self._client: Any | None = None

    async def analyze_image(self, image_uri: str, prompt: str | None = None) -> str:
        client = self._get_client()
        instruction = prompt or "Describe the image focusing on entities, actions, and salient details."
        image_payload = self._prepare_image_url(image_uri)
        try:
            response = await client.chat.completions.create(
                model=self._model,
                temperature=0.1,
                max_tokens=self._max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": instruction},
                            {"type": "image_url", "image_url": {"url": image_payload}},
                        ],
                    }
                ],
            )
        except Exception as exc:
            parsed = urlparse(image_uri)
            if (
                parsed.scheme in {"http", "https"}
                and image_payload == image_uri
                and self._response_adapter.is_remote_url_unsupported_error(exc)
            ):
                raise ValueError(
                    "The configured vision model does not accept remote image URLs directly and "
                    "the server could not fetch that URL into a data URI. "
                    "Use an accessible direct image URL or upload the image file."
                ) from exc
            raise
        extracted = self._response_adapter.extract_text(response)
        return extracted or self._empty_response_fallback(image_uri)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import AsyncOpenAI
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("openai package is required for OpenAI vision provider") from exc
        # Some OpenAI-compatible local servers do not enforce API keys.
        self._client = AsyncOpenAI(api_key=self._api_key or "local-placeholder-key", base_url=self._base_url)
        return self._client

    def _prepare_image_url(self, image_uri: str) -> str:
        parsed = urlparse(image_uri)
        if parsed.scheme in {"http", "https"}:
            remote_data_uri = self._fetch_remote_image_data_uri(image_uri)
            return remote_data_uri or image_uri
        if parsed.scheme == "data":
            return image_uri
        file_path = self._resolve_local_path(image_uri, parsed)
        if file_path is None:
            return image_uri
        media_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
        return f"data:{media_type};base64,{encoded}"

    def _fetch_remote_image_data_uri(self, image_uri: str) -> str | None:
        request = Request(
            image_uri,
            headers={
                "User-Agent": "MMAA-VisionClient/0.1",
                "Accept": "image/*,*/*;q=0.8",
            },
        )
        try:
            with urlopen(request, timeout=self._remote_fetch_timeout_sec) as response:
                payload = response.read(self._max_remote_image_bytes + 1)
                content_type = response.headers.get("Content-Type", "")
        except Exception:
            return None

        if not payload or len(payload) > self._max_remote_image_bytes:
            return None

        media_type = content_type.split(";", maxsplit=1)[0].strip().lower()
        if not media_type.startswith("image/"):
            guessed = mimetypes.guess_type(image_uri)[0] or ""
            media_type = guessed if guessed.startswith("image/") else ""
        if not media_type:
            return None

        encoded = base64.b64encode(payload).decode("ascii")
        return f"data:{media_type};base64,{encoded}"

    def _empty_response_fallback(self, image_uri: str) -> str:
        parsed = urlparse(image_uri)
        filename = Path(parsed.path).name or image_uri
        tags = re.findall(r"[a-z0-9]+", filename.lower())
        hint = ", ".join(tags[:5]) if tags else "unknown-subject"
        return f"Image analyzed ({hint}) but model returned an empty response."

    def _resolve_local_path(self, image_uri: str, parsed_uri) -> Path | None:
        if parsed_uri.scheme == "file":
            path_value = unquote(parsed_uri.path)
            if re.match(r"^/[A-Za-z]:/", path_value):
                path_value = path_value[1:]
            return Path(path_value)
        if re.match(r"^[a-zA-Z]:[\\/]", image_uri):
            return Path(image_uri)
        candidate = Path(image_uri)
        return candidate if candidate.exists() else None


def build_multimodal_clients(
    provider: str = "auto",
    vision_model: str = "gpt-4o-mini",
    api_key: str | None = None,
    base_url: str | None = None,
) -> MultimodalClientSelection:
    normalized_provider = provider.strip().lower()

    if normalized_provider == "openai":
        openai_available = importlib.util.find_spec("openai") is not None
        can_use_openai = (bool(api_key) or bool(base_url)) and openai_available
        if not can_use_openai:
            raise RuntimeError(
                "multimodal provider 'openai' requires MMAA_MULTIMODAL_API_KEY or MMAA_MULTIMODAL_BASE_URL and openai package"
            )
        return MultimodalClientSelection(
            vision=OpenAIVisionClient(model=vision_model, api_key=api_key, base_url=base_url),
            video=HeuristicVideoClient(),
            provider_name="openai+heuristic-video",
        )

    if normalized_provider in {"auto", "heuristic", "fallback"}:
        return MultimodalClientSelection(
            vision=HeuristicVisionClient(),
            video=HeuristicVideoClient(),
            provider_name="heuristic",
        )

    raise ValueError(f"Unsupported multimodal provider '{provider}'")
