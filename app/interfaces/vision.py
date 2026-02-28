from __future__ import annotations

from typing import Protocol


class VisionClient(Protocol):
    async def analyze_image(self, image_uri: str, prompt: str | None = None) -> str:
        ...

