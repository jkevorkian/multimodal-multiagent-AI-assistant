from __future__ import annotations

from typing import Protocol


class VideoClient(Protocol):
    async def analyze_video(
        self,
        video_uri: str,
        prompt: str | None = None,
        sample_fps: float = 1.0,
        max_frames: int = 32,
    ) -> str:
        ...

