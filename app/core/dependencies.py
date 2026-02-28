from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.interfaces import EmbeddingClient, LLMClient, Retriever, Tool, VideoClient, VisionClient


class StubLLMClient:
    async def generate(self, prompt: str, context: list[str] | None = None) -> str:
        return "Stubbed LLM response."


class StubEmbeddingClient:
    async def embed_text(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class StubRetriever:
    async def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        return [{"source": "stub", "snippet": "No corpus indexed yet."}]


class StubVisionClient:
    async def analyze_image(self, image_uri: str, prompt: str | None = None) -> str:
        return f"Stub vision analysis for {image_uri}"


class StubVideoClient:
    async def analyze_video(
        self,
        video_uri: str,
        prompt: str | None = None,
        sample_fps: float = 1.0,
        max_frames: int = 32,
    ) -> str:
        return f"Stub video analysis for {video_uri}"


class StubTool:
    name = "stub_tool"

    async def run(self, payload: dict) -> dict:
        return {"status": "ok", "payload": payload}


@dataclass
class ServiceContainer:
    llm: LLMClient
    embeddings: EmbeddingClient
    retriever: Retriever
    vision: VisionClient
    video: VideoClient
    tools: list[Tool]


@lru_cache
def get_container() -> ServiceContainer:
    return ServiceContainer(
        llm=StubLLMClient(),
        embeddings=StubEmbeddingClient(),
        retriever=StubRetriever(),
        vision=StubVisionClient(),
        video=StubVideoClient(),
        tools=[StubTool()],
    )

