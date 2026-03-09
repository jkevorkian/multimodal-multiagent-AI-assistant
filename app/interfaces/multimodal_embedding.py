from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from typing import Protocol


Modality = Literal["text", "image", "video"]


@dataclass(frozen=True)
class MultimodalEmbeddingInput:
    modality: Modality
    text: str
    source_uri: str | None = None


class MultimodalEmbeddingClient(Protocol):
    async def embed(self, inputs: list[MultimodalEmbeddingInput]) -> list[list[float]]:
        ...

    async def embed_query(self, text: str) -> list[float]:
        ...
