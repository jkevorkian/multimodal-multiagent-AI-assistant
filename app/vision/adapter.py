from __future__ import annotations

from dataclasses import dataclass

from app.interfaces.vision import VisionClient
from app.vision.preprocess import ProcessedImage, VisionPreprocessor


@dataclass(frozen=True)
class VisionAnalysis:
    processed_image: ProcessedImage
    prompt: str | None
    summary: str


class VisionAdapter:
    def __init__(self, vision_client: VisionClient, preprocessor: VisionPreprocessor) -> None:
        self._vision_client = vision_client
        self._preprocessor = preprocessor

    async def analyze(self, image_uri: str, prompt: str | None = None) -> VisionAnalysis:
        processed = await self._preprocessor.preprocess(image_uri)
        summary = await self._vision_client.analyze_image(
            image_uri=processed.inference_uri,
            prompt=prompt,
        )
        return VisionAnalysis(processed_image=processed, prompt=prompt, summary=summary)
