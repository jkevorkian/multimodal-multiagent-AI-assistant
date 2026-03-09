from __future__ import annotations

from dataclasses import dataclass

from app.interfaces.video import VideoClient
from app.interfaces.vision import VisionClient
from app.video.frame_sampler import FrameSamplingResult, VideoFrameSampler
from app.video.temporal_aggregator import TemporalAggregator


@dataclass(frozen=True)
class VideoAnalysisResult:
    summary: str
    key_events: list[str]
    confidence: float
    processed_frames: int


class VideoAnalysisAdapter:
    def __init__(
        self,
        video_client: VideoClient,
        frame_sampler: VideoFrameSampler,
        temporal_aggregator: TemporalAggregator,
        vision_client: VisionClient | None = None,
        require_frame_level_findings: bool = False,
        frame_prompt_template: str = (
            "Describe this video frame at t={timestamp_sec:.1f}s with concrete entities, actions, "
            "and scene changes relevant to a timeline summary. Respond in English unless the user explicitly asks for another language."
        ),
    ) -> None:
        self._video_client = video_client
        self._frame_sampler = frame_sampler
        self._temporal_aggregator = temporal_aggregator
        self._vision_client = vision_client
        self._require_frame_level_findings = require_frame_level_findings
        self._frame_prompt_template = frame_prompt_template

    async def analyze(
        self,
        *,
        video_uri: str,
        prompt: str | None,
        sample_fps: float,
        max_frames: int,
    ) -> VideoAnalysisResult:
        sampling = self._frame_sampler.sample(
            video_uri=video_uri,
            sample_fps=sample_fps,
            max_frames=max_frames,
        )
        frame_findings = await self._analyze_frames(
            prompt=prompt,
            sampling=sampling,
        )
        if self._require_frame_level_findings and not frame_findings:
            raise ValueError("frame_level_analysis_unavailable")
        raw_summary = await self._video_client.analyze_video(
            video_uri=video_uri,
            prompt=prompt,
            sample_fps=sample_fps,
            max_frames=max_frames,
        )
        aggregated = self._temporal_aggregator.aggregate(
            video_uri=video_uri,
            prompt=prompt,
            raw_summary=raw_summary,
            sampling=sampling,
            frame_findings=frame_findings,
        )
        return VideoAnalysisResult(
            summary=aggregated.summary,
            key_events=aggregated.key_events,
            confidence=aggregated.confidence,
            processed_frames=aggregated.processed_frames,
        )

    async def _analyze_frames(
        self,
        *,
        prompt: str | None,
        sampling: FrameSamplingResult,
    ) -> list[str]:
        if self._vision_client is None:
            return []

        findings: list[str] = []
        for frame in sampling.frames:
            if not frame.image_uri:
                continue
            frame_prompt = self._frame_prompt_template.format(timestamp_sec=frame.timestamp_sec)
            if prompt:
                frame_prompt = f"{frame_prompt} User focus: {prompt}"
            try:
                response = await self._vision_client.analyze_image(image_uri=frame.image_uri, prompt=frame_prompt)
            except Exception:
                continue
            normalized = " ".join((response or "").split()).strip()
            if normalized:
                findings.append(normalized)
        return findings
