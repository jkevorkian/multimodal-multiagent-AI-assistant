from __future__ import annotations

import re
from dataclasses import dataclass

from app.video.frame_sampler import FrameSamplingResult


@dataclass(frozen=True)
class TemporalAggregation:
    summary: str
    key_events: list[str]
    confidence: float
    processed_frames: int


class TemporalAggregator:
    """Compose timeline-aware events from model summary + sampled frame schedule."""

    def __init__(self, max_key_events: int = 5) -> None:
        if max_key_events <= 0:
            raise ValueError("max_key_events must be > 0")
        self._max_key_events = max_key_events

    def aggregate(
        self,
        *,
        video_uri: str,
        prompt: str | None,
        raw_summary: str,
        sampling: FrameSamplingResult,
        frame_findings: list[str] | None = None,
    ) -> TemporalAggregation:
        normalized_summary = " ".join((raw_summary or "").split()).strip()
        clauses = [item.strip() for item in (frame_findings or []) if item and item.strip()]
        if not clauses:
            clauses = [part.strip(" -") for part in re.split(r"[;\n\.]+", normalized_summary) if part.strip()]
        if not clauses:
            clauses = ["No temporal findings extracted from video content."]

        frames = sampling.frames or []
        key_events: list[str] = []
        for index, clause in enumerate(clauses[: self._max_key_events]):
            if frames:
                frame = frames[min(index, len(frames) - 1)]
                key_events.append(f"[t={frame.timestamp_sec:.1f}s][source={video_uri}] {clause}")
            else:
                key_events.append(f"[source={video_uri}] {clause}")

        confidence = self._confidence(
            prompt=prompt,
            sampling=sampling,
            key_events=key_events,
            frame_findings=clauses if frame_findings else [],
        )
        if frame_findings:
            summary = " ".join(clauses[: min(3, len(clauses))]).strip()
        else:
            summary = normalized_summary or key_events[0]
        return TemporalAggregation(
            summary=summary,
            key_events=key_events,
            confidence=confidence,
            processed_frames=sampling.processed_frames,
        )

    def _confidence(
        self,
        *,
        prompt: str | None,
        sampling: FrameSamplingResult,
        key_events: list[str],
        frame_findings: list[str],
    ) -> float:
        score = 0.35
        score += min(0.35, 0.03 * sampling.processed_frames)
        if sampling.budget_limited:
            score -= 0.05
        if prompt:
            score += 0.05
        if len(key_events) >= 2:
            score += 0.08
        if sampling.extraction_mode == "decoded":
            score += 0.07
        if frame_findings:
            score += min(0.08, 0.02 * len(frame_findings))
        return round(min(0.95, max(0.1, score)), 2)
