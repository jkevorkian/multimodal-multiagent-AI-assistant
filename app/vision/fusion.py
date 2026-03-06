from __future__ import annotations

import re

from app.vision.adapter import VisionAnalysis


class VisionFusion:
    def compose(self, analysis: VisionAnalysis) -> tuple[str, list[str], float]:
        summary = analysis.summary.strip()
        evidence_tag = self._build_evidence_tag(analysis)
        clauses = [part.strip(" -") for part in re.split(r"[;\n\.]+", summary) if part.strip()]
        if not clauses:
            clauses = ["No visual findings extracted."]
        findings = [f"{clause} {evidence_tag}".strip() for clause in clauses[:5]]
        confidence = self._confidence(analysis=analysis, findings=findings)
        return summary or "No summary available.", findings, confidence

    def _build_evidence_tag(self, analysis: VisionAnalysis) -> str:
        image = analysis.processed_image
        size_token = (
            f"{image.width}x{image.height}"
            if image.width is not None and image.height is not None
            else "unknown-size"
        )
        hash_token = image.content_hash[:10] if image.content_hash else "n/a"
        return (
            "[evidence:"
            f"source={image.image_uri};"
            f"mime={image.mime_type};"
            f"size={size_token};"
            f"sha={hash_token}"
            "]"
        )

    def _confidence(self, analysis: VisionAnalysis, findings: list[str]) -> float:
        score = 0.45
        if analysis.processed_image.width and analysis.processed_image.height:
            score += 0.15
        if analysis.processed_image.content_hash:
            score += 0.1
        if len(findings) >= 2:
            score += 0.1
        if analysis.prompt:
            score += 0.05
        return round(min(0.95, max(0.1, score)), 2)
