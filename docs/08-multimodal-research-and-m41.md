# 08 - Multimodal Research + M4.1 Integration Notes

## 1. Scope and Date
- Date: 2026-03-06
- Goal: research recent multimodal papers + current industry practices, then translate findings into concrete architecture and implementation choices for this project.

## 2. Research Snapshot (Papers)

### 2.1 Multimodal RAG and Video-RAG
- **VideoRAG (2025)** highlights that long-video understanding benefits from query-aware retrieval over multimodal evidence, not only full-context summarization.
  - https://arxiv.org/abs/2502.01549
- **VideoRAG (ACL Findings 2025)** shows gains by treating long videos as retrievable multimodal corpora (visual/audio/subtitle metadata).
  - https://aclanthology.org/2025.findings-acl.1193/
- **mRAG (2025)** emphasizes modality-specific indexing/retrieval decisions and warns against naive "single embedding only" pipelines.
  - https://arxiv.org/abs/2505.24073
- **RAG-Anything (2025)** frames multimodal content as dual graphs (cross-modal + textual) to improve retrieval over mixed document structures.
  - https://arxiv.org/abs/2510.12323
- **REAL-MM-RAG benchmark (ACL Findings 2025)** reports large performance gaps on real-world multimodal retrieval settings and reinforces the need for stronger retrieval components.
  - https://aclanthology.org/2025.findings-acl.1115/
- **ViDoRAG (EMNLP Findings 2025)** supports multi-granularity indexing for visual documents and the value of retrieval-reranking stacks.
  - https://aclanthology.org/2025.findings-emnlp.1331/
- **CLaMR (2025)** reports strong results from late-interaction multimodal retrieval over frame/ocr/audio/text signals.
  - https://arxiv.org/abs/2506.06144

### 2.2 Practical Conclusions from Papers
- Use **multiple representations per asset** (global summary + segment/frame evidence), not just one caption.
- Use **modality-aware metadata** and **time-aware evidence tags** for videos.
- Keep **hybrid retrieval + reranking** available for difficult queries.
- Prefer **adaptive retrieval depth** by query complexity/cost budget.

## 3. Industry Practices and Recent Updates

### 3.1 Model/API Capabilities (official docs)
- OpenAI API guidance: image-capable multimodal prompting in the Responses/Chat APIs.
  - https://platform.openai.com/docs/guides/images-vision
- OpenAI model update context (GPT-4.1 family, strong long-context + multimodal evals).
  - https://openai.com/index/gpt-4-1/
- Google Gemini official video understanding flow includes native video file analysis and timestamped processing behavior.
  - https://ai.google.dev/gemini-api/docs/video-understanding
- Google Gemini API updates include broader I/O and video-centric improvements.
  - https://developers.googleblog.com/en/new-features-for-the-gemini-api-and-google-ai-studio/
- Anthropic vision docs confirm strong image support but also important limitations for spatial/detail-heavy tasks.
  - https://docs.anthropic.com/en/docs/build-with-claude/vision

### 3.2 Vector/Index Infrastructure Patterns
- Qdrant docs document **multivectors** and late-interaction support patterns relevant to multimodal retrieval.
  - https://qdrant.tech/documentation/concepts/vectors/
- Vertex AI multimodal embedding docs show shared semantic space patterns for cross-modal retrieval (image/text embeddings).
  - https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-multimodal-embeddings
- Weaviate multi-vector search patterns reinforce multi-representation query strategies.
  - https://weaviate.io/developers/weaviate/search/multi-vector

## 4. Recommended "Single RAG Structure for Any Modality"

Use one shared retrieval corpus with modality-aware metadata, plus representation layers:

1. **Canonical text layer (required)**
   - Convert every source into retrieval text units:
     - text/doc: extracted chunks
     - image: caption/findings + optional OCR text
     - video: global summary + timestamped key events + per-frame findings

2. **Evidence layer (recommended)**
   - Keep explicit references (`source`, `modality`, `timestamp_sec`, `frame_index`, `snippet`) in metadata so answers can cite grounded evidence.

3. **Optional dense multimodal layer (future M5/M6)**
   - Add modality-specific vectors (or multivectors) for harder cross-modal recall while preserving canonical text retrieval as baseline.

4. **Retrieval policy**
   - Start with hybrid text retrieval (already in project).
   - Route hard multimodal queries to deeper retrieval/rerank path using budgets.

## 5. M4.1 Implementation in This Repo

Implemented now:
- Real local/remote video frame extraction path in `VideoFrameSampler` (strict decode via `cv2`); no heuristic frame fallback.
- `SampledFrame` now carries `image_uri` payloads for per-frame VLM analysis.
- `VideoAnalysisAdapter` now supports per-frame vision analysis and passes those findings to temporal aggregation.
- `TemporalAggregator` now prioritizes frame findings when available and derives top-level summary from those findings.
- `DocumentIngestionService` now uses the same video adapter for ingestion, storing timeline-aware video representations in the shared RAG pipeline.

Contract preserved:
- `/video/analyze` response shape unchanged: `summary`, `key_events`, `confidence`, `processed_frames`, `trace`.

## 6. Next Integration Steps (Recommended)
1. Add OCR extraction for sampled frames (for text-heavy videos/slides).
2. Store per-event metadata objects (timestamp/frame id) in chunk metadata for finer citation UX.
3. Add modality-aware reranking policy for difficult cross-modal queries.
4. Add evaluation slices specific to video grounding (timestamp citation precision + event recall).

## 7. Context Compaction Note (Codex-style pattern)

Based on OpenAI Codex public docs/issues:
- Codex supports conversation-state compaction and auto-compaction in long sessions.
- Reported behavior includes compaction around high token thresholds (community report around ~100K tokens).
- The pattern is: summarize/prune history while preserving important context, then continue from compacted state.
  - References:
    - https://openai.com/index/introducing-codex/
    - https://developers.openai.com/codex/ide
    - https://github.com/openai/codex/issues/1257

Recommended adaptation for this project:
1. Trigger by token budget threshold at orchestrator/query entry.
2. Keep pinned blocks (goals, constraints, citations, tool outputs, open tasks).
3. Replace old turns with compacted checkpoint summary + pointer metadata.
4. Track compaction events in traces for auditability.
