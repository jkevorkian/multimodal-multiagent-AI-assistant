# 10 - Qwen3-VL Multimodal Embedding Stack (2026-03-07)

## 1. Why This Matters
Your target is correct: one retrieval system that can index and search text, images, and videos in a shared space, then rerank with multimodal relevance.

This is now practical with the Qwen3-VL embedding/reranker family and a multivector-capable vector database.

## 2. Release Clarification (Important)
As of 2026-03-07:
- `Qwen3-VL-Embedding` and `Qwen3-VL-Reranker` are available in `2B` and `8B` variants (not `4B` for VL).
- `Qwen3-Embedding` / `Qwen3-Reranker` (text-centric) include `4B`.

So:
- If you want unified multimodal retrieval: use `Qwen3-VL-Embedding-*` + `Qwen3-VL-Reranker-*`.
- If you want text-only fallback or hybrid text path: use `Qwen3-Embedding-*` / `Qwen3-Reranker-*`.

## 3. Perceiver Resampler in This Context
Perceiver Resampler is a token-compression bridge (popularized in Flamingo) that maps many visual tokens into a fixed latent budget before sending to the language backbone.

Practical implication for this repo:
- It is mainly a model-internal architecture decision.
- You do not implement it directly unless you train/fine-tune model internals.
- At application level, you emulate the same objective with controlled frame/pixel/token budgets and staged retrieval.

## 4. Recommended Stack for This Repo

### 4.1 Single Collection, Named Vectors
Use one Qdrant collection, with named vectors:
- `text_dense` for canonical text chunks.
- `mm_dense` for modality-native embeddings (image/video/text queries in same space).
- Optional `late_interaction` multivector field for future ColBERT-style upgrades.

Keep metadata payload unified:
- `source`, `modality`, `chunk_id`, `offset`, `timestamp_sec`, `frame_index`, `snippet`.

### 4.2 Ingestion Strategy
For each asset:
1. Canonical text representation (already present in this repo).
2. Multimodal embedding representation:
   - text chunks -> `mm_dense`
   - image findings/captions -> `mm_dense`
   - video segment evidence (timestamped events + key frames) -> `mm_dense`
3. Persist both vectors (when available) in one record payload.

### 4.3 Retrieval Strategy
1. Query embedding with `Qwen3-VL-Embedding`.
2. Dense search over `mm_dense` (and optionally fused with `text_dense`).
3. Rerank top-N with `Qwen3-VL-Reranker`.
4. Return grounded snippets with modality/timestamp evidence.

## 5. Optimization Playbook (High Impact)

### 5.1 Model Runtime
- Prefer `2B` for latency-sensitive interactive paths; use `8B` for offline indexing or high-accuracy slices.
- Use BF16 + FlashAttention2 where supported.
- Use vLLM path supported by Qwen3-VL embedding repo for higher throughput.

### 5.2 Video Ingestion Budgets
- Use adaptive sampling:
  - low motion: low FPS + sparse segments
  - high motion: dynamic FPS bursts around shot boundaries
- Enforce strict controls:
  - `max_frames`
  - `fps`
  - image/video pixel caps (`max_pixels`, `total_pixels`)
- Keep timestamped segment chunks small enough for fast reranking.

### 5.3 Embedding Dimensionality / Storage Cost
- Start with full-dimension vectors for quality baseline.
- Then test Matryoshka-style lower dimensions (if enabled by provider/config) for memory-latency tradeoff.
- Profile recall@k vs storage and query latency before reducing dimensions globally.

### 5.4 Reranker Budgeting
- Do not rerank full candidate sets.
- Use two-stage budgets (example):
  - retrieval pool: 60-120
  - rerank pool: 20-40
  - final top_k: 5-10

## 6. Proposed Implementation Sequence
1. Add multimodal embedding client abstraction and Qwen3-VL provider wiring.
2. Extend Qdrant adapter to named-vector upsert/search.
3. Update ingestion to emit multimodal vectors for image/video/text evidence units.
4. Update retriever to query `mm_dense` and run VL reranker.
5. Add evaluation slices for cross-modal retrieval and timestamp grounding.

## 7. Primary Sources
- Qwen3-VL official repo (includes `qwen-vl-embedding`): https://github.com/QwenLM/Qwen3-VL
- Qwen3-VL-Embedding repo section (models, usage, vLLM guidance): https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-embedding
- Qwen3-VL Embedding technical report (2026): https://arxiv.org/abs/2601.04720
- Qwen3 text embedding model card (includes 4B line): https://huggingface.co/Qwen/Qwen3-Embedding-4B
- Qdrant vectors docs (named vectors / multivectors): https://qdrant.tech/documentation/concepts/vectors/
- Qdrant multivector article (late interaction context): https://qdrant.tech/articles/hybrid-search/
- Flamingo paper (Perceiver Resampler concept): https://arxiv.org/abs/2204.14198
