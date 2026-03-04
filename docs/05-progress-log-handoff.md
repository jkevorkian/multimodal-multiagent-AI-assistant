# 05 - Progress Log Handoff

## 1. Snapshot
- Date: 2026-03-03
- Branch head (before commit): `893364b`
- Working tree state: multiple unstaged changes, plus new files (see section 7)
- Last test run: `.\venv\Scripts\python -m pytest`
- Test result: `19 passed, 8 warnings` (FastAPI `on_event` deprecation warnings only)

## 2. Scope Completed So Far
Implementation has been advanced through **M1** only.

- M0 scaffold remains in place.
- M1 was upgraded to a modernized P1 baseline:
  - provider-pluggable embeddings
  - hybrid retrieval (dense + lexical) with reciprocal-rank fusion (RRF)
  - reranker hook layer
  - external vector DB adapter (Qdrant) with fallback coexistence
  - batched embedding in ingestion
  - updated tests and docs
- M2+ runtime code has **not** been implemented yet (only planning/docs updates).

## 3. Major Technical Changes (M1)

### 3.1 Embeddings
- Replaced hash-only embedding path with provider factory:
  - deterministic fallback provider
  - OpenAI provider (optional dependency)
  - SentenceTransformers provider (optional dependency)
- Added batched embedding interface support (`embed_texts`).
- Files:
  - `app/rag/embeddings.py`
  - `app/interfaces/embedding.py`
  - `app/core/dependencies.py`
  - `app/core/config.py`

### 3.2 Retrieval
- Upgraded retriever to:
  - dense retrieval via vector search
  - lexical retrieval via keyword search
  - fusion via RRF
  - reranker hook on fused candidate pool
- Files:
  - `app/rag/retriever.py`
  - `app/storage/pgvector_store.py`
  - `app/storage/qdrant_store.py`
  - `app/storage/fallback_vector_store.py`
  - `app/interfaces/vector_store.py`

### 3.3 Reranking
- Added reranker abstraction and implementations:
  - `NoOpReranker`
  - `LexicalOverlapReranker`
  - optional `CrossEncoderReranker`
- Files:
  - `app/interfaces/reranker.py`
  - `app/rag/reranker.py`
  - `app/interfaces/__init__.py`
  - `app/rag/__init__.py`

### 3.4 Ingestion / API Wiring
- Ingestion now uses batched embeddings per chunk batch.
- DI container now exposes selected embedding provider and reranker provider.
- Existing M1 route behavior remains stable (`/ingest/documents`, `/query`) while using upgraded backend pipeline.
- Files:
  - `app/rag/ingestion.py`
  - `app/core/dependencies.py`
  - `app/api/routes/ingest.py`
  - `app/api/routes/query.py`

### 3.5 Tests
- Existing contract tests kept passing.
- Added M1 tests for:
  - embedding provider fallback behavior
  - hybrid retrieval branch usage
  - reranker hook execution
- Files:
  - `tests/test_m1_rag.py`
  - `tests/test_api_contracts.py`

## 4. Documentation Updated
- Requirements updated with modernized M1 requirements:
  - `docs/01-requirements.md`
- Roadmap updated:
  - M1 modernization details
  - M2+ improvement points (durability/MCP/eval upgrades)
  - `docs/02-implementation-roadmap.md`
- Didactic traceability updated:
  - decision log + rationale for P1 modernization
  - `docs/03-didactic-traceability.md`
- File traceability extracted and maintained separately:
  - `docs/04-file-traceability-by-milestone.md`

## 5. Runtime Configuration Added (M1)
New/extended settings in `app/core/config.py`:

- Vector store:
  - `MMAA_RAG_VECTOR_STORE_PROVIDER`
  - `MMAA_RAG_VECTOR_STORE_MIRROR_WRITES`
  - `MMAA_QDRANT_URL`
  - `MMAA_QDRANT_API_KEY`
  - `MMAA_QDRANT_COLLECTION_NAME`
  - `MMAA_QDRANT_PREFER_GRPC`
  - `MMAA_QDRANT_TIMEOUT_SEC`
- Embeddings:
  - `MMAA_RAG_EMBEDDING_PROVIDER` (`deterministic` default)
  - `MMAA_RAG_EMBEDDING_MODEL`
  - `MMAA_RAG_SENTENCE_TRANSFORMER_MODEL`
  - `MMAA_RAG_OPENAI_API_KEY`
  - `MMAA_RAG_EMBEDDING_DIMENSIONS`
  - `MMAA_RAG_REQUESTED_EMBEDDING_DIMENSIONS`
- Retrieval/rerank:
  - `MMAA_RAG_DENSE_TOP_K`
  - `MMAA_RAG_LEXICAL_TOP_K`
  - `MMAA_RAG_RRF_K`
  - `MMAA_RAG_RERANK_POOL_SIZE`
  - `MMAA_RAG_RERANKER_ENABLED`
  - `MMAA_RAG_RERANKER_PROVIDER` (`lexical` default)
  - `MMAA_RAG_RERANKER_MODEL`

## 6. Operational Notes for Takeover
- Defaults are set to deterministic embeddings + lexical reranker for stable local runs.
- Neural providers are optional and dependency-gated:
  - `openai` package required for OpenAI embeddings
  - `sentence-transformers` package required for local neural embeddings/cross-encoder reranking
- `requirements.txt` currently includes `openai`.
- PGVector path remains available when `MMAA_DATABASE_URL` and `psycopg` are available; otherwise falls back to in-memory store.

## 7. Current File Change Inventory
Modified:
- `app/api/routes/ingest.py`
- `app/api/routes/query.py`
- `app/contracts/schemas.py`
- `app/core/config.py`
- `app/core/dependencies.py`
- `app/interfaces/__init__.py`
- `app/interfaces/embedding.py`
- `app/interfaces/vector_store.py`
- `docs/01-requirements.md`
- `docs/02-implementation-roadmap.md`
- `docs/03-didactic-traceability.md`
- `requirements.txt`
- `tests/test_api_contracts.py`

New:
- `app/interfaces/reranker.py`
- `app/rag/__init__.py`
- `app/rag/chunking.py`
- `app/rag/embeddings.py`
- `app/rag/ingestion.py`
- `app/rag/reranker.py`
- `app/rag/retriever.py`
- `app/storage/__init__.py`
- `app/storage/pgvector_store.py`
- `app/storage/qdrant_store.py`
- `app/storage/fallback_vector_store.py`
- `docs/04-file-traceability-by-milestone.md`
- `tests/test_m1_rag.py`
- `tests/test_vector_store_fallback.py`

This file:
- `docs/05-progress-log-handoff.md`

## 8. Recommended Next Steps (For Next Agent)
1. Review and commit M1 changes as a single milestone commit.
2. Decide whether to enable neural embedding provider in current environment (`openai` or `sentence-transformers`) and add integration tests for that provider path.
3. Keep M2 code untouched until review is complete; only proceed after explicit approval.
