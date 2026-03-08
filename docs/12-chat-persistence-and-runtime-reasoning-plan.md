# 12 - Persistent Chat and Runtime Reasoning Plan

## 1. Scope
Define a production-style chat architecture for this project so chat is not a UI-only surface:
- multiple saved chats that can be revisited and continued,
- per-chat file/context isolation,
- multi-turn cohesive answers grounded in prior turns,
- live, scrollable runtime transcript with step history and evidence/reasoning details.

Research baseline date: 2026-03-08.

Current implementation status (2026-03-08): core M2.4 scope is implemented in the repository (durable chat sessions, chat-scoped retrieval metadata filters, transcript-enriched runtime events, and frontend persistent chat integration).

## 2. Industry Findings (Primary Sources)

### 2.1 Conversation State Is Server-Side and Durable
- OpenAI conversation state guidance describes stateful responses with explicit `conversation_id` and `previous_response_id`, plus server-side compaction controls for long threads.
- LangGraph persistence guidance centers on `thread_id` + checkpointing for durable execution, resume, and long-running workflows.
- Open WebUI chat-history docs reflect user-facing expectations: saved chats, revisit, archive/export/import.

Implication for this repo:
- We need first-class backend conversation objects, not only Streamlit `session_state`.

### 2.2 File Context Is Attached and Filtered by Scope
- OpenAI retrieval/file-search docs expose vector-store file attributes and metadata filtering patterns.
- Conversation/item APIs support attaching files as conversation content.

Implication for this repo:
- Ingest metadata must include `chat_id` and `message_id` so retrieval can be filtered per chat (plus optional global scope).

### 2.3 Modern "Thinking" UX Uses Structured Runtime Streams
- LangGraph streaming supports multiple modes (`updates`, `values`, `messages`, `custom`, `debug`) for incremental runtime UX.
- Anthropic extended-thinking docs define explicit thinking blocks and integrity constraints (thinking blocks must be preserved when returned).
- vLLM and SGLang document reasoning parsers and separate reasoning content channels for reasoning models.

Implication for this repo:
- Runtime UX should be an event timeline (scrollable), not a single mutable string.
- Reasoning display must be provider-aware:
  - show explicit reasoning stream when provider/model exposes it,
  - otherwise show structured execution traces (retrieval evidence, tool calls, step decisions) without fabricating hidden chain-of-thought.

### 2.4 Current Open-Source Thinking-Model Direction
- DeepSeek-R1 and Qwen3 reports confirm mainstream adoption of reasoning-specialized models.
- Serving stacks (vLLM/SGLang) already expose dedicated reasoning output channels that UIs can render separately from final answer text.

Implication for this repo:
- Our event contract should reserve fields for `reasoning_text` deltas and `evidence` blocks now, even if some providers do not emit both.

## 3. Gap Analysis vs Current Repo
- Current Chat tab is frontend-session based only (not durable across restarts/sessions).
- Files ingested in chat are not scoped or saved as chat-owned assets.
- Runtime status box is live but collapses history and does not render a per-step transcript with retrieved evidence payload.
- Retrieval path has no metadata-filter contract for `chat_id`.

## 4. Target Architecture

### 4.1 Persistence Model
- `chats`:
  - `chat_id`, `title`, `created_at`, `updated_at`, `archived`, defaults (`mode`, `top_k`, steering profile)
- `chat_messages`:
  - `message_id`, `chat_id`, `role`, `content`, `created_at`, `run_id`, `mode`, `metadata_json`
- `chat_files`:
  - `file_id`, `chat_id`, `message_id`, `original_name`, `uri`, `mime_type`, `size_bytes`, `sha256`, `created_at`
- `chat_runs`:
  - `run_id`, `chat_id`, `message_id`, `state`, `started_at`, `ended_at`

Storage approach:
- start with SQLite-backed store for local/dev durability,
- keep interface boundary so Postgres migration is straightforward later.

### 4.2 API Surface
- `POST /chat/sessions`
- `GET /chat/sessions`
- `GET /chat/sessions/{chat_id}`
- `PATCH /chat/sessions/{chat_id}`
- `POST /chat/sessions/{chat_id}/messages`
- `GET /chat/sessions/{chat_id}/messages`
- `POST /chat/sessions/{chat_id}/files`
- `GET /chat/sessions/{chat_id}/runs/{run_id}/events`

Compatibility:
- existing `/query`, `/agents/run`, `/runs/{run_id}/events`, `/runs/{run_id}/status` remain available.
- new chat endpoints orchestrate these routes plus persistence and scoping.

### 4.3 Retrieval Scoping
- Extend ingest metadata with:
  - `chat_id`
  - `message_id`
  - `file_id` (if uploaded)
  - `scope` (`chat` or `global`)
- Extend retriever/vector-store interfaces with optional metadata filter.
- Default chat mode queries only `scope=chat` and matching `chat_id`.

### 4.4 Runtime Transcript Contract
- Keep current status snapshot endpoint.
- Add scrollable timeline payload composed of ordered event rows:
  - stage start/completion,
  - retrieved chunks (source, chunk id, score, snippet preview),
  - tool call args/result summaries,
  - revision/guardrail events,
  - optional provider reasoning deltas when available.

Event metadata normalization:
- `event_family`: `stage | retrieval | tool | model_reasoning | revision | guardrail`
- `summary_text`: one-line human readable status
- `details`: structured JSON payload per family

## 5. Implementation Plan (Before Coding)

### Phase A - Contracts and Store
- Add `app/contracts/chat.py` schemas.
- Add `app/storage/chat_store.py` (SQLite implementation) + interface.
- Add migration/bootstrap script for local DB file.

### Phase B - Backend Routes and Services
- Add `app/api/routes/chat.py`.
- Add chat service to:
  - create/list/select chats,
  - persist turns,
  - persist chat-owned files.
- Link chat runs to existing agent runtime IDs.

### Phase C - Ingestion/Retrieval Scoping
- Add metadata-filter support across:
  - `app/interfaces/vector_store.py`
  - `app/storage/pgvector_store.py`
  - `app/storage/qdrant_store.py`
  - `app/storage/fallback_vector_store.py`
  - `app/rag/retriever.py`
  - `app/rag/ingestion.py`

### Phase D - Runtime Transcript Enrichment
- Extend runtime-event metadata for retrieval and tool/result details.
- Frontend: replace single-status-only view with:
  - compact current-status badge,
  - scrollable event transcript panel with step history and evidence blocks.

### Phase E - Frontend Chat Workspace
- Sidebar chat list: create/select/archive chats.
- Persisted message timeline per chat.
- Per-chat uploads and source panel.
- Continue conversation from saved chat state across browser/app restarts.

### Phase F - Tests and Docs
- Add:
  - `tests/test_m24_chat_sessions.py`
  - `tests/test_m24_chat_scoped_retrieval.py`
  - `tests/test_m24_runtime_transcript.py`
- Update roadmap + didactic + traceability docs after each slice.

## 5.1 Delivered Status (2026-03-08)
- Phase A implemented:
  - `app/contracts/chat.py`
  - `app/storage/chat_store.py`
- Phase B implemented:
  - `app/api/routes/chat.py`
  - chat session/message/file APIs and chat-run event stream path
- Phase C implemented:
  - metadata filters through vector-store/retriever/ingestion stack
- Phase D implemented:
  - event metadata enrichment for retrieval previews, analysis traces, and answer previews
  - frontend live status now renders cumulative transcript-style step history
- Phase E implemented:
  - Streamlit chat moved from local-only session state to backend-persisted multi-chat workflow
- Phase F implemented:
  - `tests/test_m24_chat_sessions.py`
  - `tests/test_m24_chat_scoped_retrieval.py`
  - `tests/test_m24_runtime_transcript.py`

## 6. Key Design Rules
- Do not leak data across chats by default.
- Do not invent hidden chain-of-thought; only render provider-exposed reasoning or structured execution evidence.
- Keep existing public routes backward compatible.
- Preserve compaction (M5.1) and steering (M5.2) behavior in chat routes.

## 7. Sources
- OpenAI conversation state guide: https://platform.openai.com/docs/guides/conversation-state
- OpenAI conversation state + compaction guide: https://platform.openai.com/docs/guides/conversation-state?api-mode=responses#conversation-state-guide
- OpenAI retrieval/file-search guide: https://platform.openai.com/docs/guides/retrieval
- OpenAI conversations API reference: https://platform.openai.com/docs/api-reference/conversations
- LangGraph persistence: https://docs.langchain.com/oss/python/langgraph/persistence
- LangGraph streaming: https://docs.langchain.com/oss/python/langgraph/streaming
- Anthropic extended thinking: https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking
- Anthropic context windows + extended thinking: https://docs.anthropic.com/en/docs/build-with-claude/context-windows
- vLLM reasoning outputs: https://docs.vllm.ai/en/latest/features/reasoning_outputs.html
- SGLang reasoning parser: https://docs.sglang.ai/advanced_features/separate_reasoning.html
- Open WebUI chat history: https://docs.openwebui.com/features/chat-history/
- DeepSeek-R1 paper: https://arxiv.org/abs/2501.12948
- Qwen3 technical report: https://arxiv.org/abs/2505.09388
