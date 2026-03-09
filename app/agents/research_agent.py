from __future__ import annotations

import re

from app.agents.state import AgentState
from app.core.event_bus import InMemoryEventBus
from app.interfaces.retriever import Retriever
from app.tools.registry import ToolRegistry


class ResearchAgent:
    def __init__(
        self,
        retriever: Retriever,
        tool_registry: ToolRegistry,
        retrieval_top_k: int = 5,
        tool_timeout_sec: float = 2.0,
        tool_retries: int = 1,
        max_tools_per_pass: int = 2,
        retry_on_empty_retrieval: bool = True,
        retry_empty_retrieval_top_k: int = 12,
        event_bus: InMemoryEventBus | None = None,
    ) -> None:
        self._retriever = retriever
        self._tool_registry = tool_registry
        self._retrieval_top_k = retrieval_top_k
        self._tool_timeout_sec = tool_timeout_sec
        self._tool_retries = tool_retries
        self._max_tools_per_pass = max(1, int(max_tools_per_pass))
        self._retry_on_empty_retrieval = bool(retry_on_empty_retrieval)
        self._retry_empty_retrieval_top_k = max(1, int(retry_empty_retrieval_top_k))
        self._event_bus = event_bus

    async def run(self, state: AgentState) -> AgentState:
        state.record_step("research_agent")
        retrieval_top_k = self._retrieval_top_k
        raw_override = state.trace.get("retrieval_top_k")
        if raw_override:
            try:
                retrieval_top_k = max(1, int(raw_override))
            except (TypeError, ValueError):
                retrieval_top_k = self._retrieval_top_k

        try:
            state.retrieved_context = await self._retriever.retrieve(
                state.query,
                top_k=retrieval_top_k,
                metadata_filter=state.retrieval_filter or None,
            )
        except Exception as exc:
            state.errors.append(f"retrieval_error:{exc}")
            state.retrieved_context = []
        if (
            self._retry_on_empty_retrieval
            and not state.retrieved_context
            and self._query_targets_video_text(state.query)
        ):
            retry_top_k = max(retrieval_top_k, self._retry_empty_retrieval_top_k)
            retry_filter = dict(state.retrieval_filter)
            if "modality" not in retry_filter:
                retry_filter["modality"] = "video"
            try:
                state.retrieved_context = await self._retriever.retrieve(
                    state.query,
                    top_k=retry_top_k,
                    metadata_filter=retry_filter or None,
                )
            except Exception as exc:
                state.errors.append(f"retrieval_retry_error:{exc}")
                state.retrieved_context = []
            if not state.retrieved_context:
                augmented_query = (
                    f"{state.query} "
                    "Focus on spoken text, transcript, and what is said in the video."
                )
                try:
                    state.retrieved_context = await self._retriever.retrieve(
                        augmented_query,
                        top_k=retry_top_k,
                        metadata_filter=state.retrieval_filter or None,
                    )
                except Exception as exc:
                    state.errors.append(f"retrieval_retry_augmented_error:{exc}")
                    state.retrieved_context = []
            if state.retrieved_context:
                state.analysis_notes.append(
                    f"retrieval_retry_applied:video_text_intent;retrieved={len(state.retrieved_context)}"
                )

        effective_query = self._extract_effective_tool_query(state.query)
        selected_tools = self._select_tools(
            query=effective_query or state.query,
            allowed_tools=state.allowed_tools,
            retrieved_context=state.retrieved_context,
        )[: self._max_tools_per_pass]
        for tool_name in selected_tools:
            if not state.consume_tool_budget():
                break
            state.record_tool_call(tool_name)
            await self._emit_tool_event(
                state=state,
                event_type="tool.call.started",
                tool_name=tool_name,
                status_text=f"Running tool: {tool_name}",
                metadata={
                    "timeout_sec": self._tool_timeout_sec,
                    "max_retries": self._tool_retries,
                    "tool_budget_remaining": state.tool_budget,
                },
            )
            result = await self._tool_registry.run_tool(
                tool_name=tool_name,
                payload=self._build_tool_payload(
                    state=state,
                    tool_name=tool_name,
                    effective_query=effective_query,
                ),
                timeout_sec=self._tool_timeout_sec,
                max_retries=self._tool_retries,
            )
            state.tool_outputs.append(result)
            if result.get("status") == "ok":
                await self._emit_tool_event(
                    state=state,
                    event_type="tool.call.completed",
                    tool_name=tool_name,
                    status_text=f"Tool completed: {tool_name}",
                    metadata={
                        "attempts": int(result.get("attempts", 0)),
                        "latency_ms": int(result.get("latency_ms", 0)),
                    },
                )
            else:
                tool_error = str(result.get("error", "unknown"))
                state.errors.append(f"tool_error:{tool_name}:{tool_error}")
                await self._emit_tool_event(
                    state=state,
                    event_type="tool.call.failed",
                    tool_name=tool_name,
                    status_text=f"Tool failed: {tool_name}",
                    metadata={
                        "attempts": int(result.get("attempts", 0)),
                        "latency_ms": int(result.get("latency_ms", 0)),
                        "error": tool_error,
                    },
                )

        return state

    async def _emit_tool_event(
        self,
        *,
        state: AgentState,
        event_type: str,
        tool_name: str,
        status_text: str,
        metadata: dict,
    ) -> None:
        if self._event_bus is None:
            return
        run_id = state.trace.get("run_id")
        trace_id = state.trace.get("trace_id", "unknown")
        if not run_id:
            return
        await self._event_bus.emit(
            run_id=run_id,
            trace_id=trace_id,
            event_type=event_type,
            status_text=status_text,
            agent="research_agent",
            tool=tool_name,
            metadata=metadata,
        )

    def _build_tool_payload(
        self,
        *,
        state: AgentState,
        tool_name: str,
        effective_query: str | None = None,
    ) -> dict:
        compact_context: list[dict] = []
        prioritized_context = self._prioritize_tool_payload_context(state.retrieved_context, max_items=12)
        for row in prioritized_context:
            if not isinstance(row, dict):
                continue
            compact_context.append(
                {
                    "source": str(row.get("source", "")),
                    "modality": str(row.get("modality", "text")),
                    "chunk_id": row.get("chunk_id"),
                    "score": row.get("score"),
                    "timestamp_sec": row.get("timestamp_sec"),
                    "frame_index": row.get("frame_index"),
                    "snippet": str(row.get("snippet", ""))[:300],
                }
            )

        tool_query = (effective_query or self._extract_effective_tool_query(state.query) or state.query).strip()
        payload: dict = {
            "query": tool_query,
            "orchestration_query": state.query,
            "retrieved_count": len(state.retrieved_context),
            "retrieved_context": compact_context,
            "metadata_filter": dict(state.retrieval_filter),
        }

        if tool_name == "rag_debug":
            payload["top_k"] = max(1, min(4, self._retrieval_top_k))
            # Avoid expensive full source listing on every autonomous pass.
            payload["include_indexed_sources"] = False

        if tool_name == "web_search":
            payload["max_query_chars"] = 220

        if tool_name == "video_probe":
            payload["sample_fps"] = 1.0
            payload["max_frames"] = 24

        return payload

    @classmethod
    def _prioritize_tool_payload_context(cls, rows: list[dict], max_items: int) -> list[dict]:
        transcript_rows: list[dict] = []
        other_rows: list[dict] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            snippet = str(row.get("snippet", ""))
            if cls._is_transcript_like_snippet(snippet):
                transcript_rows.append(row)
            else:
                other_rows.append(row)
        return [*transcript_rows, *other_rows][: max(1, int(max_items))]

    def _select_tools(
        self,
        *,
        query: str,
        allowed_tools: list[str],
        retrieved_context: list[dict],
    ) -> list[str]:
        available = [name for name in allowed_tools if self._tool_registry.has_tool(name)]
        if not available:
            return []

        lowered = query.lower()
        query_terms = set(re.findall(r"[a-z0-9_]+", lowered))
        score_by_tool: dict[str, int] = {name: 0 for name in available}
        media_rows = [row for row in retrieved_context if isinstance(row, dict)]
        has_image_hits = any(str(row.get("modality", "")).strip().lower() == "image" for row in media_rows)
        has_video_hits = any(str(row.get("modality", "")).strip().lower() == "video" for row in media_rows)

        visual_intent = any(
            token in lowered
            for token in (
                "image",
                "photo",
                "picture",
                "visual",
                "look",
                "looks like",
                "appearance",
                "see",
            )
        )
        video_intent = any(
            token in lowered
            for token in (
                "video",
                "frame",
                "scene",
                "event",
                "happen",
                "happens",
                "clip",
                "timestamp",
            )
        )
        audio_in_video_intent = any(
            token in lowered
            for token in (
                "audio",
                "speech",
                "spoken",
                "transcript",
                "what do i say",
                "what is said",
            )
        )
        clarification_intent = any(
            token in lowered
            for token in ("clarify", "double check", "verify", "recheck", "reanalyze", "inspect")
        )

        # Direct token overlap between query and tool name.
        for name in available:
            normalized_name = name.lower()
            if any(token in normalized_name for token in query_terms):
                score_by_tool[name] += 3

        # Query-intent to tool capability mapping.
        intent_patterns = [
            (("search", "latest", "news", "lookup", "internet", "web"), ("web_search",)),
            (("http://", "https://", "url", "fetch", "page"), ("url_fetch",)),
            (("file", "read", "open", "dir", "directory", "list", "path"), ("filesystem",)),
            (("retrieval", "chunk", "citation", "debug", "grounding"), ("rag_debug",)),
            (("cpu", "memory", "latency", "metrics", "system"), ("system_metrics",)),
        ]
        for keywords, tool_fragments in intent_patterns:
            if not any(keyword in lowered for keyword in keywords):
                continue
            for name in available:
                normalized_name = name.lower()
                if any(fragment in normalized_name for fragment in tool_fragments):
                    score_by_tool[name] += 5

        # Clarification probes for media questions after retrieval.
        if has_image_hits and (visual_intent or clarification_intent):
            for name in available:
                if "vision_probe" in name.lower():
                    score_by_tool[name] += 8
        if has_video_hits and (video_intent or visual_intent or clarification_intent or audio_in_video_intent):
            for name in available:
                normalized_name = name.lower()
                if "video_probe" in normalized_name:
                    score_by_tool[name] += 9
                if audio_in_video_intent and "video_probe" in normalized_name:
                    score_by_tool[name] += 3

        prioritized = sorted(
            available,
            key=lambda name: (score_by_tool.get(name, 0), name),
            reverse=True,
        )
        positively_scored = [name for name in prioritized if score_by_tool.get(name, 0) > 0]
        if positively_scored:
            return positively_scored

        # Default-safe fallback when no intent is detected.
        if "rag_debug" in available:
            return ["rag_debug"]
        return available[:1]

    def _extract_effective_tool_query(self, query: str) -> str:
        original = query.strip()
        if not original:
            return ""

        marker = "current user request:"
        lowered = original.lower()
        marker_index = lowered.rfind(marker)
        current_request = ""
        if marker_index >= 0:
            tail = original[marker_index + len(marker) :].strip()
            tail = tail.split("\n\n", maxsplit=1)[0]
            current_request = " ".join(part.strip() for part in tail.splitlines() if part.strip())

        user_turns: list[str] = []
        for raw_line in original.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.upper().startswith("USER:"):
                _, _, content = line.partition(":")
                normalized = " ".join(content.split())
                if normalized:
                    user_turns.append(normalized)

        if current_request:
            if self._is_generic_retry_request(current_request):
                for prior in reversed(user_turns):
                    if prior.lower() == current_request.lower():
                        continue
                    if self._is_generic_retry_request(prior):
                        continue
                    return prior[:500]
            return current_request[:500]

        if user_turns:
            return user_turns[-1][:500]

        return " ".join(original.split())[:500]

    @staticmethod
    def _is_generic_retry_request(text: str) -> bool:
        normalized = " ".join(text.strip().lower().split())
        if not normalized:
            return False
        generic_tokens = {
            "again",
            "try again",
            "retry",
            "please retry",
            "do it again",
            "can you try again",
            "please try again",
            "retry that",
            "recheck",
            "check again",
            "try once more",
        }
        return normalized in generic_tokens

    @staticmethod
    def _query_targets_video_text(query: str) -> bool:
        lowered = query.lower()
        video_markers = ("video", "clip", "frame", "scene", "timestamp")
        text_markers = (
            "what is said",
            "what did he say",
            "what do i say",
            "speech",
            "spoken",
            "transcript",
            "audio",
            "text in the video",
            "text on the video",
            "check the text",
        )
        has_video = any(marker in lowered for marker in video_markers)
        has_text_or_audio = any(marker in lowered for marker in text_markers)
        return has_video and has_text_or_audio

    @staticmethod
    def _is_transcript_like_snippet(snippet: str) -> bool:
        lowered = snippet.strip().lower()
        if not lowered:
            return False
        transcript_patterns = (
            "audio event:",
            "audio transcript:",
            "spoken text:",
            "transcript:",
            "subtitle:",
            "caption:",
        )
        return any(token in lowered for token in transcript_patterns)
