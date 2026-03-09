from __future__ import annotations

import re
from typing import Any

from app.agents.state import AgentState
from app.core.event_bus import InMemoryEventBus
from app.interfaces.llm import LLMClient


class AnswerAgent:
    _MAX_RETRIEVAL_CONTEXT = 10
    _MAX_TOOL_CONTEXT = 10
    _MAX_ANALYSIS_CONTEXT = 6

    def __init__(self, llm: LLMClient, event_bus: InMemoryEventBus | None = None) -> None:
        self._llm = llm
        self._event_bus = event_bus

    async def run(self, state: AgentState) -> AgentState:
        state.record_step("answer_agent")
        context_parts = self._build_context_parts(state)

        try:
            await self._emit_model_call_event(state, context_parts=context_parts)
            state.final_answer = await self._llm.generate(state.query, context_parts)
        except Exception as exc:
            state.errors.append(f"answer_error:{exc}")
            state.final_answer = "I could not complete answer synthesis due to an internal error."

        if not state.final_answer.strip():
            state.final_answer = "I do not have enough evidence yet to answer confidently."
            state.confidence = min(state.confidence, 0.2)
        state.clamp_confidence()
        return state

    def _build_context_parts(self, state: AgentState) -> list[str]:
        retrieval_context = self._format_retrieval_context(state.retrieved_context)
        tool_context = self._format_tool_context(state.tool_outputs)
        analysis_context = self._format_analysis_notes(state.analysis_notes)
        combined = [*retrieval_context, *tool_context, *analysis_context]
        return self._dedupe_non_empty(combined)

    def _format_retrieval_context(self, retrieved_context: list[dict[str, Any]]) -> list[str]:
        transcript_rows: list[str] = []
        other_rows: list[str] = []
        for item in retrieved_context:
            if not isinstance(item, dict):
                continue
            snippet = str(item.get("snippet", "")).strip()
            if not snippet:
                continue
            source = str(item.get("source", "unknown"))
            chunk_id = item.get("chunk_id", -1)
            modality = str(item.get("modality", "text")).strip().lower() or "text"
            prefix = f"[retrieval:{modality}][source={source}#chunk-{chunk_id}] "
            row = prefix + self._compact_text(snippet, limit=360)
            if self._is_transcript_like_snippet(snippet):
                transcript_rows.append(row)
            else:
                other_rows.append(row)

        ordered = [*transcript_rows, *other_rows]
        return ordered[: self._MAX_RETRIEVAL_CONTEXT]

    def _format_tool_context(self, tool_outputs: list[dict[str, Any]]) -> list[str]:
        rows: list[str] = []
        for item in tool_outputs:
            if not isinstance(item, dict):
                continue
            tool_name = str(item.get("tool", "unknown"))
            status = str(item.get("status", "unknown")).strip().lower()
            result = item.get("result")
            if status != "ok":
                error = self._compact_text(str(item.get("error", "unknown")), limit=220)
                rows.append(f"[tool:{tool_name}][status=error] {error}")
                continue
            if not isinstance(result, dict):
                rows.append(f"[tool:{tool_name}][status=ok] result_available")
                continue
            rows.extend(self._format_single_tool_result(tool_name=tool_name, result=result))
            if len(rows) >= self._MAX_TOOL_CONTEXT:
                break
        return rows[: self._MAX_TOOL_CONTEXT]

    def _format_single_tool_result(self, *, tool_name: str, result: dict[str, Any]) -> list[str]:
        if tool_name == "web_search":
            provider = str(result.get("provider", "unknown"))
            query = self._compact_text(str(result.get("query", "")), limit=180)
            rows = [f"[tool:web_search] provider={provider}; query={query}"]
            entries = result.get("results", [])
            if isinstance(entries, list):
                for entry in entries[:3]:
                    if not isinstance(entry, dict):
                        continue
                    title = self._compact_text(str(entry.get("title", "")), limit=110)
                    snippet = self._compact_text(str(entry.get("snippet", "")), limit=210)
                    url = self._compact_text(str(entry.get("url", "")), limit=180)
                    rows.append(f"[tool:web_search][result] {title} | {snippet} | {url}")
            return rows

        if tool_name == "video_probe":
            rows = []
            summary = self._compact_text(str(result.get("summary", "")), limit=320)
            if summary:
                rows.append(f"[tool:video_probe][summary] {summary}")
            text_evidence = result.get("text_evidence", [])
            if isinstance(text_evidence, list):
                for entry in text_evidence[:3]:
                    snippet = self._compact_text(str(entry), limit=220)
                    if snippet:
                        rows.append(f"[tool:video_probe][text] {snippet}")
            snippet_anchor = self._compact_text(str(result.get("snippet_anchor", "")), limit=220)
            if snippet_anchor:
                rows.append(f"[tool:video_probe][anchor] {snippet_anchor}")
            return rows

        if tool_name == "rag_debug":
            hit_count = result.get("hit_count")
            rows = [f"[tool:rag_debug] hit_count={hit_count}"]
            hits = result.get("hits", [])
            if isinstance(hits, list):
                for hit in hits[:2]:
                    if not isinstance(hit, dict):
                        continue
                    snippet = self._compact_text(str(hit.get("snippet", "")), limit=210)
                    source = self._compact_text(str(hit.get("source", "")), limit=140)
                    rows.append(f"[tool:rag_debug][hit] {source} | {snippet}")
            return rows

        compact = self._compact_text(str(result), limit=260)
        return [f"[tool:{tool_name}] {compact}"] if compact else [f"[tool:{tool_name}] status=ok"]

    def _format_analysis_notes(self, analysis_notes: list[str]) -> list[str]:
        return [f"[analysis] {self._compact_text(note, limit=260)}" for note in analysis_notes[: self._MAX_ANALYSIS_CONTEXT]]

    async def _emit_model_call_event(self, state: AgentState, *, context_parts: list[str]) -> None:
        if self._event_bus is None:
            return
        run_id = state.trace.get("run_id")
        trace_id = state.trace.get("trace_id", "unknown")
        if not run_id:
            return
        context_preview = [segment[:180] for segment in context_parts[:4]]
        await self._event_bus.emit(
            run_id=run_id,
            trace_id=trace_id,
            event_type="model.call.in_progress",
            status_text="Synthesizing final answer...",
            agent="answer_agent",
            metadata={
                "model": self._llm.__class__.__name__,
                "context_preview": context_preview,
                "analysis_notes": state.analysis_notes[-3:],
            },
        )

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
        if any(token in lowered for token in transcript_patterns):
            return True
        return bool(re.search(r"\b(he says|she says|they say|says)\b", lowered))

    @staticmethod
    def _compact_text(value: str, *, limit: int) -> str:
        compact = " ".join(value.split())
        if len(compact) <= limit:
            return compact
        if limit <= 3:
            return compact[:limit]
        return compact[: limit - 3].rstrip() + "..."

    @staticmethod
    def _dedupe_non_empty(rows: list[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for row in rows:
            normalized = row.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique.append(normalized)
        return unique
