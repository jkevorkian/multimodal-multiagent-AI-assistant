from __future__ import annotations

import asyncio
import html
import os
import platform
import re
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from app.interfaces.retriever import Retriever
from app.interfaces.tool import Tool
from app.interfaces.video import VideoClient
from app.interfaces.vector_store import VectorStore
from app.interfaces.vision import VisionClient

_URL_PATTERN = re.compile(r"https?://[^\s)>\]}]+", flags=re.IGNORECASE)
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg"}
_RETRY_QUERY_TOKENS = {
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
_DDG_RESULT_LINK_PATTERN = re.compile(
    r'(?is)<a[^>]*class="result__a"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>'
)
_DDG_RESULT_SNIPPET_PATTERN = re.compile(r'(?is)class="result__snippet"[^>]*>(?P<snippet>.*?)</(?:a|span)>')


def _compact_text(text: str) -> str:
    return " ".join(text.strip().split())


def _extract_search_query(raw_query: str, *, max_chars: int = 220) -> str:
    normalized_query = raw_query.strip()
    if not normalized_query:
        return ""

    lowered = normalized_query.lower()
    marker = "current user request:"
    marker_index = lowered.rfind(marker)
    current_request = ""
    if marker_index >= 0:
        tail = normalized_query[marker_index + len(marker) :].strip()
        tail = tail.split("\n\n", maxsplit=1)[0]
        current_request = _compact_text(" ".join(line.strip() for line in tail.splitlines() if line.strip()))

    user_turns: list[str] = []
    for raw_line in normalized_query.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.upper().startswith("USER:"):
            _, _, content = line.partition(":")
            compact = _compact_text(content)
            if compact:
                user_turns.append(compact)

    candidate = current_request
    if candidate and candidate.lower() in _RETRY_QUERY_TOKENS:
        for prior in reversed(user_turns):
            if prior.lower() == candidate.lower():
                continue
            if prior.lower() in _RETRY_QUERY_TOKENS:
                continue
            candidate = prior
            break
    elif not candidate and user_turns:
        candidate = user_turns[-1]

    if not candidate:
        candidate = _compact_text(normalized_query)

    if len(candidate) <= max_chars:
        return candidate
    if max_chars <= 3:
        return candidate[:max_chars]
    return candidate[: max_chars - 3].rstrip() + "..."


def _strip_html_tags(value: str) -> str:
    text = re.sub(r"(?is)<[^>]+>", " ", value)
    return " ".join(html.unescape(text).split())


def _normalize_duckduckgo_result_url(href: str) -> str:
    parsed = urlparse(href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        query_params = parse_qs(parsed.query)
        encoded = query_params.get("uddg", [""])[0]
        if encoded:
            return unquote(encoded)
    return href


class StubTool:
    name = "stub_tool"
    description = "Echoes agent payload for smoke testing and demos."

    async def run(self, payload: dict) -> dict:
        return {"status": "ok", "payload": payload}


class RAGDebugTool:
    name = "rag_debug"
    description = "Shows retrieval preview and indexed source coverage for the query."

    def __init__(self, retriever: Retriever, vector_store: VectorStore) -> None:
        self._retriever = retriever
        self._vector_store = vector_store

    async def run(self, payload: dict) -> dict:
        query = str(payload.get("query", "")).strip()
        if not query:
            return {"status": "error", "error": "missing_query"}

        top_k = payload.get("top_k", 5)
        try:
            top_k_value = max(1, min(12, int(top_k)))
        except (TypeError, ValueError):
            top_k_value = 5

        metadata_filter = payload.get("metadata_filter")
        if not isinstance(metadata_filter, dict):
            metadata_filter = None
        include_indexed_sources = bool(payload.get("include_indexed_sources", True))
        indexed_source_timeout_sec = payload.get("indexed_sources_timeout_sec", 0.9)
        try:
            indexed_source_timeout = max(0.2, min(float(indexed_source_timeout_sec), 3.0))
        except (TypeError, ValueError):
            indexed_source_timeout = 0.9

        retrieved = await self._retriever.retrieve(query, top_k=top_k_value, metadata_filter=metadata_filter)
        indexed: list[dict[str, Any]] = []
        indexed_sources_timed_out = False
        if include_indexed_sources:
            try:
                indexed = await asyncio.wait_for(
                    self._vector_store.list_indexed_sources(limit=15, metadata_filter=metadata_filter),
                    timeout=indexed_source_timeout,
                )
            except asyncio.TimeoutError:
                indexed_sources_timed_out = True
                indexed = []
            except Exception:
                indexed = []

        compact_hits: list[dict[str, Any]] = []
        for row in retrieved[:top_k_value]:
            if not isinstance(row, dict):
                continue
            compact_hits.append(
                {
                    "source": str(row.get("source", "unknown")),
                    "chunk_id": int(row.get("chunk_id", -1) or -1),
                    "score": float(row.get("score", 0.0) or 0.0),
                    "snippet": str(row.get("snippet", ""))[:220],
                    "modality": str(row.get("modality", "text")),
                }
            )

        modality_counts: dict[str, int] = {}
        for source_row in indexed:
            if not isinstance(source_row, dict):
                continue
            modality = str(source_row.get("modality", "text"))
            modality_counts[modality] = modality_counts.get(modality, 0) + int(source_row.get("chunk_count", 0) or 0)

        return {
            "status": "ok",
            "query": query,
            "hit_count": len(compact_hits),
            "hits": compact_hits,
            "indexed_source_count": len(indexed),
            "indexed_modalities": modality_counts,
            "indexed_sources_timed_out": indexed_sources_timed_out,
        }


class WebSearchTool:
    name = "web_search"
    description = "Runs lightweight public web search (Wikipedia API) for latest/background lookup."

    def __init__(self, timeout_sec: float = 6.0, max_results: int = 5) -> None:
        self._timeout_sec = timeout_sec
        self._max_results = max(1, min(max_results, 10))

    async def run(self, payload: dict) -> dict:
        max_query_chars_raw = payload.get("max_query_chars", 220)
        try:
            max_query_chars = max(40, min(int(max_query_chars_raw), 500))
        except (TypeError, ValueError):
            max_query_chars = 220

        raw_query = str(payload.get("query", "")).strip()
        query = _extract_search_query(raw_query, max_chars=max_query_chars)
        if not query:
            return {"status": "error", "error": "missing_query"}

        user_agent = "mmaa-assistant/0.1 (tool:web_search)"
        wiki_error = ""
        async with httpx.AsyncClient(
            timeout=self._timeout_sec,
            follow_redirects=True,
            headers={"User-Agent": user_agent},
        ) as client:
            try:
                wiki_results = await self._search_wikipedia(client=client, query=query)
                if wiki_results:
                    return {"status": "ok", "query": query, "provider": "wikipedia", "results": wiki_results}
            except Exception as exc:  # noqa: BLE001
                wiki_error = str(exc)

            try:
                ddg_results = await self._search_duckduckgo_html(client=client, query=query)
                if ddg_results:
                    return {
                        "status": "ok",
                        "query": query,
                        "provider": "duckduckgo_html",
                        "fallback_used": True,
                        "results": ddg_results,
                    }
            except Exception as exc:  # noqa: BLE001
                ddg_html_error = str(exc)
                try:
                    ddg_results = await self._search_duckduckgo(client=client, query=query)
                    if ddg_results:
                        return {
                            "status": "ok",
                            "query": query,
                            "provider": "duckduckgo_api",
                            "fallback_used": True,
                            "results": ddg_results,
                        }
                except Exception as ddg_api_exc:  # noqa: BLE001
                    ddg_error = str(ddg_api_exc)
                    details = f"wikipedia={wiki_error or 'empty'}; duckduckgo_html={ddg_html_error}; duckduckgo_api={ddg_error}"
                    raise RuntimeError(f"web_search_failed:{details}") from ddg_api_exc

        return {
            "status": "ok",
            "query": query,
            "provider": "none",
            "fallback_used": bool(wiki_error),
            "results": [],
        }

    async def _search_wikipedia(self, *, client: httpx.AsyncClient, query: str) -> list[dict[str, str]]:
        response = await client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "opensearch",
                "search": query,
                "limit": self._max_results,
                "namespace": 0,
                "format": "json",
            },
        )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, list) or len(body) < 4:
            return []
        titles = body[1] if isinstance(body[1], list) else []
        summaries = body[2] if isinstance(body[2], list) else []
        links = body[3] if isinstance(body[3], list) else []
        results: list[dict[str, str]] = []
        for title, summary, link in zip(titles, summaries, links):
            results.append({"title": str(title), "snippet": str(summary)[:260], "url": str(link)})
        return results[: self._max_results]

    async def _search_duckduckgo(self, *, client: httpx.AsyncClient, query: str) -> list[dict[str, str]]:
        response = await client.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
        )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            return []

        results: list[dict[str, str]] = []
        abstract_text = str(body.get("AbstractText", "")).strip()
        abstract_url = str(body.get("AbstractURL", "")).strip()
        if abstract_text and abstract_url:
            results.append({"title": "DuckDuckGo Abstract", "snippet": abstract_text[:260], "url": abstract_url})

        related_topics = body.get("RelatedTopics", [])
        if isinstance(related_topics, list):
            for topic in related_topics:
                if not isinstance(topic, dict):
                    continue
                text = str(topic.get("Text", "")).strip()
                url = str(topic.get("FirstURL", "")).strip()
                if not text or not url:
                    nested = topic.get("Topics")
                    if isinstance(nested, list):
                        for nested_topic in nested:
                            if not isinstance(nested_topic, dict):
                                continue
                            nested_text = str(nested_topic.get("Text", "")).strip()
                            nested_url = str(nested_topic.get("FirstURL", "")).strip()
                            if nested_text and nested_url:
                                results.append(
                                    {
                                        "title": nested_text.split(" - ", maxsplit=1)[0][:100],
                                        "snippet": nested_text[:260],
                                        "url": nested_url,
                                    }
                                )
                            if len(results) >= self._max_results:
                                break
                    if len(results) >= self._max_results:
                        break
                    continue
                results.append({"title": text.split(" - ", maxsplit=1)[0][:100], "snippet": text[:260], "url": url})
                if len(results) >= self._max_results:
                    break

        return results[: self._max_results]

    async def _search_duckduckgo_html(self, *, client: httpx.AsyncClient, query: str) -> list[dict[str, str]]:
        response = await client.get("https://html.duckduckgo.com/html/", params={"q": query})
        response.raise_for_status()
        body = response.text

        links = list(_DDG_RESULT_LINK_PATTERN.finditer(body))
        snippets = [_strip_html_tags(match.group("snippet")) for match in _DDG_RESULT_SNIPPET_PATTERN.finditer(body)]
        results: list[dict[str, str]] = []
        for index, match in enumerate(links[: self._max_results]):
            raw_title = str(match.group("title") or "").strip()
            raw_href = str(match.group("href") or "").strip()
            if not raw_title or not raw_href:
                continue
            title = _strip_html_tags(raw_title)
            url = _normalize_duckduckgo_result_url(html.unescape(raw_href))
            snippet = snippets[index] if index < len(snippets) else ""
            results.append({"title": title[:120], "snippet": snippet[:260], "url": url})
        return results


class URLFetchTool:
    name = "url_fetch"
    description = "Fetches and sanitizes a URL into compact plain text for grounding."

    def __init__(self, timeout_sec: float = 8.0, max_chars: int = 2500) -> None:
        self._timeout_sec = timeout_sec
        self._max_chars = max(500, max_chars)

    async def run(self, payload: dict) -> dict:
        candidate_url = str(payload.get("url", "")).strip()
        if not candidate_url:
            query = str(payload.get("query", ""))
            match = _URL_PATTERN.search(query)
            candidate_url = match.group(0) if match else ""
        if not candidate_url:
            return {"status": "error", "error": "no_url_found_in_payload"}

        async with httpx.AsyncClient(timeout=self._timeout_sec, follow_redirects=True) as client:
            response = await client.get(candidate_url)
            response.raise_for_status()
            content_type = str(response.headers.get("content-type", ""))
            text = response.text

        clean = re.sub(r"(?is)<(script|style)\b[^>]*>.*?</\1>", " ", text)
        clean = re.sub(r"(?s)<[^>]+>", " ", clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        if len(clean) > self._max_chars:
            clean = clean[: self._max_chars].rstrip() + "..."

        return {
            "status": "ok",
            "url": candidate_url,
            "content_type": content_type.split(";", maxsplit=1)[0].strip().lower(),
            "text": clean,
        }


class FileSystemTool:
    name = "filesystem"
    description = "Lists or reads files under the configured workspace root."

    def __init__(self, workspace_root: Path, max_file_bytes: int = 120_000) -> None:
        self._workspace_root = workspace_root.resolve()
        self._max_file_bytes = max(4_096, max_file_bytes)

    async def run(self, payload: dict) -> dict:
        action = str(payload.get("action", "")).strip().lower()
        query = str(payload.get("query", "")).lower()
        if not action:
            if any(keyword in query for keyword in ("read", "show", "open", "cat")):
                action = "read"
            else:
                action = "list"

        requested_path = str(payload.get("path", "")).strip() or "."
        safe_path = self._safe_resolve(requested_path)
        if safe_path is None:
            return {"status": "error", "error": "path_outside_workspace"}

        if action == "list":
            if not safe_path.exists():
                return {"status": "error", "error": "path_not_found"}
            target = safe_path if safe_path.is_dir() else safe_path.parent
            entries: list[dict[str, Any]] = []
            for child in sorted(target.iterdir(), key=lambda row: row.name.lower())[:80]:
                try:
                    size = child.stat().st_size if child.is_file() else 0
                except OSError:
                    size = 0
                entries.append(
                    {
                        "name": child.name,
                        "is_dir": child.is_dir(),
                        "size_bytes": int(size),
                        "path": str(child.relative_to(self._workspace_root)),
                    }
                )
            return {"status": "ok", "action": "list", "root": str(self._workspace_root), "entries": entries}

        if action == "read":
            if not safe_path.exists() or not safe_path.is_file():
                return {"status": "error", "error": "file_not_found"}
            try:
                payload_bytes = safe_path.read_bytes()
            except OSError as exc:
                return {"status": "error", "error": f"read_failed:{exc}"}
            if len(payload_bytes) > self._max_file_bytes:
                payload_bytes = payload_bytes[: self._max_file_bytes]
            try:
                text = payload_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text = payload_bytes.decode("latin-1", errors="ignore")
            return {
                "status": "ok",
                "action": "read",
                "path": str(safe_path.relative_to(self._workspace_root)),
                "text": text,
            }

        return {"status": "error", "error": f"unsupported_action:{action}"}

    def _safe_resolve(self, requested_path: str) -> Path | None:
        candidate = (self._workspace_root / requested_path).resolve()
        try:
            candidate.relative_to(self._workspace_root)
        except ValueError:
            return None
        return candidate


class SystemMetricsTool:
    name = "system_metrics"
    description = "Reports lightweight runtime/system metrics for local operations."

    def __init__(self, workspace_root: Path) -> None:
        self._workspace_root = workspace_root.resolve()

    async def run(self, payload: dict) -> dict:  # noqa: ARG002
        disk_total, disk_used, disk_free = shutil.disk_usage(self._workspace_root)
        load_avg: tuple[float, float, float] | None = None
        if hasattr(os, "getloadavg"):
            try:
                load_avg = tuple(float(item) for item in os.getloadavg())
            except OSError:
                load_avg = None
        return {
            "status": "ok",
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count() or 0,
            "load_avg": load_avg,
            "pid": os.getpid(),
            "workspace_root": str(self._workspace_root),
            "disk_total_bytes": int(disk_total),
            "disk_used_bytes": int(disk_used),
            "disk_free_bytes": int(disk_free),
        }


def _normalize_source_modality(source: str, fallback: str = "text") -> str:
    parsed = urlparse(source)
    suffix = Path(parsed.path if parsed.path else source).suffix.lower()
    if suffix in _IMAGE_EXTENSIONS:
        return "image"
    if suffix in _VIDEO_EXTENSIONS:
        return "video"
    normalized_fallback = fallback.strip().lower()
    return normalized_fallback or "text"


def _as_number(value: Any, number_type: type[float] | type[int]) -> float | int | None:
    if value is None:
        return None
    try:
        return number_type(value)
    except (TypeError, ValueError):
        return None


def _extract_media_candidates(payload: dict, modality: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    explicit_source = str(payload.get("source", "")).strip()
    if explicit_source:
        inferred_modality = _normalize_source_modality(explicit_source, fallback=modality)
        if inferred_modality == modality:
            candidates.append(
                {
                    "source": explicit_source,
                    "modality": inferred_modality,
                    "score": 10.0,
                    "chunk_id": _as_number(payload.get("chunk_id"), int),
                    "timestamp_sec": _as_number(payload.get("timestamp_sec"), float),
                    "frame_index": _as_number(payload.get("frame_index"), int),
                    "snippet": str(payload.get("snippet", ""))[:280],
                }
            )
            seen_keys.add(f"{explicit_source}::{payload.get('chunk_id')}::{payload.get('timestamp_sec')}::{payload.get('frame_index')}")

    raw_sources = payload.get("sources", [])
    if isinstance(raw_sources, list):
        for source in raw_sources:
            source_value = str(source).strip()
            if not source_value:
                continue
            inferred_modality = _normalize_source_modality(source_value, fallback="text")
            if inferred_modality != modality:
                continue
            key = f"{source_value}::list"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            candidates.append(
                {
                    "source": source_value,
                    "modality": inferred_modality,
                    "score": 5.0,
                    "chunk_id": None,
                    "timestamp_sec": None,
                    "frame_index": None,
                    "snippet": "",
                }
            )

    raw_context = payload.get("retrieved_context", [])
    if isinstance(raw_context, list):
        for item in raw_context:
            if not isinstance(item, dict):
                continue
            source_value = str(item.get("source", "")).strip()
            if not source_value:
                continue
            inferred_modality = _normalize_source_modality(source_value, fallback=str(item.get("modality", "text")))
            if inferred_modality != modality:
                continue
            chunk_id = _as_number(item.get("chunk_id"), int)
            timestamp_sec = _as_number(item.get("timestamp_sec"), float)
            frame_index = _as_number(item.get("frame_index"), int)
            key = f"{source_value}::{chunk_id}::{timestamp_sec}::{frame_index}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            score = _as_number(item.get("score"), float)
            candidates.append(
                {
                    "source": source_value,
                    "modality": inferred_modality,
                    "score": float(score) if isinstance(score, float) else 0.0,
                    "chunk_id": chunk_id,
                    "timestamp_sec": timestamp_sec,
                    "frame_index": frame_index,
                    "snippet": str(item.get("snippet", ""))[:280],
                }
            )

    candidates.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
    return candidates


class VisionProbeTool:
    name = "vision_probe"
    description = (
        "Re-analyzes a retrieved image source with a query-specific clarification prompt "
        "when grounding from ingestion snippets is insufficient."
    )

    def __init__(self, vision_client: VisionClient) -> None:
        self._vision_client = vision_client

    async def run(self, payload: dict) -> dict:
        query = str(payload.get("query", "")).strip()
        prompt = str(payload.get("prompt", "")).strip()
        candidates = _extract_media_candidates(payload, modality="image")
        if not candidates:
            return {"status": "error", "error": "no_image_source_candidate"}

        selected = candidates[0]
        source = str(selected.get("source", "")).strip()
        if not source:
            return {"status": "error", "error": "invalid_image_source"}

        effective_prompt = prompt or (
            "Clarify the visual evidence relevant to this query with concrete entities, attributes, and visible text. "
            f"Query: {query}"
        )
        summary = await self._vision_client.analyze_image(image_uri=source, prompt=effective_prompt)
        return {
            "status": "ok",
            "modality": "image",
            "source": source,
            "prompt": effective_prompt,
            "summary": summary,
            "chunk_id": selected.get("chunk_id"),
            "timestamp_sec": selected.get("timestamp_sec"),
            "frame_index": selected.get("frame_index"),
            "snippet_anchor": selected.get("snippet", ""),
        }


class VideoProbeTool:
    name = "video_probe"
    description = (
        "Re-analyzes a retrieved video source with a query-specific clarification prompt, "
        "optionally centered around retrieved timestamp/frame hints."
    )

    def __init__(
        self,
        video_client: VideoClient,
        *,
        default_sample_fps: float = 1.0,
        default_max_frames: int = 24,
    ) -> None:
        self._video_client = video_client
        self._default_sample_fps = max(0.1, float(default_sample_fps))
        self._default_max_frames = max(1, int(default_max_frames))

    async def run(self, payload: dict) -> dict:
        query = str(payload.get("query", "")).strip()
        prompt = str(payload.get("prompt", "")).strip()
        candidates = _extract_media_candidates(payload, modality="video")
        if not candidates:
            return {"status": "error", "error": "no_video_source_candidate"}

        selected = candidates[0]
        source = str(selected.get("source", "")).strip()
        if not source:
            return {"status": "error", "error": "invalid_video_source"}

        sample_fps_raw = _as_number(payload.get("sample_fps"), float)
        max_frames_raw = _as_number(payload.get("max_frames"), int)
        sample_fps = float(sample_fps_raw) if isinstance(sample_fps_raw, float) else self._default_sample_fps
        max_frames = int(max_frames_raw) if isinstance(max_frames_raw, int) else self._default_max_frames
        sample_fps = max(0.1, min(sample_fps, 10.0))
        max_frames = max(1, min(max_frames, 512))

        timestamp_sec = selected.get("timestamp_sec")
        frame_index = selected.get("frame_index")
        focus_hint = ""
        if timestamp_sec is not None:
            focus_hint = f" Prioritize evidence around timestamp {float(timestamp_sec):.1f}s."
        elif frame_index is not None:
            focus_hint = f" Prioritize evidence around frame index {int(frame_index)}."
        effective_prompt = prompt or (
            "Clarify events, actions, and spoken/visible evidence in this video that answer the user query."
            f" Query: {query}.{focus_hint}"
        )

        summary = await self._video_client.analyze_video(
            video_uri=source,
            prompt=effective_prompt,
            sample_fps=sample_fps,
            max_frames=max_frames,
        )
        return {
            "status": "ok",
            "modality": "video",
            "source": source,
            "prompt": effective_prompt,
            "summary": summary,
            "sample_fps": sample_fps,
            "max_frames": max_frames,
            "chunk_id": selected.get("chunk_id"),
            "timestamp_sec": timestamp_sec,
            "frame_index": frame_index,
            "snippet_anchor": selected.get("snippet", ""),
        }


def build_default_tools(
    *,
    retriever: Retriever,
    vector_store: VectorStore,
    workspace_root: Path,
    enable_network_tools: bool = True,
    vision_client: VisionClient | None = None,
    video_client: VideoClient | None = None,
    video_probe_sample_fps: float = 1.0,
    video_probe_max_frames: int = 24,
) -> list[Tool]:
    tools: list[Tool] = [
        StubTool(),
        RAGDebugTool(retriever=retriever, vector_store=vector_store),
        FileSystemTool(workspace_root=workspace_root),
        SystemMetricsTool(workspace_root=workspace_root),
    ]
    if vision_client is not None:
        tools.append(VisionProbeTool(vision_client=vision_client))
    if video_client is not None:
        tools.append(
            VideoProbeTool(
                video_client=video_client,
                default_sample_fps=video_probe_sample_fps,
                default_max_frames=video_probe_max_frames,
            )
        )
    if enable_network_tools:
        tools.extend([WebSearchTool(), URLFetchTool()])
    return tools
