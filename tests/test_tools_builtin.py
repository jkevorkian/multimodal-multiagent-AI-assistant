from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from app.tools.builtin import (
    FileSystemTool,
    RAGDebugTool,
    SystemMetricsTool,
    URLFetchTool,
    VideoProbeTool,
    VisionProbeTool,
    WebSearchTool,
    _extract_search_query,
    build_default_tools,
)


class _RetrieverStub:
    async def retrieve(self, query: str, top_k: int = 5, metadata_filter: dict | None = None) -> list[dict]:  # noqa: ARG002
        return [
            {
                "source": "doc://alpha",
                "chunk_id": 0,
                "score": 0.91,
                "snippet": f"retrieved::{query}",
                "modality": "text",
            }
        ][:top_k]


class _VectorStoreStub:
    async def list_indexed_sources(self, limit: int = 200, metadata_filter: dict | None = None) -> list[dict]:  # noqa: ARG002
        return [{"source": "doc://alpha", "modality": "text", "chunk_count": 1}][:limit]


class _SlowVectorStoreStub:
    async def list_indexed_sources(self, limit: int = 200, metadata_filter: dict | None = None) -> list[dict]:  # noqa: ARG002
        await asyncio.sleep(2.0)
        return [{"source": "doc://slow", "modality": "text", "chunk_count": 2}][:limit]


class _VisionClientStub:
    async def analyze_image(self, image_uri: str, prompt: str | None = None) -> str:
        return f"vision::{image_uri}::{prompt or ''}"


class _VideoClientStub:
    async def analyze_video(
        self,
        video_uri: str,
        prompt: str | None = None,
        sample_fps: float = 1.0,
        max_frames: int = 32,
    ) -> str:
        return f"video::{video_uri}::{prompt or ''}::{sample_fps:.1f}::{max_frames}"


def test_build_default_tools_includes_expected_names(tmp_path: Path) -> None:
    names_with_network = [tool.name for tool in build_default_tools(
        retriever=_RetrieverStub(),  # type: ignore[arg-type]
        vector_store=_VectorStoreStub(),  # type: ignore[arg-type]
        workspace_root=tmp_path,
        enable_network_tools=True,
    )]
    assert {"stub_tool", "rag_debug", "filesystem", "system_metrics", "web_search", "url_fetch"}.issubset(
        set(names_with_network)
    )

    names_without_network = [tool.name for tool in build_default_tools(
        retriever=_RetrieverStub(),  # type: ignore[arg-type]
        vector_store=_VectorStoreStub(),  # type: ignore[arg-type]
        workspace_root=tmp_path,
        enable_network_tools=False,
    )]
    assert "web_search" not in names_without_network
    assert "url_fetch" not in names_without_network


def test_build_default_tools_includes_media_probe_tools_when_clients_available(tmp_path: Path) -> None:
    names = [
        tool.name
        for tool in build_default_tools(
            retriever=_RetrieverStub(),  # type: ignore[arg-type]
            vector_store=_VectorStoreStub(),  # type: ignore[arg-type]
            workspace_root=tmp_path,
            enable_network_tools=False,
            vision_client=_VisionClientStub(),  # type: ignore[arg-type]
            video_client=_VideoClientStub(),  # type: ignore[arg-type]
            video_probe_sample_fps=1.5,
            video_probe_max_frames=42,
        )
    ]
    assert "vision_probe" in names
    assert "video_probe" in names


def test_rag_debug_tool_returns_retrieval_preview() -> None:
    tool = RAGDebugTool(retriever=_RetrieverStub(), vector_store=_VectorStoreStub())  # type: ignore[arg-type]
    payload = asyncio.run(tool.run({"query": "policy controls"}))
    assert payload["status"] == "ok"
    assert payload["hit_count"] == 1
    assert payload["hits"][0]["source"] == "doc://alpha"


def test_rag_debug_tool_times_out_indexed_sources_gracefully() -> None:
    tool = RAGDebugTool(retriever=_RetrieverStub(), vector_store=_SlowVectorStoreStub())  # type: ignore[arg-type]
    payload = asyncio.run(
        tool.run(
            {
                "query": "policy controls",
                "include_indexed_sources": True,
                "indexed_sources_timeout_sec": 0.25,
            }
        )
    )
    assert payload["status"] == "ok"
    assert payload["hit_count"] == 1
    assert payload["indexed_source_count"] == 0
    assert payload["indexed_sources_timed_out"] is True


def test_filesystem_tool_list_and_read_respects_workspace_root(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir(parents=True)
    sample = root / "notes.txt"
    sample.write_text("hello tools", encoding="utf-8")

    tool = FileSystemTool(workspace_root=root)
    listed = asyncio.run(tool.run({"action": "list", "path": "."}))
    assert listed["status"] == "ok"
    assert any(item.get("name") == "notes.txt" for item in listed["entries"])

    read = asyncio.run(tool.run({"action": "read", "path": "notes.txt"}))
    assert read["status"] == "ok"
    assert "hello tools" in read["text"]

    escaped = asyncio.run(tool.run({"action": "read", "path": "../outside.txt"}))
    assert escaped["status"] == "error"
    assert "outside_workspace" in escaped["error"]


def test_system_metrics_tool_returns_runtime_keys(tmp_path: Path) -> None:
    tool = SystemMetricsTool(workspace_root=tmp_path)
    payload = asyncio.run(tool.run({}))
    assert payload["status"] == "ok"
    assert payload["cpu_count"] >= 0
    assert "disk_free_bytes" in payload


def test_network_tools_validate_required_input() -> None:
    web = WebSearchTool()
    url_fetch = URLFetchTool()
    web_result = asyncio.run(web.run({"query": ""}))
    fetch_result = asyncio.run(url_fetch.run({"query": "no url here"}))
    assert web_result["status"] == "error"
    assert web_result["error"] == "missing_query"
    assert fetch_result["status"] == "error"
    assert "no_url" in fetch_result["error"]


def test_extract_search_query_uses_previous_user_intent_on_retry() -> None:
    composed_query = (
        "Conversation context (most recent turns):\n"
        "USER: Hey search online about NTT data\n"
        "ASSISTANT: I could not complete web search.\n"
        "USER: try again\n\n"
        "Current user request:\n"
        "try again"
    )
    extracted = _extract_search_query(composed_query)
    assert "ntt data" in extracted.lower()
    assert "current user request" not in extracted.lower()


def test_web_search_tool_falls_back_when_wikipedia_forbidden(monkeypatch) -> None:
    class _FakeResponse:
        def __init__(self, *, url: str, status_code: int, payload: object, text: str = "") -> None:
            self.url = url
            self.status_code = status_code
            self._payload = payload
            self.text = text
            self.request = httpx.Request("GET", url)

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"Client error '{self.status_code}' for url '{self.url}'",
                    request=self.request,
                    response=httpx.Response(status_code=self.status_code, request=self.request),
                )

        def json(self) -> object:
            return self._payload

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401, ANN002, ANN003
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001, ANN201
            return False

        async def get(self, url: str, params: dict | None = None) -> _FakeResponse:
            if "wikipedia.org" in url:
                return _FakeResponse(url=url, status_code=403, payload={"error": "forbidden"})
            if "html.duckduckgo.com" in url:
                return _FakeResponse(
                    url=url,
                    status_code=200,
                    payload={},
                    text=(
                        '<a class="result__a" href="https://example.com/ntt-data">NTT DATA - Profile</a>'
                        '<span class="result__snippet">NTT DATA is a global IT services company.</span>'
                    ),
                )
            return _FakeResponse(url=url, status_code=404, payload={})

    monkeypatch.setattr("app.tools.builtin.httpx.AsyncClient", _FakeAsyncClient)

    tool = WebSearchTool(timeout_sec=2.0, max_results=3)
    payload = asyncio.run(
        tool.run(
            {
                "query": (
                    "Conversation context (most recent turns):\n"
                    "USER: Hey search online about NTT data\n"
                    "USER: try again\n\nCurrent user request:\ntry again"
                )
            }
        )
    )
    assert payload["status"] == "ok"
    assert payload["provider"] == "duckduckgo_html"
    assert payload["results"]
    assert "ntt data" in payload["query"].lower()


def test_vision_probe_uses_retrieved_image_candidate() -> None:
    tool = VisionProbeTool(vision_client=_VisionClientStub())  # type: ignore[arg-type]
    payload = asyncio.run(
        tool.run(
            {
                "query": "what appears in the image?",
                "retrieved_context": [
                    {
                        "source": "file:///tmp/picture.png",
                        "modality": "image",
                        "chunk_id": 3,
                        "score": 0.91,
                        "snippet": "Image source",
                    }
                ],
            }
        )
    )
    assert payload["status"] == "ok"
    assert payload["modality"] == "image"
    assert payload["source"] == "file:///tmp/picture.png"
    assert "what appears in the image" in payload["prompt"].lower()


def test_video_probe_uses_retrieved_video_candidate_and_timestamp_hint() -> None:
    tool = VideoProbeTool(video_client=_VideoClientStub(), default_sample_fps=2.0, default_max_frames=18)  # type: ignore[arg-type]
    payload = asyncio.run(
        tool.run(
            {
                "query": "what happens at the key event?",
                "retrieved_context": [
                    {
                        "source": "file:///tmp/clip.mp4",
                        "modality": "video",
                        "chunk_id": 4,
                        "timestamp_sec": 12.5,
                        "frame_index": 25,
                        "score": 0.88,
                        "snippet": "event marker",
                    }
                ],
            }
        )
    )
    assert payload["status"] == "ok"
    assert payload["modality"] == "video"
    assert payload["source"] == "file:///tmp/clip.mp4"
    assert payload["sample_fps"] == 2.0
    assert payload["max_frames"] == 18
    assert payload["timestamp_sec"] == 12.5
    assert isinstance(payload.get("text_evidence"), list)
    assert payload["text_evidence"]
    assert payload["text_evidence_status"] == "missing_in_context"


def test_video_probe_collects_transcript_evidence_from_retrieved_context() -> None:
    tool = VideoProbeTool(video_client=_VideoClientStub(), default_sample_fps=1.0, default_max_frames=12)  # type: ignore[arg-type]
    payload = asyncio.run(
        tool.run(
            {
                "query": "what is said in the clip?",
                "retrieved_context": [
                    {
                        "source": "file:///tmp/clip.mp4",
                        "modality": "video",
                        "chunk_id": 2,
                        "timestamp_sec": 1.0,
                        "score": 0.95,
                        "snippet": "Visual event: Person gives thumbs up.",
                    },
                    {
                        "source": "file:///tmp/clip.mp4",
                        "modality": "video",
                        "chunk_id": 9,
                        "timestamp_sec": 1.1,
                        "score": 0.65,
                        "snippet": "Audio event: Me gustan los tacos",
                    },
                ],
            }
        )
    )
    assert payload["status"] == "ok"
    assert payload["text_evidence_status"] == "found"
    assert any("me gustan los tacos" in str(item).lower() for item in payload["text_evidence"])
