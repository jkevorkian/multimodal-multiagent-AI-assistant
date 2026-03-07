from __future__ import annotations

import hashlib
import io
import tempfile
import uuid
from pathlib import Path
from typing import Any

import httpx
import streamlit as st

try:
    from streamlit_paste_button import paste_image_button
except Exception:  # pragma: no cover - optional dependency
    paste_image_button = None

try:
    from frontend.architecture import (
        agent_pipeline_state_rows,
        build_agents_pipeline_dot,
        build_architecture_dot,
        high_level_flow_points,
    )
except ModuleNotFoundError:
    from architecture import (
        agent_pipeline_state_rows,
        build_agents_pipeline_dot,
        build_architecture_dot,
        high_level_flow_points,
    )


DEFAULT_BACKEND_URL = "http://localhost:8000"
REQUEST_TIMEOUT_SEC = 120.0
INGEST_UPLOAD_TYPES = [
    "pdf",
    "txt",
    "md",
    "markdown",
    "docx",
    "pptx",
    "xlsx",
    "html",
    "htm",
    "csv",
    "tsv",
    "json",
    "jsonl",
    "yaml",
    "yml",
    "xml",
    "log",
    "ini",
    "cfg",
    "toml",
    "png",
    "jpg",
    "jpeg",
    "webp",
    "gif",
    "bmp",
    "tif",
    "tiff",
    "mp4",
    "mov",
    "avi",
    "mkv",
    "webm",
    "m4v",
    "mpeg",
    "mpg",
]
IMAGE_UPLOAD_TYPES = ["png", "jpg", "jpeg", "webp", "gif", "bmp", "tif", "tiff"]
VIDEO_UPLOAD_TYPES = ["mp4", "mov", "avi", "mkv", "webm", "m4v", "mpeg", "mpg"]


def _request_json(
    method: str,
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any] | list[Any] | str]:
    url = f"{base_url.rstrip('/')}{path}"
    timeout_sec = float(st.session_state.get("request_timeout_sec", REQUEST_TIMEOUT_SEC))
    try:
        with httpx.Client(timeout=timeout_sec) as client:
            response = client.request(method=method, url=url, json=payload)
        try:
            body: dict[str, Any] | list[Any] | str = response.json()
        except ValueError:
            body = response.text
        return response.status_code, body
    except httpx.HTTPError as error:
        return 0, {"error": str(error), "url": url}


def _render_response(status_code: int, body: dict[str, Any] | list[Any] | str) -> None:
    if status_code == 0:
        st.error("Request failed before reaching backend.")
    elif 200 <= status_code < 300:
        st.success(f"HTTP {status_code}")
    else:
        st.warning(f"HTTP {status_code}")
    if isinstance(body, (dict, list)):
        st.json(body)
    else:
        st.code(str(body), language="text")


def _auto_select_answer_mode(query: str, selected_tools: list[str]) -> str:
    if selected_tools:
        return "Agentic Run"
    lowered = query.lower()
    agentic_keywords = (
        "latest",
        "search",
        "lookup",
        "tool",
        "api",
        "compare",
        "plan",
        "investigate",
    )
    if any(keyword in lowered for keyword in agentic_keywords):
        return "Agentic Run"
    return "RAG Query"


def _fetch_agent_tools(base_url: str) -> tuple[list[str], dict[str, str]]:
    status_code, body = _request_json("GET", base_url, "/agents/tools")
    if not (200 <= status_code < 300):
        return [], {}
    if not isinstance(body, dict):
        return [], {}
    raw_tools = body.get("tools", [])
    if not isinstance(raw_tools, list):
        return [], {}

    names: list[str] = []
    descriptions: dict[str, str] = {}
    for item in raw_tools:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        description = str(item.get("description", "")).strip()
        names.append(name)
        descriptions[name] = description
    return names, descriptions


def _persist_uploaded_files(uploaded_files: list[Any] | None) -> list[str]:
    if not uploaded_files:
        return []

    upload_root = Path(tempfile.gettempdir()) / "mmaa_streamlit_uploads"
    upload_root.mkdir(parents=True, exist_ok=True)
    saved_uris: list[str] = []
    for uploaded in uploaded_files:
        original_name = Path(getattr(uploaded, "name", "upload.bin"))
        safe_stem = "".join(ch for ch in original_name.stem if ch.isalnum() or ch in {"-", "_"}).strip() or "upload"
        suffix = original_name.suffix or ".bin"
        destination = upload_root / f"{safe_stem}-{uuid.uuid4().hex}{suffix}"
        destination.write_bytes(uploaded.getvalue())
        saved_uris.append(destination.resolve().as_uri())
    return saved_uris


def _persist_clipboard_images(key_prefix: str) -> list[str]:
    if paste_image_button is None:
        st.caption("Install `streamlit-paste-button` to enable clipboard image paste (`Ctrl+V`).")
        return st.session_state.get(f"{key_prefix}_clipboard_sources", [])

    result = paste_image_button(
        label="Paste image from clipboard",
        key=f"{key_prefix}_paste_button",
    )

    sources_key = f"{key_prefix}_clipboard_sources"
    digest_key = f"{key_prefix}_clipboard_digest"
    saved_sources = list(st.session_state.get(sources_key, []))

    if result.image_data is not None:
        buffer = io.BytesIO()
        result.image_data.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        digest = hashlib.sha256(image_bytes).hexdigest()
        if st.session_state.get(digest_key) != digest:
            upload_root = Path(tempfile.gettempdir()) / "mmaa_streamlit_uploads"
            upload_root.mkdir(parents=True, exist_ok=True)
            destination = upload_root / f"{key_prefix}-clipboard-{uuid.uuid4().hex}.png"
            destination.write_bytes(image_bytes)
            saved_sources.append(destination.resolve().as_uri())
            st.session_state[sources_key] = saved_sources
            st.session_state[digest_key] = digest
            st.success("Clipboard image captured and added as source.")

    if saved_sources:
        st.caption(f"Clipboard images queued: {len(saved_sources)}")
    return saved_sources


def _render_answer_block(title: str, payload: dict[str, Any] | list[Any] | str) -> None:
    st.subheader(title)
    if not isinstance(payload, dict):
        st.write(payload)
        return
    if payload.get("answer"):
        st.markdown(f"**Answer**: {payload['answer']}")
    if payload.get("summary"):
        st.markdown(f"**Summary**: {payload['summary']}")
    if payload.get("confidence") is not None:
        st.markdown(f"**Confidence**: `{payload['confidence']}`")
    if payload.get("citations"):
        st.markdown("**Citations**")
        for citation in payload["citations"]:
            st.markdown(f"- `{citation}`")
    if payload.get("findings"):
        st.markdown("**Findings**")
        for finding in payload["findings"]:
            st.markdown(f"- {finding}")
    if payload.get("key_events"):
        st.markdown("**Key Events**")
        for event in payload["key_events"]:
            st.markdown(f"- {event}")
    st.caption("Raw payload")
    st.json(payload)


def _extract_agent_execution_from_payload(payload: dict[str, Any] | list[Any] | str) -> tuple[list[str], list[str]]:
    if not isinstance(payload, dict):
        return [], []
    raw_steps = payload.get("steps", [])
    raw_tool_calls = payload.get("tool_calls", [])
    steps = [str(item) for item in raw_steps] if isinstance(raw_steps, list) else []
    tool_calls = [str(item) for item in raw_tool_calls] if isinstance(raw_tool_calls, list) else []
    return steps, tool_calls


def _render_agents_pipeline_helper(latest_payload: dict[str, Any] | list[Any] | str) -> None:
    steps, tool_calls = _extract_agent_execution_from_payload(latest_payload)
    st.markdown("### Pipeline Visualizer")
    st.caption("Graph shows agent stages, execution order, and optional tool branch from research.")
    st.graphviz_chart(build_agents_pipeline_dot(executed_steps=steps, tool_calls=tool_calls))
    if steps:
        st.caption(f"Last run steps: {' -> '.join(steps)}")
    if tool_calls:
        st.caption(f"Last run tool calls: {', '.join(tool_calls)}")
    st.markdown("### Stage Logic and State Changes")
    st.dataframe(agent_pipeline_state_rows(), hide_index=True, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="MMAA Frontend", page_icon=":material/hub:", layout="wide")
    st.title("Multimodal Multi-Agent Assistant")
    st.caption("M2.2+M3 frontend: user-friendly assistant flow plus route playground.")

    with st.sidebar:
        st.subheader("Backend Connection")
        backend_url = st.text_input("FastAPI base URL", value=DEFAULT_BACKEND_URL)
        st.number_input(
            "Request timeout (seconds)",
            min_value=10.0,
            max_value=600.0,
            value=REQUEST_TIMEOUT_SEC,
            step=5.0,
            key="request_timeout_sec",
        )
        st.caption("Run backend first, then use tabs to exercise each route.")
    available_tool_names, available_tool_descriptions = _fetch_agent_tools(backend_url)

    (
        implementation_tab,
        architecture_tab,
        health_tab,
        ingest_tab,
        indexed_sources_tab,
        query_tab,
        agents_tab,
        vision_tab,
        video_tab,
        metrics_tab,
    ) = st.tabs(
        [
            "Implementation",
            "Architecture",
            "Health",
            "Ingest",
            "Indexed Sources",
            "Query",
            "Agents",
            "Vision",
            "Video",
            "Metrics",
        ]
    )

    with implementation_tab:
        st.subheader("Actual Solution (User Flow)")
        st.caption("This tab mirrors the real workflow: index sources, ask questions, and analyze media.")

        st.markdown("### 1) Index Knowledge Sources")
        impl_sources = st.text_area(
            "Paste one source per line (URLs, file paths, image/video URIs)",
            value="https://example.com\nhttps://example.com/sample.png",
            height=120,
            key="impl_sources",
        )
        impl_source_type = st.selectbox(
            "Source type",
            options=["mixed", "text", "url", "pdf", "docx", "pptx", "xlsx", "markdown", "html", "image", "video"],
            key="impl_source_type",
        )
        impl_uploaded_files = st.file_uploader(
            "Upload files from your machine",
            accept_multiple_files=True,
            type=INGEST_UPLOAD_TYPES,
            key="impl_uploaded_files",
        )
        impl_clipboard_sources = _persist_clipboard_images("impl")
        st.caption("Uploads are saved in your local temp folder and passed to backend as `file://` URIs.")
        if st.button("Index Sources", key="impl_ingest_button"):
            manual_sources = [line.strip() for line in impl_sources.splitlines() if line.strip()]
            uploaded_sources = _persist_uploaded_files(impl_uploaded_files)
            sources = manual_sources + uploaded_sources + impl_clipboard_sources
            if not sources:
                st.error("Add at least one source or upload one file.")
            else:
                if uploaded_sources:
                    st.info(f"Prepared {len(uploaded_sources)} uploaded file(s) for ingestion.")
                status_code, body = _request_json(
                    "POST",
                    backend_url,
                    "/ingest/documents",
                    payload={"sources": sources, "source_type": impl_source_type},
                )
                _render_response(status_code, body)

        st.markdown("### 2) Ask the Assistant")
        impl_query = st.text_input("Question", value="What does the indexed material say?", key="impl_query")
        impl_mode = st.radio("Mode", options=["Auto", "RAG Query", "Agentic Run"], horizontal=True, key="impl_mode")
        impl_top_k = st.slider("Top K context chunks", min_value=1, max_value=20, value=5, key="impl_top_k")
        impl_tools = st.multiselect(
            "Agent tools (optional)",
            options=available_tool_names,
            default=[],
            format_func=lambda name: (
                f"{name} - {available_tool_descriptions.get(name, '')}"
                if available_tool_descriptions.get(name)
                else name
            ),
            key="impl_tools",
        )
        if not available_tool_names:
            st.info("No tool catalog available from backend right now.")
        st.caption(
            "RAG Query: direct grounded answer from retrieved chunks. "
            "Agentic Run: multi-step orchestration with tools. "
            "Auto: heuristic route selection. If no tools are selected, backend enables all available tools."
        )
        if st.button("Get Answer", key="impl_answer_button"):
            effective_mode = impl_mode if impl_mode != "Auto" else _auto_select_answer_mode(impl_query, impl_tools)
            if impl_mode == "Auto":
                st.info(f"Auto selected mode: {effective_mode}")
            if effective_mode == "RAG Query":
                status_code, body = _request_json(
                    "POST",
                    backend_url,
                    "/query",
                    payload={"query": impl_query, "top_k": impl_top_k},
                )
            else:
                payload: dict[str, Any] = {"query": impl_query}
                if impl_tools:
                    payload["tools"] = impl_tools
                status_code, body = _request_json("POST", backend_url, "/agents/run", payload=payload)
            _render_response(status_code, body)
            _render_answer_block("Assistant Output", body)

        st.markdown("### 3) Analyze Image / Video")
        left, right = st.columns(2)
        with left:
            image_uri = st.text_input(
                "Image URI/Path",
                value="https://example.com/traffic-sign.jpg",
                key="impl_image_uri",
            )
            impl_image_upload = st.file_uploader(
                "Or upload image for analysis",
                accept_multiple_files=False,
                type=IMAGE_UPLOAD_TYPES,
                key="impl_image_upload",
            )
            image_prompt = st.text_input("Image prompt (optional)", value="Describe key entities.", key="impl_image_prompt")
            if st.button("Analyze Image", key="impl_image_button"):
                effective_image_uri = image_uri.strip()
                if impl_image_upload is not None:
                    uploaded_sources = _persist_uploaded_files([impl_image_upload])
                    if uploaded_sources:
                        effective_image_uri = uploaded_sources[0]
                        st.info("Using uploaded image file for analysis.")
                if not effective_image_uri:
                    st.error("Provide an image URI/path or upload an image file.")
                else:
                    status_code, body = _request_json(
                        "POST",
                        backend_url,
                        "/vision/analyze",
                        payload={"image_uri": effective_image_uri, "prompt": image_prompt or None},
                    )
                    _render_response(status_code, body)
                    _render_answer_block("Vision Output", body)

        with right:
            video_uri = st.text_input(
                "Video URI/Path",
                value="https://example.com/street-scene.mp4",
                key="impl_video_uri",
            )
            impl_video_upload = st.file_uploader(
                "Or upload video for analysis",
                accept_multiple_files=False,
                type=VIDEO_UPLOAD_TYPES,
                key="impl_video_upload",
            )
            video_prompt = st.text_input("Video prompt (optional)", value="Summarize key events.", key="impl_video_prompt")
            sample_fps = st.number_input("Video sample FPS", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
            max_frames = st.number_input("Video max frames", min_value=1, max_value=1000, value=32, step=1)
            if st.button("Analyze Video", key="impl_video_button"):
                effective_video_uri = video_uri.strip()
                if impl_video_upload is not None:
                    uploaded_sources = _persist_uploaded_files([impl_video_upload])
                    if uploaded_sources:
                        effective_video_uri = uploaded_sources[0]
                        st.info("Using uploaded video file for analysis.")
                if not effective_video_uri:
                    st.error("Provide a video URI/path or upload a video file.")
                else:
                    status_code, body = _request_json(
                        "POST",
                        backend_url,
                        "/video/analyze",
                        payload={
                            "video_uri": effective_video_uri,
                            "prompt": video_prompt or None,
                            "sample_fps": float(sample_fps),
                            "max_frames": int(max_frames),
                        },
                    )
                    _render_response(status_code, body)
                    _render_answer_block("Video Output", body)

    with architecture_tab:
        st.subheader("High-Level Architecture")
        st.graphviz_chart(build_architecture_dot())
        st.subheader("Flow Summary")
        for item in high_level_flow_points():
            st.markdown(f"- {item}")

    with health_tab:
        st.subheader("GET /health")
        if st.button("Check health", key="health_button"):
            status_code, body = _request_json("GET", backend_url, "/health")
            _render_response(status_code, body)

    with ingest_tab:
        st.subheader("POST /ingest/documents")
        sources_text = st.text_area(
            "Sources (one URL/path per line)",
            value="https://example.com",
            height=140,
        )
        source_type = st.selectbox(
            "Source type",
            options=["mixed", "url", "pdf", "docx", "pptx", "xlsx", "markdown", "html", "text", "image", "video"],
        )
        uploaded_files = st.file_uploader(
            "Upload local files",
            accept_multiple_files=True,
            type=INGEST_UPLOAD_TYPES,
            key="play_ingest_uploaded_files",
        )
        play_clipboard_sources = _persist_clipboard_images("play_ingest")
        st.caption("Uploaded files are saved in your local temp folder before ingestion.")
        if st.button("Run ingest", key="ingest_button"):
            manual_sources = [line.strip() for line in sources_text.splitlines() if line.strip()]
            uploaded_sources = _persist_uploaded_files(uploaded_files)
            sources = manual_sources + uploaded_sources + play_clipboard_sources
            if not sources:
                st.error("Add at least one source or upload one file.")
            else:
                if uploaded_sources:
                    st.info(f"Prepared {len(uploaded_sources)} uploaded file(s) for ingestion.")
                status_code, body = _request_json(
                    "POST",
                    backend_url,
                    "/ingest/documents",
                    payload={"sources": sources, "source_type": source_type},
                )
                _render_response(status_code, body)

    with indexed_sources_tab:
        st.subheader("GET /ingest/sources")
        st.caption("Inspect indexed source coverage, chunking ranges, modality, and sample snippet.")
        if st.button("Refresh indexed sources", key="indexed_sources_refresh_button"):
            status_code, body = _request_json("GET", backend_url, "/ingest/sources")
            if 200 <= status_code < 300 and isinstance(body, dict):
                st.session_state["indexed_sources_payload"] = body
            _render_response(status_code, body)

        payload = st.session_state.get("indexed_sources_payload", {})
        if isinstance(payload, dict):
            raw_sources = payload.get("sources", [])
            if isinstance(raw_sources, list) and raw_sources:
                total_chunks = sum(int(item.get("chunk_count", 0) or 0) for item in raw_sources if isinstance(item, dict))
                modalities = sorted(
                    {
                        str(item.get("modality", "text"))
                        for item in raw_sources
                        if isinstance(item, dict) and str(item.get("modality", "")).strip()
                    }
                )
                st.caption(f"Indexed sources: {len(raw_sources)} | Total chunks: {total_chunks}")
                modality_filter = st.selectbox(
                    "Filter by modality",
                    options=["all"] + modalities,
                    key="indexed_sources_modality_filter",
                )
                filtered_sources = [
                    item
                    for item in raw_sources
                    if isinstance(item, dict)
                    and (modality_filter == "all" or str(item.get("modality", "text")) == modality_filter)
                ]
                st.dataframe(filtered_sources, hide_index=True, use_container_width=True)
            else:
                st.info("No indexed sources found yet. Run ingestion first and then refresh.")

    with query_tab:
        st.subheader("POST /query")
        query_text = st.text_input("Query", value="What is this project about?")
        top_k = st.slider("Top K", min_value=1, max_value=20, value=5)
        if st.button("Run query", key="query_button"):
            status_code, body = _request_json(
                "POST",
                backend_url,
                "/query",
                payload={"query": query_text, "top_k": top_k},
            )
            _render_response(status_code, body)

    with agents_tab:
        st.subheader("POST /agents/run")
        st.caption("Default agent pipeline: research_agent -> analyst_agent -> answer_agent.")
        latest_agents_payload = st.session_state.get("last_agents_run_payload", {})
        _render_agents_pipeline_helper(latest_agents_payload)
        agent_query = st.text_area("Agent query", value="Analyze available context and summarize findings.")
        selected_tools = st.multiselect(
            "Allowed tools (optional)",
            options=available_tool_names,
            default=[],
            format_func=lambda name: (
                f"{name} - {available_tool_descriptions.get(name, '')}"
                if available_tool_descriptions.get(name)
                else name
            ),
            key="agents_tools",
        )
        if not available_tool_names:
            st.info("No tool catalog available from backend right now.")
        else:
            st.caption("If none are selected, backend enables all listed tools.")
        if st.button("Run agents", key="agents_button"):
            payload: dict[str, Any] = {"query": agent_query}
            if selected_tools:
                payload["tools"] = selected_tools
            status_code, body = _request_json("POST", backend_url, "/agents/run", payload=payload)
            if 200 <= status_code < 300 and isinstance(body, dict):
                st.session_state["last_agents_run_payload"] = body
            _render_response(status_code, body)

    with vision_tab:
        st.subheader("POST /vision/analyze")
        image_uri = st.text_input("Image URI", value="https://example.com/image.png", key="play_image_uri")
        play_image_upload = st.file_uploader(
            "Or upload image for /vision/analyze",
            accept_multiple_files=False,
            type=IMAGE_UPLOAD_TYPES,
            key="play_image_upload",
        )
        prompt = st.text_input("Prompt", value="Describe relevant visual evidence.", key="play_image_prompt")
        if st.button("Run vision", key="vision_button"):
            effective_image_uri = image_uri.strip()
            if play_image_upload is not None:
                uploaded_sources = _persist_uploaded_files([play_image_upload])
                if uploaded_sources:
                    effective_image_uri = uploaded_sources[0]
                    st.info("Using uploaded image file for analysis.")
            if not effective_image_uri:
                st.error("Provide an image URI/path or upload an image file.")
            else:
                status_code, body = _request_json(
                    "POST",
                    backend_url,
                    "/vision/analyze",
                    payload={"image_uri": effective_image_uri, "prompt": prompt or None},
                )
                _render_response(status_code, body)

    with video_tab:
        st.subheader("POST /video/analyze")
        video_uri = st.text_input("Video URI", value="https://example.com/video.mp4", key="play_video_uri")
        play_video_upload = st.file_uploader(
            "Or upload video for /video/analyze",
            accept_multiple_files=False,
            type=VIDEO_UPLOAD_TYPES,
            key="play_video_upload",
        )
        prompt = st.text_input("Prompt", value="Summarize key temporal events.", key="play_video_prompt")
        sample_fps = st.number_input("Sample FPS", min_value=0.1, max_value=10.0, value=1.0, key="play_video_fps")
        max_frames = st.number_input("Max frames", min_value=1, max_value=1000, value=32, key="play_video_frames")
        if st.button("Run video", key="video_button"):
            effective_video_uri = video_uri.strip()
            if play_video_upload is not None:
                uploaded_sources = _persist_uploaded_files([play_video_upload])
                if uploaded_sources:
                    effective_video_uri = uploaded_sources[0]
                    st.info("Using uploaded video file for analysis.")
            if not effective_video_uri:
                st.error("Provide a video URI/path or upload a video file.")
            else:
                status_code, body = _request_json(
                    "POST",
                    backend_url,
                    "/video/analyze",
                    payload={
                        "video_uri": effective_video_uri,
                        "prompt": prompt or None,
                        "sample_fps": float(sample_fps),
                        "max_frames": int(max_frames),
                    },
                )
                _render_response(status_code, body)

    with metrics_tab:
        st.subheader("GET /metrics")
        if st.button("Fetch metrics", key="metrics_button"):
            status_code, body = _request_json("GET", backend_url, "/metrics")
            _render_response(status_code, body)


if __name__ == "__main__":
    main()
