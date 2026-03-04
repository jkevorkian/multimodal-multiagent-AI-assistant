from __future__ import annotations

from typing import Any

import httpx
import streamlit as st

from frontend.architecture import build_architecture_dot, high_level_flow_points


DEFAULT_BACKEND_URL = "http://localhost:8000"
REQUEST_TIMEOUT_SEC = 30.0


def _request_json(
    method: str,
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any] | list[Any] | str]:
    url = f"{base_url.rstrip('/')}{path}"
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT_SEC) as client:
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
    st.json(body)


def main() -> None:
    st.set_page_config(page_title="MMAA Frontend", page_icon=":material/hub:", layout="wide")
    st.title("Multimodal Multi-Agent Assistant")
    st.caption("M2.2 Streamlit frontend: architecture visualization + backend route playground.")

    with st.sidebar:
        st.subheader("Backend Connection")
        backend_url = st.text_input("FastAPI base URL", value=DEFAULT_BACKEND_URL)
        st.caption("Run backend first, then use tabs to exercise each route.")

    architecture_tab, health_tab, ingest_tab, query_tab, agents_tab, metrics_tab = st.tabs(
        ["Architecture", "Health", "Ingest", "Query", "Agents", "Metrics"]
    )

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
        source_type = st.selectbox("Source type", options=["mixed", "url", "pdf", "text"])
        if st.button("Run ingest", key="ingest_button"):
            sources = [line.strip() for line in sources_text.splitlines() if line.strip()]
            status_code, body = _request_json(
                "POST",
                backend_url,
                "/ingest/documents",
                payload={"sources": sources, "source_type": source_type},
            )
            _render_response(status_code, body)

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
        agent_query = st.text_area("Agent query", value="Analyze available context and summarize findings.")
        tools_csv = st.text_input("Tools (comma-separated, optional)", value="")
        if st.button("Run agents", key="agents_button"):
            payload: dict[str, Any] = {"query": agent_query}
            tools = [item.strip() for item in tools_csv.split(",") if item.strip()]
            if tools:
                payload["tools"] = tools
            status_code, body = _request_json("POST", backend_url, "/agents/run", payload=payload)
            _render_response(status_code, body)

    with metrics_tab:
        st.subheader("GET /metrics")
        if st.button("Fetch metrics", key="metrics_button"):
            status_code, body = _request_json("GET", backend_url, "/metrics")
            _render_response(status_code, body)


if __name__ == "__main__":
    main()
