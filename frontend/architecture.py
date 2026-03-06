from __future__ import annotations


def build_architecture_dot() -> str:
    return """
digraph MMAA {
    graph [rankdir=LR, fontsize=10, fontname="Helvetica", labelloc="t", label="Multimodal Multi-Agent Assistant - High-Level Architecture"];
    node [shape=box, style="rounded,filled", color="#2D3748", fillcolor="#EDF2F7", fontname="Helvetica", fontsize=10];
    edge [color="#4A5568", arrowsize=0.7, fontname="Helvetica", fontsize=9];

    user [label="User"];
    streamlit [label="Streamlit Frontend\\n(M2.2 + M3 UX)"];
    api [label="FastAPI Backend\\n/api routes"];

    subgraph cluster_agents {
        label="Agentic Orchestration (M2)";
        color="#CBD5E0";
        orchestrator [label="AgentOrchestrator"];
        research [label="ResearchAgent"];
        analyst [label="AnalystAgent"];
        answer [label="AnswerAgent"];
        registry [label="ToolRegistry"];
        tool_catalog [label="/agents/tools\\n(tool discovery)"];
        checkpoint [label="CheckpointStore"];
    }

    subgraph cluster_rag {
        label="RAG Pipeline (M1)";
        color="#CBD5E0";
        ingestion [label="Ingestion + Chunking"];
        embeddings [label="Embedding Providers"];
        retriever [label="Hybrid Retriever\\n(Dense + Lexical + RRF + Rerank)"];
        store [label="VectorStore Adapters\\n(Qdrant / pgvector / in-memory fallback)"];
    }

    subgraph cluster_mm {
        label="Multimodal Layer (M3)";
        color="#CBD5E0";
        preprocess [label="VisionPreprocessor\\n(validate + page-image resolution)"];
        vision_adapter [label="VisionAdapter"];
        vision_fusion [label="VisionFusion"];
        vision [label="/vision/analyze"];
        video [label="/video/analyze"];
        mm_clients [label="Multimodal Clients\\n(OpenAI vision / heuristic fallback)"];
    }

    subgraph cluster_llm {
        label="LLM Layer";
        color="#CBD5E0";
        llm_select [label="LLM Provider Selector"];
        llm_provider [label="OpenAI or\\nHeuristic Grounded LLM"];
    }

    observability [label="Logging + Metrics"];

    user -> streamlit [label="inputs / inspection"];
    streamlit -> api [label="HTTP JSON"];

    api -> ingestion [label="/ingest/documents"];
    api -> retriever [label="/query"];
    api -> orchestrator [label="/agents/run"];
    api -> tool_catalog;
    api -> vision;
    api -> video;

    ingestion -> embeddings;
    embeddings -> store;
    retriever -> store;

    orchestrator -> research;
    orchestrator -> analyst;
    orchestrator -> answer;
    research -> retriever;
    research -> registry;
    tool_catalog -> registry;
    orchestrator -> checkpoint;
    answer -> llm_select;
    llm_select -> llm_provider;

    vision -> preprocess;
    preprocess -> vision_adapter;
    vision_adapter -> mm_clients;
    vision -> vision_fusion;
    video -> mm_clients;

    api -> observability;
}
"""


def high_level_flow_points() -> list[str]:
    return [
        "User interacts with Streamlit, which calls FastAPI endpoints over HTTP.",
        "RAG path: ingest -> chunk -> embed -> persist -> retrieve -> answer with citations.",
        "Agent path: LangGraph orchestrator executes research, analysis, and answer stages with bounded tools.",
        "Tool calls are routed through the registry with timeout/retry controls; /agents/tools exposes discoverable tool names/descriptions.",
        "LLM path uses provider selection: OpenAI when configured, grounded heuristic fallback otherwise.",
        "Vision path runs preprocess -> adapter -> fusion so findings carry explicit evidence tags; webpage URLs can be resolved to concrete image assets.",
        "Vector storage is adapter-based: external DB (e.g., Qdrant) can coexist with pgvector/in-memory fallback.",
        "Multimodal ingestion can index image/video descriptors into the same retrieval pipeline as text.",
        "Metrics and logs expose operational behavior for debugging and milestone evaluation.",
    ]
