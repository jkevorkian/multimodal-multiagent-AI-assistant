from __future__ import annotations


def build_architecture_dot() -> str:
    return """
digraph MMAA {
    graph [rankdir=LR, fontsize=10, fontname="Helvetica", labelloc="t", label="Multimodal Multi-Agent Assistant - High-Level Architecture"];
    node [shape=box, style="rounded,filled", color="#2D3748", fillcolor="#EDF2F7", fontname="Helvetica", fontsize=10];
    edge [color="#4A5568", arrowsize=0.7, fontname="Helvetica", fontsize=9];

    user [label="User"];
    streamlit [label="Streamlit Frontend\\n(M2.2)"];
    api [label="FastAPI Backend\\n/api routes"];

    subgraph cluster_agents {
        label="Agentic Orchestration (M2)";
        color="#CBD5E0";
        orchestrator [label="AgentOrchestrator"];
        research [label="ResearchAgent"];
        analyst [label="AnalystAgent"];
        answer [label="AnswerAgent"];
        registry [label="ToolRegistry"];
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
        label="Multimodal Endpoints";
        color="#CBD5E0";
        vision [label="/vision/analyze"];
        video [label="/video/analyze"];
    }

    observability [label="Logging + Metrics"];

    user -> streamlit [label="inputs / inspection"];
    streamlit -> api [label="HTTP JSON"];

    api -> ingestion [label="/ingest/documents"];
    api -> retriever [label="/query"];
    api -> orchestrator [label="/agents/run"];
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
    orchestrator -> checkpoint;

    api -> observability;
}
"""


def high_level_flow_points() -> list[str]:
    return [
        "User interacts with Streamlit, which calls FastAPI endpoints over HTTP.",
        "RAG path: ingest -> chunk -> embed -> persist -> retrieve -> answer with citations.",
        "Agent path: orchestrator executes research, analysis, and answer stages with bounded tools.",
        "Tool calls are routed through the registry with timeout/retry controls and checkpoint-safe flow.",
        "Vector storage is adapter-based: external DB (e.g., Qdrant) can coexist with pgvector/in-memory fallback.",
        "Vision/video routes stay independent and can feed future multimodal fusion logic.",
        "Metrics and logs expose operational behavior for debugging and milestone evaluation.",
    ]

