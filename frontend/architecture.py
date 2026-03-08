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
        label="Multimodal Layer (M3-M4.1)";
        color="#CBD5E0";
        preprocess [label="VisionPreprocessor\\n(validate + page-image resolution)"];
        vision_adapter [label="VisionAdapter"];
        vision_fusion [label="VisionFusion"];
        frame_sampler [label="VideoFrameSampler\\n(strict decoded frames)"];
        frame_vlm [label="Per-frame Vision Analysis\\n(VLM over sampled frames)"];
        temporal_agg [label="TemporalAggregator"];
        video_adapter [label="VideoAnalysisAdapter"];
        vision [label="/vision/analyze"];
        video [label="/video/analyze"];
        ingest_sources [label="/ingest/sources"];
        mm_clients [label="Multimodal Clients\\n(OpenAI vision / heuristic fallback)"];
    }

    subgraph cluster_runtime {
        label="Runtime Controls (M5 planned)";
        color="#CBD5E0";
        context_compactor [label="ContextCompactor\\n(Codex-style compaction)"];
        steering [label="SteeringPolicy\\n(style/risk/tool constraints)"];
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
    api -> ingest_sources;
    api -> retriever [label="/query"];
    api -> orchestrator [label="/agents/run"];
    api -> tool_catalog;
    api -> vision;
    api -> video;

    ingestion -> embeddings;
    embeddings -> store;
    retriever -> store;
    ingest_sources -> store [style=dashed];

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
    video -> video_adapter;
    video_adapter -> mm_clients;
    video_adapter -> frame_sampler;
    video_adapter -> frame_vlm;
    video_adapter -> temporal_agg;
    frame_vlm -> temporal_agg;

    orchestrator -> context_compactor [style=dashed];
    answer -> steering [style=dashed];
    research -> steering [style=dashed];

    api -> observability;
}
"""


def build_agents_pipeline_dot(executed_steps: list[str] | None = None, tool_calls: list[str] | None = None) -> str:
    executed = set(executed_steps or [])
    tools_called = tool_calls or []

    research_fill = "#C6F6D5" if "research_agent" in executed else "#EDF2F7"
    analyst_fill = "#C6F6D5" if "analyst_agent" in executed else "#EDF2F7"
    answer_fill = "#C6F6D5" if "answer_agent" in executed else "#EDF2F7"
    tools_fill = "#FEEBC8" if tools_called else "#EDF2F7"
    tools_label = "ToolRegistry\\n(optional tool execution)"
    if tools_called:
        tools_label = f"ToolRegistry\\ncalled={len(tools_called)}"

    return f"""
digraph AgentPipeline {{
    graph [rankdir=LR, fontsize=10, fontname="Helvetica", labelloc="t", label="Agent Pipeline (runtime view)"];
    node [shape=box, style="rounded,filled", color="#2D3748", fillcolor="#EDF2F7", fontname="Helvetica", fontsize=10];
    edge [color="#4A5568", arrowsize=0.7, fontname="Helvetica", fontsize=9];

    start [shape=circle, label="START", fillcolor="#E2E8F0"];
    end [shape=doublecircle, label="END", fillcolor="#E2E8F0"];

    research [label="ResearchAgent\\nretrieve context + select tools\\nstep=research_agent", fillcolor="{research_fill}"];
    analyst [label="AnalystAgent\\nanalyze evidence + confidence\\nstep=analyst_agent", fillcolor="{analyst_fill}"];
    answer [label="AnswerAgent\\nLLM synthesis\\nstep=answer_agent", fillcolor="{answer_fill}"];
    tools [label="{tools_label}", fillcolor="{tools_fill}"];

    start -> research;
    research -> analyst;
    analyst -> answer;
    answer -> end;
    research -> tools [style=dashed, label="tool_budget + allowed_tools"];
}}
"""


def build_agents_revision_loop_dot() -> str:
    return """
digraph AgentRevisionLoop {
    graph [rankdir=LR, fontsize=10, fontname="Helvetica", labelloc="t", label="Agent Revision Loop (M2.3 planned)"];
    node [shape=box, style="rounded,filled", color="#2D3748", fillcolor="#EDF2F7", fontname="Helvetica", fontsize=10];
    edge [color="#4A5568", arrowsize=0.7, fontname="Helvetica", fontsize=9];

    start [shape=circle, label="START", fillcolor="#E2E8F0"];
    end [shape=doublecircle, label="END", fillcolor="#E2E8F0"];
    safe_exit [shape=doublecircle, label="SAFE_EXIT\\n(partial answer + guardrail flag)", fillcolor="#FED7D7"];

    planner [label="Planner / Research\\nselect next action"];
    executor [label="Executor\\nrun reasoning/tool step"];
    tools [label="Tool Layer\\n(optional; bounded)"];
    critic [label="Critic / Verifier\\ncheck evidence + policy + quality"];
    answer [label="Answer Composer\\nfinal grounded response"];
    router [shape=diamond, label="Revision Gate\\napprove? / revise?"];

    step_guard [label="Max Step Guard\\nmax_steps", fillcolor="#FEEBC8"];
    tool_guard [label="Tool Budget Guard\\nmax_tool_calls / retries", fillcolor="#FEEBC8"];
    stagnation_guard [label="Stagnation Guard\\nno state delta N times", fillcolor="#FEEBC8"];
    timeout_guard [label="Timeout Guard\\nwall-clock SLA", fillcolor="#FEEBC8"];

    start -> planner;
    planner -> executor;
    executor -> tools [style=dashed, label="if tool needed"];
    tools -> critic [style=dashed];
    executor -> critic;
    critic -> router;
    router -> answer [label="approve"];
    answer -> end;

    router -> planner [label="revise\\nif remaining budget", color="#2B6CB0"];

    router -> step_guard [style=dotted];
    router -> tool_guard [style=dotted];
    router -> stagnation_guard [style=dotted];
    router -> timeout_guard [style=dotted];
    step_guard -> safe_exit [color="#C53030"];
    tool_guard -> safe_exit [color="#C53030"];
    stagnation_guard -> safe_exit [color="#C53030"];
    timeout_guard -> safe_exit [color="#C53030"];
}
"""


def agent_pipeline_state_rows() -> list[dict[str, str]]:
    return [
        {
            "stage": "research_agent",
            "inputs": "query, allowed_tools, tool_budget",
            "state_updates": "steps, retrieved_context, tool_outputs, tool_calls, errors",
            "logic": "retrieves top-k evidence, then optionally calls selected tools",
        },
        {
            "stage": "analyst_agent",
            "inputs": "retrieved_context, tool_outputs",
            "state_updates": "steps, analysis_notes, confidence",
            "logic": "summarizes retrieval/tool outcomes and computes confidence",
        },
        {
            "stage": "answer_agent",
            "inputs": "query, top snippets, analysis_notes",
            "state_updates": "steps, final_answer, confidence, errors",
            "logic": "synthesizes grounded answer with LLM client",
        },
    ]


def agent_loop_guardrail_rows() -> list[dict[str, str]]:
    return [
        {
            "guardrail": "max_steps",
            "purpose": "Hard-stop unbounded plan/revision cycles",
            "trigger_action": "Exit loop with best-effort answer + guardrail flag",
        },
        {
            "guardrail": "max_tool_calls",
            "purpose": "Cap tool-induced latency/cost growth",
            "trigger_action": "Disable further tool calls and continue with available evidence",
        },
        {
            "guardrail": "stagnation_detector",
            "purpose": "Detect repeated revisions with no state improvement",
            "trigger_action": "Break loop and return partial result with diagnostic hint",
        },
        {
            "guardrail": "timeout_budget",
            "purpose": "Guarantee bounded wall-clock response time",
            "trigger_action": "Terminate loop and emit partial/abstained response",
        },
    ]


def live_status_event_rows() -> list[dict[str, str]]:
    return [
        {
            "event_type": "run.started",
            "user_text": "Starting run...",
            "payload_core": "run_id, trace_id, query_summary",
        },
        {
            "event_type": "agent.step.started",
            "user_text": "Researching / analyzing...",
            "payload_core": "agent_name, step_index, budget_remaining",
        },
        {
            "event_type": "tool.call.started",
            "user_text": "Using tool: {tool_name}",
            "payload_core": "tool_name, call_id, timeout_sec",
        },
        {
            "event_type": "tool.call.completed",
            "user_text": "Tool finished: {tool_name}",
            "payload_core": "tool_name, call_id, latency_ms, ok/error",
        },
        {
            "event_type": "agent.revision.requested",
            "user_text": "Revising answer draft...",
            "payload_core": "reason_code, iteration, remaining_steps",
        },
        {
            "event_type": "model.call.in_progress",
            "user_text": "Thinking...",
            "payload_core": "model, stream_seq",
        },
        {
            "event_type": "run.completed",
            "user_text": "Finalizing answer...",
            "payload_core": "final_confidence, citations_count",
        },
        {
            "event_type": "run.guardrail_triggered",
            "user_text": "Stopped safely: budget/loop guard triggered",
            "payload_core": "guard_name, iteration, partial_result",
        },
    ]


def high_level_flow_points() -> list[str]:
    return [
        "User interacts with Streamlit, which calls FastAPI endpoints over HTTP.",
        "RAG path: ingest -> chunk -> embed -> persist -> retrieve -> answer with citations.",
        "Indexed Sources path: /ingest/sources summarizes what is currently stored in the vector layer.",
        "Agent path: LangGraph orchestrator executes research, analysis, and answer stages with bounded tools.",
        "Tool calls are routed through the registry with timeout/retry controls; /agents/tools exposes discoverable tool names/descriptions.",
        "LLM path uses provider selection: OpenAI when configured, grounded heuristic fallback otherwise.",
        "Vision path runs preprocess -> adapter -> fusion so findings carry explicit evidence tags; webpage URLs can be resolved to concrete image assets.",
        "Video path (M4.1) runs strict sampled frame decode + per-frame VLM analysis + temporal aggregation for timeline-oriented key events.",
        "M2.3 adds live run-status telemetry (SSE/event stream) so users can see step-by-step progress across model calls, agent stages, and tools.",
        "Advanced agent flow supports bounded revision loops with explicit exit guardrails (max steps, max tool calls, stagnation, timeout).",
        "Runtime hardening roadmap adds context compaction (Codex-style summary checkpoints when context grows) and configurable steering policies.",
        "Vector storage is adapter-based: external DB (e.g., Qdrant) can coexist with pgvector/in-memory fallback.",
        "Multimodal ingestion can index image/video descriptors into the same retrieval pipeline as text.",
        "Metrics and logs expose operational behavior for debugging and milestone evaluation.",
    ]
