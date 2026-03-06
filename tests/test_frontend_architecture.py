from frontend.architecture import build_architecture_dot, high_level_flow_points


def test_architecture_dot_contains_core_nodes() -> None:
    dot = build_architecture_dot()
    assert "Streamlit Frontend" in dot
    assert "FastAPI Backend" in dot
    assert "AgentOrchestrator" in dot
    assert "Hybrid Retriever" in dot
    assert "VisionPreprocessor" in dot
    assert "VisionFusion" in dot
    assert "LLM Provider Selector" in dot


def test_high_level_flow_points_present() -> None:
    points = high_level_flow_points()
    assert len(points) >= 5
    assert any("Streamlit" in point for point in points)
    assert any("RAG" in point for point in points)
