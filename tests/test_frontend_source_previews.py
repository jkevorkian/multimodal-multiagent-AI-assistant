from __future__ import annotations

from frontend.streamlit_app import (
    _detect_source_type_from_sources,
    _extract_timestamp_from_snippet,
    _guess_modality_from_source,
    _resolve_local_media_path,
)


def test_resolve_local_media_path_supports_file_uri(tmp_path) -> None:
    source = tmp_path / "sample.png"
    source.write_bytes(b"png")
    resolved = _resolve_local_media_path(source.as_uri())
    assert resolved is not None
    assert resolved == source


def test_guess_modality_from_source_detects_common_extensions() -> None:
    assert _guess_modality_from_source("https://example.com/a.jpg", fallback="text") == "image"
    assert _guess_modality_from_source("https://example.com/b.mp4", fallback="text") == "video"
    assert _guess_modality_from_source("https://example.com/c.pdf", fallback="text") == "pdf"
    assert _guess_modality_from_source("https://example.com/d.bin", fallback="text") == "text"


def test_extract_timestamp_from_snippet_parses_temporal_marker() -> None:
    snippet = "Key event [t=12.75s] object enters frame."
    assert _extract_timestamp_from_snippet(snippet) == 12.75
    assert _extract_timestamp_from_snippet("No marker here") is None


def test_detect_source_type_prefers_specific_when_uniform() -> None:
    assert _detect_source_type_from_sources(["https://example.com/a.mp4"]) == "video"
    assert _detect_source_type_from_sources(["https://example.com/a.pdf"]) == "pdf"
    assert _detect_source_type_from_sources(["https://example.com/a.md"]) == "markdown"


def test_detect_source_type_returns_mixed_for_cross_modality_inputs() -> None:
    sources = [
        "https://example.com/a.mp4",
        "https://example.com/b.pdf",
    ]
    assert _detect_source_type_from_sources(sources) == "mixed"
