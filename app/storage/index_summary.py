from __future__ import annotations

from typing import Any


def summarize_indexed_sources(metadata_rows: list[dict[str, Any]], limit: int = 200) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for metadata in metadata_rows:
        source = str(metadata.get("source", "")).strip() or "unknown"
        modality = str(metadata.get("modality", "text")).strip() or "text"
        key = (source, modality)

        row = grouped.get(key)
        if row is None:
            row = {
                "source": source,
                "modality": modality,
                "chunk_count": 0,
                "min_chunk_id": None,
                "max_chunk_id": None,
                "min_offset": None,
                "max_offset": None,
                "sample_snippet": None,
            }
            grouped[key] = row

        row["chunk_count"] += 1

        chunk_id = _safe_int(metadata.get("chunk_id"))
        if chunk_id is not None:
            row["min_chunk_id"] = chunk_id if row["min_chunk_id"] is None else min(row["min_chunk_id"], chunk_id)
            row["max_chunk_id"] = chunk_id if row["max_chunk_id"] is None else max(row["max_chunk_id"], chunk_id)

        offset = _safe_int(metadata.get("offset"))
        if offset is not None:
            row["min_offset"] = offset if row["min_offset"] is None else min(row["min_offset"], offset)
            row["max_offset"] = offset if row["max_offset"] is None else max(row["max_offset"], offset)

        if row["sample_snippet"] is None:
            snippet = str(metadata.get("snippet", "")).strip()
            if snippet:
                row["sample_snippet"] = snippet[:180]

    ordered = sorted(
        grouped.values(),
        key=lambda item: (
            -int(item.get("chunk_count", 0)),
            str(item.get("source", "")),
            str(item.get("modality", "")),
        ),
    )
    return ordered[:limit]


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
