from __future__ import annotations

import html
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from app.interfaces.embedding import EmbeddingClient
from app.interfaces.vector_store import VectorStore
from app.interfaces.video import VideoClient
from app.interfaces.vision import VisionClient
from app.rag.chunking import chunk_text
from app.video import TemporalAggregator, VideoAnalysisAdapter, VideoFrameSampler

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}
_DOCX_CONTENT_TYPES = {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
_PPTX_CONTENT_TYPES = {"application/vnd.openxmlformats-officedocument.presentationml.presentation"}
_XLSX_CONTENT_TYPES = {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}
_HTML_CONTENT_TYPES = {"text/html", "application/xhtml+xml"}

# Plain-text and structured-text formats that can be decoded directly.
_TEXT_LIKE_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".xml",
    ".docx",
    ".pptx",
    ".xlsx",
    ".log",
    ".ini",
    ".cfg",
    ".toml",
    ".html",
    ".htm",
}


@dataclass(frozen=True)
class IngestionSummary:
    accepted_sources: int
    indexed_chunks: int


class DocumentIngestionService:
    def __init__(
        self,
        embedding_client: EmbeddingClient,
        vector_store: VectorStore,
        chunk_size: int = 700,
        chunk_overlap: int = 120,
        max_source_bytes: int = 2_000_000,
        vision_client: VisionClient | None = None,
        video_client: VideoClient | None = None,
        video_sample_fps: float = 1.0,
        video_max_frames: int = 24,
        video_latency_budget_ms: int = 1500,
        video_estimated_ms_per_frame: float = 12.0,
        video_max_window_sec: float = 120.0,
        video_max_key_events: int = 5,
        video_remote_fetch_timeout_sec: float = 20.0,
        video_max_remote_source_bytes: int = 120_000_000,
        video_require_frame_findings: bool = True,
    ) -> None:
        self._embedding_client = embedding_client
        self._vector_store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._max_source_bytes = max_source_bytes
        self._vision_client = vision_client
        self._video_client = video_client
        self._video_sample_fps = video_sample_fps
        self._video_max_frames = video_max_frames
        self._video_latency_budget_ms = video_latency_budget_ms
        self._video_estimated_ms_per_frame = video_estimated_ms_per_frame
        self._video_max_window_sec = video_max_window_sec
        self._video_max_key_events = video_max_key_events
        self._video_remote_fetch_timeout_sec = video_remote_fetch_timeout_sec
        self._video_max_remote_source_bytes = video_max_remote_source_bytes
        self._video_require_frame_findings = video_require_frame_findings

    async def ingest(self, sources: list[str], source_type: str = "mixed") -> IngestionSummary:
        ids: list[str] = []
        vectors: list[list[float]] = []
        metadata: list[dict] = []
        accepted_sources = 0

        for source in sources:
            modality = self._resolve_modality(source=source, source_type=source_type)
            try:
                text = await self._load_source_representation(source=source, source_type=source_type, modality=modality)
            except Exception:
                continue
            chunks = chunk_text(
                text=text,
                source=source,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )
            if not chunks:
                continue

            accepted_sources += 1
            vectors_for_chunks = await self._embedding_client.embed_texts([chunk.text for chunk in chunks])
            if len(vectors_for_chunks) != len(chunks):
                raise ValueError("embedding provider returned mismatched batch size")
            for chunk, vector in zip(chunks, vectors_for_chunks, strict=True):
                ids.append(chunk.id)
                vectors.append(vector)
                metadata.append(
                    {
                        "source": chunk.source,
                        "chunk_id": chunk.chunk_id,
                        "offset": chunk.offset,
                        "snippet": chunk.text,
                        "modality": modality,
                    }
                )

        if ids:
            await self._vector_store.upsert(ids=ids, vectors=vectors, metadata=metadata)

        return IngestionSummary(accepted_sources=accepted_sources, indexed_chunks=len(ids))

    async def _load_source_representation(self, source: str, source_type: str, modality: str) -> str:
        if modality == "image":
            return await self._analyze_image_source(source)
        if modality == "video":
            return await self._analyze_video_source(source)
        return self._load_source_text(source, source_type)

    def _load_source_text(self, source: str, source_type: str) -> str:
        if source_type.strip().lower() == "text" and self._should_treat_as_inline_text(source):
            return source
        if source.startswith(("http://", "https://")):
            return self._load_url(source=source, source_type=source_type)
        return self._load_local_file(source=source, source_type=source_type)

    def _should_treat_as_inline_text(self, source: str) -> bool:
        stripped = source.strip()
        if not stripped:
            return False
        if stripped.startswith(("http://", "https://", "file://")):
            return False
        if re.match(r"^[A-Za-z]:[\\/]", stripped):
            return False
        return not Path(stripped).exists()

    async def _analyze_image_source(self, source: str) -> str:
        if self._vision_client is None:
            return f"Image source: {source}"
        analysis = await self._vision_client.analyze_image(source)
        return f"Image source: {source}\nAnalysis: {analysis}"

    async def _analyze_video_source(self, source: str) -> str:
        if self._video_client is None:
            return f"Video source: {source}"

        adapter = VideoAnalysisAdapter(
            video_client=self._video_client,
            vision_client=self._vision_client,
            require_frame_level_findings=self._video_require_frame_findings,
            frame_sampler=VideoFrameSampler(
                latency_budget_ms=self._video_latency_budget_ms,
                estimated_ms_per_frame=self._video_estimated_ms_per_frame,
                max_window_sec=self._video_max_window_sec,
                remote_fetch_timeout_sec=self._video_remote_fetch_timeout_sec,
                max_remote_source_bytes=self._video_max_remote_source_bytes,
            ),
            temporal_aggregator=TemporalAggregator(max_key_events=self._video_max_key_events),
        )
        analysis = await adapter.analyze(
            video_uri=source,
            prompt="Extract retrieval-grounded timeline events from this video.",
            sample_fps=self._video_sample_fps,
            max_frames=self._video_max_frames,
        )
        events = " | ".join(analysis.key_events[: self._video_max_key_events])
        return (
            f"Video source: {source}\n"
            f"Summary: {analysis.summary}\n"
            f"Key events: {events}\n"
            f"Processed frames: {analysis.processed_frames}\n"
            f"Confidence: {analysis.confidence:.2f}"
        )

    def _resolve_modality(self, source: str, source_type: str) -> str:
        normalized_source_type = source_type.strip().lower()
        if normalized_source_type in {"image", "video"}:
            return normalized_source_type

        parsed = urlparse(source)
        candidate_path = parsed.path if parsed.path else source
        extension = Path(candidate_path).suffix.lower()

        if extension in _IMAGE_EXTENSIONS:
            return "image"
        if extension in _VIDEO_EXTENSIONS:
            return "video"
        return "text"

    def _load_local_file(self, source: str, source_type: str) -> str:
        path_value = source
        if source.startswith("file://"):
            parsed = urlparse(source)
            path_value = unquote(parsed.path)
            if re.match(r"^/[A-Za-z]:/", path_value):
                path_value = path_value[1:]

        path = Path(path_value)
        data = path.read_bytes()
        self._validate_size(data)
        return self._decode_source_bytes(
            data,
            source_type=source_type,
            extension=path.suffix.lower(),
            content_type=None,
        )

    def _load_url(self, source: str, source_type: str) -> str:
        request = Request(source, headers={"User-Agent": "MMAA-Ingest/0.1"})
        with urlopen(request, timeout=10) as response:
            data = response.read(self._max_source_bytes + 1)
            content_type = response.headers.get("Content-Type", "")
        self._validate_size(data)
        parsed = urlparse(source)
        extension = Path(parsed.path).suffix.lower()
        return self._decode_source_bytes(
            data,
            source_type=source_type,
            extension=extension,
            content_type=content_type,
        )

    def _validate_size(self, data: bytes) -> None:
        if len(data) > self._max_source_bytes:
            raise ValueError("source exceeds max allowed size")

    def _decode_source_bytes(
        self,
        data: bytes,
        source_type: str,
        extension: str,
        content_type: str | None,
    ) -> str:
        source_hint = source_type.strip().lower()
        content_type_main = (content_type or "").split(";", maxsplit=1)[0].strip().lower()
        extraction_key = self._resolve_extractor_key(source_hint, extension, content_type_main)

        if extraction_key == "pdf":
            return self._extract_pdf_text(data)
        if extraction_key == "docx":
            return self._extract_docx_text(data)
        if extraction_key == "pptx":
            return self._extract_pptx_text(data)
        if extraction_key == "xlsx":
            return self._extract_xlsx_text(data)
        if extraction_key == "html":
            return self._extract_html_text(data)
        return self._extract_text_like(data)

    def _resolve_extractor_key(self, source_type: str, extension: str, content_type: str) -> str:
        if source_type == "pdf":
            return "pdf"
        if source_type in {"docx", "word"}:
            return "docx"
        if source_type in {"pptx", "powerpoint", "presentation"}:
            return "pptx"
        if source_type in {"xlsx", "excel", "spreadsheet"}:
            return "xlsx"
        if source_type in {"md", "markdown"}:
            return "text"
        if source_type == "html":
            return "html"

        if extension == ".pdf" or content_type == "application/pdf":
            return "pdf"
        if extension == ".docx" or content_type in _DOCX_CONTENT_TYPES:
            return "docx"
        if extension == ".pptx" or content_type in _PPTX_CONTENT_TYPES:
            return "pptx"
        if extension == ".xlsx" or content_type in _XLSX_CONTENT_TYPES:
            return "xlsx"
        if extension in {".html", ".htm"} or content_type in _HTML_CONTENT_TYPES:
            return "html"
        if extension in _TEXT_LIKE_EXTENSIONS:
            return "text"
        return "text"

    def _extract_pdf_text(self, data: bytes) -> str:
        extracted = self._extract_pdf_with_pypdf(data)
        if extracted:
            return extracted

        # Fallback for pseudo-pdf fixture files that are plain text.
        for encoding in ("utf-8", "latin-1"):
            try:
                decoded = self._normalize_whitespace(data.decode(encoding))
            except UnicodeDecodeError:
                continue
            if self._looks_like_human_text(decoded):
                return decoded
        return ""

    def _extract_pdf_with_pypdf(self, data: bytes) -> str:
        try:
            from pypdf import PdfReader
        except Exception:
            return ""

        try:
            reader = PdfReader(io.BytesIO(data))
        except Exception:
            return ""

        texts: list[str] = []
        for page in reader.pages:
            try:
                page_text = page.extract_text() or ""
            except Exception:
                continue
            if page_text.strip():
                texts.append(page_text)

        combined = self._normalize_whitespace(" ".join(texts))
        return combined if self._looks_like_human_text(combined) else ""

    def _extract_docx_text(self, data: bytes) -> str:
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                document_xml = archive.read("word/document.xml")
        except Exception:
            return ""

        try:
            root = ElementTree.fromstring(document_xml)
        except Exception:
            return ""

        words: list[str] = []
        for node in root.iter():
            if node.tag.endswith("}t") and node.text:
                words.append(node.text)
        combined = self._normalize_whitespace(" ".join(words))
        return combined if self._looks_like_human_text(combined) else ""

    def _extract_pptx_text(self, data: bytes) -> str:
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                slide_names = sorted(
                    name
                    for name in archive.namelist()
                    if name.startswith("ppt/slides/slide") and name.endswith(".xml")
                )
                texts: list[str] = []
                for slide_name in slide_names:
                    slide_xml = archive.read(slide_name)
                    root = ElementTree.fromstring(slide_xml)
                    for node in root.iter():
                        if node.tag.endswith("}t") and node.text:
                            texts.append(node.text)
        except Exception:
            return ""

        combined = self._normalize_whitespace(" ".join(texts))
        return combined if self._looks_like_human_text(combined) else ""

    def _extract_xlsx_text(self, data: bytes) -> str:
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                shared_strings = self._load_xlsx_shared_strings(archive)
                sheet_names = sorted(
                    name
                    for name in archive.namelist()
                    if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
                )
                values: list[str] = []
                for sheet_name in sheet_names:
                    sheet_xml = archive.read(sheet_name)
                    root = ElementTree.fromstring(sheet_xml)
                    values.extend(self._extract_xlsx_sheet_values(root, shared_strings))
        except Exception:
            return ""

        combined = self._normalize_whitespace(" ".join(values))
        return combined if self._looks_like_human_text(combined) else ""

    def _load_xlsx_shared_strings(self, archive: zipfile.ZipFile) -> list[str]:
        try:
            payload = archive.read("xl/sharedStrings.xml")
        except KeyError:
            return []
        except Exception:
            return []

        try:
            root = ElementTree.fromstring(payload)
        except Exception:
            return []

        items: list[str] = []
        for item in root.iter():
            if not item.tag.endswith("}si"):
                continue
            fragments: list[str] = []
            for node in item.iter():
                if node.tag.endswith("}t") and node.text:
                    fragments.append(node.text)
            items.append("".join(fragments))
        return items

    def _extract_xlsx_sheet_values(self, root: ElementTree.Element, shared_strings: list[str]) -> list[str]:
        values: list[str] = []
        for cell in root.iter():
            if not cell.tag.endswith("}c"):
                continue
            cell_type = cell.attrib.get("t", "")
            if cell_type == "s":
                value_node = self._first_child_by_suffix(cell, "}v")
                if value_node is None or not value_node.text:
                    continue
                try:
                    index = int(value_node.text.strip())
                except ValueError:
                    continue
                if 0 <= index < len(shared_strings):
                    value = shared_strings[index].strip()
                    if value:
                        values.append(value)
                continue

            if cell_type == "inlineStr":
                inline_fragments = [node.text for node in cell.iter() if node.tag.endswith("}t") and node.text]
                inline_value = "".join(inline_fragments).strip()
                if inline_value:
                    values.append(inline_value)
                continue

            value_node = self._first_child_by_suffix(cell, "}v")
            if value_node is not None and value_node.text:
                raw = value_node.text.strip()
                if raw:
                    values.append(raw)
        return values

    @staticmethod
    def _first_child_by_suffix(node: ElementTree.Element, suffix: str) -> ElementTree.Element | None:
        for child in node:
            if child.tag.endswith(suffix):
                return child
        return None

    def _extract_html_text(self, data: bytes) -> str:
        decoded = self._decode_raw_bytes(data)
        if not decoded:
            return ""
        no_scripts = re.sub(r"(?is)<(script|style)\b[^>]*>.*?</\1>", " ", decoded)
        no_tags = re.sub(r"(?s)<[^>]+>", " ", no_scripts)
        combined = self._normalize_whitespace(html.unescape(no_tags))
        return combined if self._looks_like_human_text(combined) else ""

    def _extract_text_like(self, data: bytes) -> str:
        decoded = self._decode_raw_bytes(data)
        if not decoded:
            return ""
        combined = self._normalize_whitespace(decoded)
        return combined if self._looks_like_human_text(combined) else ""

    @staticmethod
    def _decode_raw_bytes(data: bytes) -> str:
        for encoding in ("utf-8", "utf-16", "latin-1"):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="ignore")

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _looks_like_human_text(text: str) -> bool:
        if not text:
            return False

        length = len(text)
        printable = sum(1 for char in text if char.isprintable() or char in "\n\r\t")
        word_like = sum(
            1 for char in text if char.isalnum() or char.isspace() or char in ".,;:!?()[]{}'\"-_/@"
        )
        alpha = sum(1 for char in text if char.isalpha())

        printable_ratio = printable / max(1, length)
        word_like_ratio = word_like / max(1, length)
        alpha_ratio = alpha / max(1, length)

        if length < 20:
            return alpha >= 1
        return printable_ratio >= 0.95 and word_like_ratio >= 0.65 and alpha_ratio >= 0.2
