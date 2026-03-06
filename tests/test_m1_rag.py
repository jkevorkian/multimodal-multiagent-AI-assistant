import asyncio
import zipfile

import pytest
from fastapi.testclient import TestClient

from app.core.dependencies import create_test_container, get_container
from app.main import create_app
from app.rag.chunking import chunk_text
from app.rag.embeddings import build_embedding_client
from app.rag.retriever import TextRAGRetriever


@pytest.fixture()
def m1_client() -> tuple[TestClient, object]:
    app = create_app()
    container = create_test_container()
    app.dependency_overrides[get_container] = lambda: container
    with TestClient(app) as client:
        yield client, container
    app.dependency_overrides.clear()


def test_chunking_is_deterministic() -> None:
    text = "Alpha beta gamma delta " * 30
    first = chunk_text(text=text, source="doc-a", chunk_size=80, chunk_overlap=20)
    second = chunk_text(text=text, source="doc-a", chunk_size=80, chunk_overlap=20)

    assert len(first) == len(second)
    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert [chunk.offset for chunk in first] == [chunk.offset for chunk in second]
    assert first[0].chunk_id == 0
    assert first[1].offset == 60


def test_pdf_ingestion_and_query_with_citations(tmp_path, m1_client) -> None:
    client, container = m1_client
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_text("The retention policy is 30 days for logs and 365 days for reports.", encoding="utf-8")

    ingest_response = client.post(
        "/ingest/documents",
        json={"sources": [pdf_path.as_uri()], "source_type": "pdf"},
    )
    ingest_payload = ingest_response.json()

    assert ingest_response.status_code == 200
    assert ingest_payload["accepted_sources"] == 1
    assert ingest_payload["indexed_chunks"] > 0

    retrieval_payload = client.post("/query", json={"query": "What is the retention policy?", "top_k": 3}).json()
    assert retrieval_payload["citations"]
    assert any("sample.pdf" in citation for citation in retrieval_payload["citations"])

    keyword_hits = asyncio.run(container.vector_store.keyword_search("retention policy", top_k=1))
    assert keyword_hits
    metadata = keyword_hits[0]["metadata"]
    assert {"source", "chunk_id", "offset", "snippet"}.issubset(metadata.keys())


def test_binary_like_pdf_is_not_indexed_as_text(tmp_path, m1_client) -> None:
    client, _ = m1_client
    pdf_path = tmp_path / "binary.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n" + bytes(range(256)) * 20)

    ingest_response = client.post(
        "/ingest/documents",
        json={"sources": [pdf_path.as_uri()], "source_type": "pdf"},
    )
    payload = ingest_response.json()
    assert ingest_response.status_code == 200
    assert payload["accepted_sources"] == 0
    assert payload["indexed_chunks"] == 0


def test_url_ingestion_via_mocked_fetch(monkeypatch, m1_client) -> None:
    client, _ = m1_client

    class _FakeResponse:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload
            self.headers = {"Content-Type": "text/plain"}

        def read(self, _max_bytes: int | None = None) -> bytes:
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def _fake_urlopen(_request, timeout: int = 10):  # noqa: ARG001
        return _FakeResponse(b"Multi-agent orchestration uses role transitions and shared state.")

    monkeypatch.setattr("app.rag.ingestion.urlopen", _fake_urlopen)

    ingest_response = client.post(
        "/ingest/documents",
        json={"sources": ["https://example.com/guide"], "source_type": "url"},
    )
    assert ingest_response.status_code == 200
    assert ingest_response.json()["accepted_sources"] == 1

    query_payload = client.post("/query", json={"query": "What does orchestration use?", "top_k": 2}).json()
    assert query_payload["citations"]
    assert any("example.com" in citation for citation in query_payload["citations"])


def test_retrieval_smoke_with_known_corpus(tmp_path, m1_client) -> None:
    client, _ = m1_client
    text_path = tmp_path / "facts.txt"
    text_path.write_text("The blue whale is a mammal and the largest animal known.", encoding="utf-8")

    ingest_response = client.post(
        "/ingest/documents",
        json={"sources": [str(text_path)], "source_type": "text"},
    )
    assert ingest_response.status_code == 200

    payload = client.post("/query", json={"query": "What type of animal is the blue whale?"}).json()
    assert payload["citations"]
    assert "mammal" in payload["answer"].lower()


def test_inline_text_source_is_ingested_when_source_type_text(m1_client) -> None:
    client, _ = m1_client
    inline = "Joaquin Kevorkian loves mexican food, especially tacos."
    ingest_response = client.post(
        "/ingest/documents",
        json={"sources": [inline], "source_type": "text"},
    )
    assert ingest_response.status_code == 200
    assert ingest_response.json()["accepted_sources"] == 1

    query_payload = client.post("/query", json={"query": "What food does Joaquin like?", "top_k": 3}).json()
    assert query_payload["citations"]


def test_markdown_file_ingestion(tmp_path, m1_client) -> None:
    client, _ = m1_client
    md_path = tmp_path / "guide.md"
    md_path.write_text(
        "# Retention\n\nThe retention policy is **30 days** for logs and 365 for reports.",
        encoding="utf-8",
    )

    ingest_response = client.post(
        "/ingest/documents",
        json={"sources": [md_path.as_uri()], "source_type": "markdown"},
    )
    assert ingest_response.status_code == 200
    assert ingest_response.json()["accepted_sources"] == 1

    query_payload = client.post("/query", json={"query": "How long are logs retained?", "top_k": 3}).json()
    assert query_payload["citations"]
    assert any("guide.md" in citation for citation in query_payload["citations"])
    assert "30 days" in query_payload["answer"].lower()


def test_docx_file_ingestion(tmp_path, m1_client) -> None:
    client, _ = m1_client
    docx_path = tmp_path / "policy.docx"
    _create_minimal_docx(
        docx_path,
        text="The retention policy is 30 days for logs and 365 days for reports.",
    )

    ingest_response = client.post(
        "/ingest/documents",
        json={"sources": [docx_path.as_uri()], "source_type": "docx"},
    )
    assert ingest_response.status_code == 200
    assert ingest_response.json()["accepted_sources"] == 1

    query_payload = client.post("/query", json={"query": "What is the retention policy?", "top_k": 3}).json()
    assert query_payload["citations"]
    assert any("policy.docx" in citation for citation in query_payload["citations"])
    assert "30 days" in query_payload["answer"].lower()


def test_pptx_file_ingestion(tmp_path, m1_client) -> None:
    client, _ = m1_client
    pptx_path = tmp_path / "deck.pptx"
    _create_minimal_pptx(
        pptx_path,
        text="Milestone M3 includes LangGraph orchestration and multimodal processing.",
    )

    ingest_response = client.post(
        "/ingest/documents",
        json={"sources": [pptx_path.as_uri()], "source_type": "pptx"},
    )
    assert ingest_response.status_code == 200
    assert ingest_response.json()["accepted_sources"] == 1

    query_payload = client.post("/query", json={"query": "What does M3 include?", "top_k": 3}).json()
    assert query_payload["citations"]
    assert any("deck.pptx" in citation for citation in query_payload["citations"])
    assert "langgraph" in query_payload["answer"].lower()


def test_xlsx_file_ingestion(tmp_path, m1_client) -> None:
    client, _ = m1_client
    xlsx_path = tmp_path / "table.xlsx"
    _create_minimal_xlsx(
        xlsx_path,
        shared_values=["Joaquin", "FavoriteFood", "Tacos"],
    )

    ingest_response = client.post(
        "/ingest/documents",
        json={"sources": [xlsx_path.as_uri()], "source_type": "xlsx"},
    )
    assert ingest_response.status_code == 200
    assert ingest_response.json()["accepted_sources"] == 1

    query_payload = client.post("/query", json={"query": "What is Joaquin favorite food?", "top_k": 3}).json()
    assert query_payload["citations"]
    assert any("table.xlsx" in citation for citation in query_payload["citations"])
    assert "tacos" in query_payload["answer"].lower()


def test_embedding_provider_auto_falls_back_to_deterministic() -> None:
    selection = build_embedding_client(provider="auto", api_key=None, deterministic_dimensions=48)
    assert selection.provider_name == "deterministic"
    vector = asyncio.run(selection.client.embed_text("semantic retrieval baseline"))
    assert len(vector) == 48
    assert any(abs(value) > 0 for value in vector)


def test_hybrid_retrieval_uses_dense_and_lexical_branches() -> None:
    class _Store:
        async def search(self, vector: list[float], top_k: int) -> list[dict]:  # noqa: ARG002
            return [
                {
                    "id": "dense-1",
                    "metadata": {
                        "source": "dense_doc",
                        "chunk_id": 1,
                        "offset": 10,
                        "snippet": "orchestration topology and asynchronous queues",
                    },
                    "score": 0.9,
                }
            ][:top_k]

        async def keyword_search(self, query: str, top_k: int) -> list[dict]:  # noqa: ARG002
            return [
                {
                    "id": "lexical-1",
                    "metadata": {
                        "source": "lexical_doc",
                        "chunk_id": 2,
                        "offset": 20,
                        "snippet": "policy retention 365 days with auditing",
                    },
                    "score": 2.1,
                }
            ][:top_k]

    embedding_client = build_embedding_client(provider="deterministic", deterministic_dimensions=32).client
    retriever = TextRAGRetriever(
        embedding_client=embedding_client,
        vector_store=_Store(),  # type: ignore[arg-type]
        dense_top_k=5,
        lexical_top_k=5,
        rerank_pool_size=10,
    )
    hits = asyncio.run(retriever.retrieve("retention policy", top_k=2))
    sources = {hit["source"] for hit in hits}
    assert {"dense_doc", "lexical_doc"}.issubset(sources)


def test_reranker_hook_is_applied() -> None:
    class _Store:
        async def search(self, vector: list[float], top_k: int) -> list[dict]:  # noqa: ARG002
            return [
                {
                    "id": "dense-first",
                    "metadata": {
                        "source": "dense_doc",
                        "chunk_id": 0,
                        "offset": 0,
                        "snippet": "dense branch candidate",
                    },
                    "score": 1.0,
                }
            ][:top_k]

        async def keyword_search(self, query: str, top_k: int) -> list[dict]:  # noqa: ARG002
            return [
                {
                    "id": "lexical-second",
                    "metadata": {
                        "source": "lexical_doc",
                        "chunk_id": 1,
                        "offset": 1,
                        "snippet": "lexical branch candidate",
                    },
                    "score": 1.0,
                }
            ][:top_k]

    class _Reranker:
        def __init__(self) -> None:
            self.called = False

        async def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:  # noqa: ARG002
            self.called = True
            ordered = list(reversed(candidates))
            return ordered[:top_k]

    reranker = _Reranker()
    retriever = TextRAGRetriever(
        embedding_client=build_embedding_client(provider="deterministic", deterministic_dimensions=16).client,
        vector_store=_Store(),  # type: ignore[arg-type]
        reranker=reranker,  # type: ignore[arg-type]
        dense_top_k=4,
        lexical_top_k=4,
        rerank_pool_size=6,
    )

    hits = asyncio.run(retriever.retrieve("candidate", top_k=2))
    assert reranker.called is True
    assert hits
    assert hits[0]["source"] == "lexical_doc"


def _create_minimal_docx(path, text: str) -> None:
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>"
        f"{text}"
        "</w:t></w:r></w:p></w:body></w:document>"
    )
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", document_xml)


def _create_minimal_pptx(path, text: str) -> None:
    slide_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        "<p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>"
        f"{text}"
        "</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>"
    )
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("ppt/slides/slide1.xml", slide_xml)


def _create_minimal_xlsx(path, shared_values: list[str]) -> None:
    shared_items = "".join(f"<si><t>{value}</t></si>" for value in shared_values)
    shared_strings_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"{shared_items}"
        "</sst>"
    )
    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<sheetData>"
        '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c><c r="C1" t="s"><v>2</v></c></row>'
        "</sheetData>"
        "</worksheet>"
    )
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/sharedStrings.xml", shared_strings_xml)
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)
