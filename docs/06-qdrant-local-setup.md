# Qdrant Local Container Setup

This repo supports a pluggable vector-store mode where Qdrant can run as primary and pgvector/in-memory remains fallback.

## 1) Start Qdrant

```powershell
docker compose -f deployment/docker-compose.qdrant.yml up -d
```

## 2) Verify Qdrant is healthy

```powershell
Invoke-RestMethod -Method Get -Uri http://localhost:6333/collections
```

## 3) Configure this app to use Qdrant primary + fallback coexistence

Set environment variables:

```powershell
$env:MMAA_RAG_VECTOR_STORE_PROVIDER = "qdrant"
$env:MMAA_QDRANT_URL = "http://localhost:6333"
$env:MMAA_QDRANT_COLLECTION_NAME = "rag_chunks"
$env:MMAA_RAG_VECTOR_STORE_MIRROR_WRITES = "true"
```

Optional:

```powershell
$env:MMAA_QDRANT_API_KEY = "<your-key-if-needed>"
$env:MMAA_QDRANT_PREFER_GRPC = "false"
$env:MMAA_QDRANT_TIMEOUT_SEC = "10"
```

## 4) Stop Qdrant

```powershell
docker compose -f deployment/docker-compose.qdrant.yml down
```

To remove persistent data volume too:

```powershell
docker compose -f deployment/docker-compose.qdrant.yml down -v
```
