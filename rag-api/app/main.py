"""Multitenant RAG API: embed via TEI, store/search in Qdrant with `tenant_id` payload filter."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, List, Literal, Optional

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.settings import settings

logger = logging.getLogger("rag_api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Greenbrain RAG API", version="0.1.0")

_qdrant: Optional[QdrantClient] = None


def get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(
            url=settings.qdrant_url,
            timeout=settings.qdrant_timeout_s,
        )
    return _qdrant


def verify_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
    expected = (settings.rag_api_key or "").strip()
    if not expected:
        return
    if (x_api_key or "").strip() != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


class UpsertPoint(BaseModel):
    id: Optional[str] = None
    text: str = Field(..., min_length=1)
    role: Literal["user", "assistant", "system"] = "user"
    conversation_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpsertRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1, max_length=256)
    points: List[UpsertPoint] = Field(..., min_length=1, max_length=64)


class QueryRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1, max_length=256)
    query: str = Field(..., min_length=1)
    limit: int = Field(default=5, ge=1, le=50)
    score_threshold: Optional[float] = Field(default=None)


class QueryHit(BaseModel):
    id: str
    score: float
    text: str
    role: str
    payload: dict[str, Any]


class QueryResponse(BaseModel):
    hits: List[QueryHit]


def _tenant_filter(tenant_id: str) -> qm.Filter:
    return qm.Filter(
        must=[
            qm.FieldCondition(
                key="tenant_id",
                match=qm.MatchValue(value=tenant_id),
            )
        ]
    )


async def _embed_texts(client: httpx.AsyncClient, texts: List[str]) -> List[List[float]]:
    base = settings.embeddings_base_url.rstrip("/")
    url = f"{base}/embed"
    last_err: Optional[Exception] = None
    for attempt in range(8):
        try:
            r = await client.post(
                url,
                json={"inputs": texts},
                timeout=httpx.Timeout(settings.embed_timeout_s),
            )
            r.raise_for_status()
            data = r.json()
            break
        except (httpx.HTTPError, ValueError) as e:
            last_err = e
            logger.warning("embed attempt %s failed: %s", attempt + 1, e)
            import asyncio

            await asyncio.sleep(min(2.0 * (attempt + 1), 15.0))
    else:
        raise HTTPException(status_code=503, detail=f"Embedding service unavailable: {last_err}")

    # TEI returns either a single vector or a list of vectors depending on batching/version.
    if texts and isinstance(data, list) and data and isinstance(data[0], (int, float)):
        return [data]  # type: ignore[list-item]
    if not isinstance(data, list):
        raise HTTPException(status_code=502, detail="Unexpected embedding response shape")
    out: List[List[float]] = []
    for row in data:
        if isinstance(row, list) and row and isinstance(row[0], (int, float)):
            out.append([float(x) for x in row])
        else:
            raise HTTPException(status_code=502, detail="Unexpected embedding row shape")
    if len(out) != len(texts):
        raise HTTPException(
            status_code=502,
            detail=f"Embedding count mismatch: got {len(out)}, expected {len(texts)}",
        )
    return out


@app.on_event("startup")
def _startup() -> None:
    qc = get_qdrant()
    name = settings.rag_collection_name
    cols = qc.get_collections().collections
    names = {c.name for c in cols}
    if name not in names:
        logger.info("Creating Qdrant collection %s (dim=%s)", name, settings.vector_size)
        qc.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=settings.vector_size, distance=qm.Distance.COSINE),
        )


@app.get("/health")
def health() -> dict[str, str]:
    get_qdrant().get_collections()
    return {"status": "ok"}


@app.post("/v1/points/upsert", dependencies=[Depends(verify_api_key)])
async def upsert_points(body: UpsertRequest) -> dict[str, Any]:
    qc = get_qdrant()
    texts = [p.text for p in body.points]
    async with httpx.AsyncClient() as http:
        vectors = await _embed_texts(http, texts)

    batch: List[qm.PointStruct] = []
    now = datetime.now(timezone.utc).isoformat()
    for p, vec in zip(body.points, vectors, strict=True):
        pid = p.id or str(uuid.uuid4())
        payload: dict[str, Any] = {
            "tenant_id": body.tenant_id,
            "role": p.role,
            "text": p.text,
            "created_at": now,
        }
        if p.conversation_id:
            payload["conversation_id"] = p.conversation_id
        if p.metadata:
            payload["metadata"] = p.metadata
        batch.append(
            qm.PointStruct(
                id=pid,
                vector=vec,
                payload=payload,
            )
        )

    qc.upsert(collection_name=settings.rag_collection_name, points=batch, wait=True)
    return {"upserted": len(batch)}


@app.post("/v1/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
async def query_points(body: QueryRequest) -> QueryResponse:
    qc = get_qdrant()
    async with httpx.AsyncClient() as http:
        vectors = await _embed_texts(http, [body.query])
    vec = vectors[0]

    res = qc.search(
        collection_name=settings.rag_collection_name,
        query_vector=vec,
        limit=body.limit,
        score_threshold=body.score_threshold,
        query_filter=_tenant_filter(body.tenant_id),
        with_payload=True,
    )

    hits: List[QueryHit] = []
    for r in res:
        pl = r.payload or {}
        text = str(pl.get("text", ""))
        role = str(pl.get("role", "user"))
        hits.append(
            QueryHit(
                id=str(r.id),
                score=float(r.score),
                text=text,
                role=role,
                payload={k: v for k, v in pl.items() if k not in ("text",)},
            )
        )
    return QueryResponse(hits=hits)
