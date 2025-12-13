import io
import os, time, uuid
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from pydantic import field_validator
from pydantic_settings import BaseSettings
import httpx
from fastapi import UploadFile, File
from pypdf import PdfReader


# ------------------ Settings ------------------
class Settings(BaseSettings):
    OLLAMA_URL: str
    QDRANT_URL: str
    EMBED_MODEL: str = "qwen3-embedding:8b"
    EMBED_DIMS: int | None = None
    CHAT_MODEL: str = "mistral-small3.2:latest"
    QDRANT_COLLECTION: str = "personal_memory"
    BRAIN_API_KEY: str | None = None

    @field_validator("OLLAMA_URL","QDRANT_URL","EMBED_MODEL","QDRANT_COLLECTION", mode="before")
    @classmethod
    def clean_env(cls, v: str) -> str:
        return v.split("#", 1)[0].strip() if isinstance(v, str) else v

    class Config:
        # The brain service runs on a separate instance and already has its
        # environment provided there. Using the original `.env` file keeps local
        # development convenient without forcing prod secrets to be copied
        # between hosts.
        env_file = ".env"
        extra = "ignore"

settings = Settings()
# ------------------ App ------------------
app = FastAPI(title="Brain", version="1.0.0")

# Simple API key dependency
def check_api_key(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
):
    token = x_api_key
    if not token and authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    if settings.BRAIN_API_KEY and token != settings.BRAIN_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

def text_uuid(text: str) -> str:
    clean = " ".join(text.strip().split())
    return str(uuid.uuid5(uuid.NAMESPACE_URL, clean))

# ------------------ Models ------------------
class TextIn(BaseModel):
    text: str

class MemorizeIn(BaseModel):
    text: str
    meta: dict | None = None
    # optional: fester id
    id: str | None = None

class RetrieveIn(BaseModel):
    query: str
    limit: int = 5
    score_threshold: float | None = 0.75

class ChatIn(BaseModel):
    prompt: str
    top_k: int = 5
    score_threshold: float | None = 0.75
    system: str | None = None

class IngestTextIn(BaseModel):
    text: str | None = None
    chunks: list[str] | None = None
    meta: dict | None = None
    chunk_tokens: int | None = Field(
        default=None,
        description="Preferred target chunk size in tokens (guidance only).",
    )
    overlap_tokens: int | None = Field(
        default=None,
        description="Optional overlap in tokens for fallback chunking.",
    )
    collection: str | None = None


def resolve_collection(name: str | None) -> str:
    # Fallback to the original default collection when no explicit value is provided
    clean = name.strip() if isinstance(name, str) else None
    return clean or settings.QDRANT_COLLECTION


def select_chunks_for_ingest(
    text: str | None,
    provided_chunks: list[str] | None,
):
    chunks = [c.strip() for c in (provided_chunks or []) if isinstance(c, str) and c.strip()]
    if chunks:
        return chunks

    if text and text.strip():
        return [text.strip()]

    return []


@app.post("/ingest/text")
async def ingest_text(inp: IngestTextIn, _: bool = Depends(check_api_key)):
    collection = resolve_collection(inp.collection)
    chunk_texts = select_chunks_for_ingest(
        inp.text,
        inp.chunks,
    )
    if not chunk_texts:
        raise HTTPException(400, "No text content to ingest")
    chunks = [(t, {"source": "text", **(inp.meta or {})}) for t in chunk_texts]
    ids = await memorize_chunks(chunks, collection=collection)
    return {"status": "ok", "chunks": len(ids), "ids": ids[:5]}

@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...),
                      chunk_tokens: int = 800,
                      overlap_tokens: int = 120,
                      collection: str | None = None,
                      _: bool = Depends(check_api_key)):
    ct = (file.content_type or "").lower()
    raw = await file.read()

    text = ""
    if ct.startswith("text/") or file.filename.lower().endswith((".txt",".md",".log",".csv")):
        text = raw.decode("utf-8", errors="ignore")
        meta = {"source": "file", "filename": file.filename, "ctype": ct}
    elif ct in ("application/pdf","application/x-pdf") or file.filename.lower().endswith(".pdf"):
        # PDF lesen
        reader = PdfReader(io.BytesIO(raw))
        pages = [p.extract_text() or "" for p in reader.pages]
        text = "\n".join(pages)
        meta = {"source": "pdf", "filename": file.filename, "pages": len(pages)}
    else:
        raise HTTPException(415, f"Unsupported content-type: {ct} (filename={file.filename})")

    chunk_texts = select_chunks_for_ingest(text, None)
    chunks = [(t, meta) for t in chunk_texts]
    if not chunks:
        raise HTTPException(400, "No text extracted/chunked")
    target_collection = resolve_collection(collection)
    ids = await memorize_chunks(chunks, collection=target_collection)
    return {"status": "ok", "chunks": len(ids), "preview": ids[:5]}

@app.get("/health/extended")
async def health_extended():
    # prüfe ollama + qdrant kurz an
    o, q = "down", "down"
    try:
        r = await client.get(f"{settings.OLLAMA_URL}/api/version")
        if r.status_code == 200: o = "up"
    except Exception: pass
    try:
        r = await client.get(f"{settings.QDRANT_URL}/collections/{settings.QDRANT_COLLECTION}")
        if r.status_code in (200,404): q = "up"  # 404→existiert nicht, aber service erreichbar
    except Exception: pass
    return {
        "status": "ok",
        "services": {"ollama": o, "qdrant": q},
        "ollama": settings.OLLAMA_URL,
        "qdrant": settings.QDRANT_URL,
        "collection": settings.QDRANT_COLLECTION,
        "embed_model": settings.EMBED_MODEL,
        "embed_dims": settings.EMBED_DIMS,
        "version": "1.1.0"
    }

# ------------------ Helpers ------------------
TIMEOUT = httpx.Timeout(20.0, connect=5.0)
client = httpx.AsyncClient(timeout=TIMEOUT)

async def ensure_collection(collection: str | None = None):
    # 1) Ziel-Dimension bestimmen
    dims = settings.EMBED_DIMS
    if not dims:
        dims = await detect_embed_dims(settings.EMBED_MODEL)

    collection_name = resolve_collection(collection)

    # 2) Existenz + Dimension prüfen
    info = await client.get(f"{settings.QDRANT_URL}/collections/{collection_name}")
    if info.status_code == 200:
        js = info.json()
        current = (js.get("result") or {}).get("config", {}).get("params", {}).get("vectors", {})
        current_size = (current.get("size") if isinstance(current, dict) else None)
        if current_size and int(current_size) != int(dims):
            # Harte Entscheidung: failen (sauberer) oder (optional) droppen und neu anlegen
            raise RuntimeError(f"Collection '{collection_name}' has dim {current_size}, "
                               f"but embed model returns {dims}. Bitte Collection neu anlegen/leer machen.")
        return

    # 3) Anlegen
    schema = {
        "vectors": {"size": int(dims), "distance": "Cosine"},
        "optimizers_config": {"default_segment_number": 2}
    }
    cr = await client.put(
        f"{settings.QDRANT_URL}/collections/{collection_name}", json=schema
    )
    if cr.status_code not in (200, 201):
        raise RuntimeError(f"Create collection failed: {cr.text}")

async def detect_embed_dims(model: str) -> int:
    # schickt einen Mini-Text an /api/embeddings und liest die Länge des Vektors
    r = await client.post(f"{settings.OLLAMA_URL}/api/embeddings",
                          json={"model": model, "prompt": "ping"})
    if r.status_code != 200:
        raise RuntimeError(f"Embed ping failed: {r.text}")
    data = r.json()
    vec = data.get("embedding") or (data.get("embeddings") or [None])[0]
    if not vec:
        raise RuntimeError("No embedding returned on ping")
    return len(vec)

@app.on_event("startup")
async def on_startup():
    await ensure_collection()

@app.on_event("shutdown")
async def on_shutdown():
    await client.aclose()

# --- replace embed() ---
async def embed(text: str) -> list[float]:
    model = (settings.EMBED_MODEL or "nomic-embed-text").strip()

    async def _post(payload: dict) -> dict:
        r = await client.post(f"{settings.OLLAMA_URL}/api/embeddings", json=payload)
        if r.status_code != 200:
            raise HTTPException(502, f"Ollama embeddings failed: {r.text}")
        return r.json()

    # 1) Versuch: prompt (kompatibel zu deiner getesteten API)
    data = await _post({"model": model, "prompt": text})

    # Response-Varianten akzeptieren
    vec = data.get("embedding")
    if not vec and "embeddings" in data and isinstance(data["embeddings"], list) and data["embeddings"]:
        vec = data["embeddings"][0]

    # 2) Fallback: falls prompt nichts liefert, mit input erneut
    if not vec:
        data = await _post({"model": model, "input": text})
        vec = data.get("embedding")
        if not vec and "embeddings" in data and isinstance(data["embeddings"], list) and data["embeddings"]:
            vec = data["embeddings"][0]

    if not vec:
        raise HTTPException(500, "No embedding returned from Ollama")

    # Dimensions-Check nur warnend (Modellwechsel kann andere Dim haben)
    try:
        expected = int(settings.EMBED_DIMS) if settings.EMBED_DIMS else None
        if expected and len(vec) != expected:
            # nicht hart failen, nur tolerieren
            pass
    except Exception:
        pass

    return vec

async def qdrant_upsert(points: list[dict], *, collection: str | None = None):
    collection_name = resolve_collection(collection)
    await ensure_collection(collection_name)
    r = await client.put(
        f"{settings.QDRANT_URL}/collections/{collection_name}/points?wait=true",
        json={"points": points}
    )
    if r.status_code not in (200, 202):
        raise HTTPException(502, f"Qdrant upsert failed: {r.text}")

async def qdrant_search(vec: list[float], limit: int, score_threshold: float | None, *, collection: str | None = None):
    body = {
        "vector": vec,
        "limit": limit,
        "with_payload": True,
        "with_vectors": False,
    }
    if score_threshold is not None:
        body["score_threshold"] = score_threshold
    collection_name = resolve_collection(collection)
    await ensure_collection(collection_name)
    r = await client.post(
        f"{settings.QDRANT_URL}/collections/{collection_name}/points/search",
        json=body
    )
    if r.status_code != 200:
        raise HTTPException(502, f"Qdrant search failed: {r.text}")
    return r.json().get("result", [])

async def ollama_chat(system: str | None, user: str) -> str:
    msg = []
    if system:
        msg.append({"role": "system", "content": system})
    msg.append({"role": "user", "content": user})
    jr = {"model": settings.CHAT_MODEL, "messages": msg, "stream": False}
    r = await client.post(f"{settings.OLLAMA_URL}/api/chat", json=jr)
    if r.status_code != 200:
        raise HTTPException(502, f"Ollama chat failed: {r.text}")
    data = r.json()
    # Ollama returns single message for non-stream
    return data.get("message", {}).get("content", "")

# ------------------ Endpoints ------------------
@app.get("/health")
async def health():
    return {"status": "ok", "ollama": settings.OLLAMA_URL, "qdrant": settings.QDRANT_URL}

@app.post("/memorize")
async def memorize(inp: MemorizeIn, _: bool = Depends(check_api_key)):
    vec = await embed(inp.text)
    point_id = inp.id or str(uuid.uuid4())
    payload = {
        "id": point_id,
        "vector": vec,
        "payload": {
            "text": inp.text,
            "meta": inp.meta or {},
            "ts": int(time.time())
        }
    }
    await qdrant_upsert([payload])
    return {"status": "ok", "id": point_id}

@app.post("/retrieve")
async def retrieve(inp: RetrieveIn, _: bool = Depends(check_api_key)):
    vec = await embed(inp.query)
    hits = await qdrant_search(vec, limit=inp.limit, score_threshold=inp.score_threshold)
    # Format komprimiert zurückgeben
    return {
        "hits": [
            {"id": h.get("id"), "score": h.get("score"), "text": h["payload"].get("text"), "meta": h["payload"].get("meta")}
            for h in hits
        ],
        "count": len(hits)
    }

@app.post("/chat")
async def chat(inp: ChatIn, _: bool = Depends(check_api_key)):
    vec = await embed(inp.prompt)
    hits = await qdrant_search(vec, limit=inp.top_k, score_threshold=inp.score_threshold)
    memory = "\n".join(f"- {h['payload'].get('text','')}" for h in hits if h.get("payload"))
    sys = inp.system or "Nutze den folgenden Memory-Kontext, wenn er relevant ist. Antworte präzise."
    system = f"{sys}\n\nMEMORY CONTEXT:\n{memory}" if memory else sys
    answer = await ollama_chat(system=system, user=inp.prompt)
    return {"answer": answer, "memory_hits": len(hits)}

async def memorize_chunks(chunks: list[tuple[str, dict]], *, collection: str | None = None):
    points = []
    for text, meta in chunks:
        vec = await embed(text)
        pid = text_uuid(text)
        points.append({
            "id": pid,
            "vector": vec,
            "payload": {"text": text, "meta": meta or {}, "ts": int(time.time())}
        })
    # Upsert (idempotent)
    await qdrant_upsert(points, collection=collection)
    return [p["id"] for p in points]

class DeleteIn(BaseModel):
    ids: list[str | int]
    collection: str | None = None

class PurgeIn(BaseModel):
    collection: str | None = None

class ScrollByNameIn(BaseModel):
    substring: str
    limit: int = 200
    collection: str | None = None

    @field_validator("substring")
    @classmethod
    def ensure_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("substring must not be empty")
        return v

class FileIdsIn(BaseModel):
    file_id: str
    limit: int = 500
    collection: str | None = None

@app.post("/admin/delete")
async def admin_delete(inp: DeleteIn, _: bool = Depends(check_api_key)):
    # collection aus Payload auswerten, sonst Fallback auf Default
    collection_name = resolve_collection(inp.collection)
    await ensure_collection(collection_name)

    r = await client.post(
        f"{settings.QDRANT_URL}/collections/{collection_name}/points/delete?wait=true",
        json={"points": inp.ids},
    )
    if r.status_code not in (200, 202):
        raise HTTPException(502, f"Qdrant delete failed: {r.text}")
    return {"status": "ok", "deleted": inp.ids, "collection": collection_name}

@app.post("/admin/purge")
async def admin_purge(inp: PurgeIn, _: bool = Depends(check_api_key)):
    # collection aus Payload oder Default
    collection_name = resolve_collection(inp.collection)
    await ensure_collection(collection_name)

    r = await client.post(
        f"{settings.QDRANT_URL}/collections/{collection_name}/points/delete?wait=true",
        json={"filter": {"must": []}},
    )
    if r.status_code not in (200, 202):
        raise HTTPException(502, f"Qdrant purge failed: {r.text}")
    return {"status": "ok", "purged": True, "collection": collection_name}

@app.post("/admin/scroll-by-name")
async def admin_scroll_by_name(inp: ScrollByNameIn, _: bool = Depends(check_api_key)):
    collection_name = resolve_collection(inp.collection)
    await ensure_collection(collection_name)

    results: list[dict] = []
    offset = None
    substring_lower = inp.substring.lower()

    def _iter_strings(value):
        if value is None:
            return
        if isinstance(value, str):
            yield value
            return
        if isinstance(value, (int, float, bool)):
            yield str(value)
            return
        if isinstance(value, dict):
            for v in value.values():
                yield from _iter_strings(v)
            return
        if isinstance(value, (list, tuple, set)):
            for v in value:
                yield from _iter_strings(v)
            return
    while True:
        body = {"limit": min(inp.limit, 64), "with_payload": True, "offset": offset}
        r = await client.post(
            f"{settings.QDRANT_URL}/collections/{collection_name}/points/scroll",
            json=body,
        )
        if r.status_code != 200:
            raise HTTPException(502, f"Qdrant scroll failed: {r.text}")
        data = r.json().get("result") or {}
        points = data.get("points") or []
        for p in points:
            payload = p.get("payload") or {}
            file_name = str(
                payload.get("file_name")
                or payload.get("path")
                or (payload.get("meta") or {}).get("path")
                or ""
            )
            searchable_strings = []
            if file_name:
                searchable_strings.append(file_name)
            text = payload.get("text")
            if isinstance(text, str):
                searchable_strings.append(text)
            searchable_strings.extend(list(_iter_strings(payload.get("meta") or {})))

            if any(substring_lower in s.lower() for s in searchable_strings):
                results.append(p)
                if len(results) >= inp.limit:
                    return {"results": results, "count": len(results)}
        offset = data.get("next_page_offset")
        if not offset:
            break
    return {"results": results, "count": len(results)}


@app.post("/admin/ids-for-file")
async def admin_ids_for_file(inp: FileIdsIn, _: bool = Depends(check_api_key)):
    collection_name = resolve_collection(inp.collection)
    await ensure_collection(collection_name)

    ids: list[str | int] = []
    offset = None
    payload_filter = {"must": [{"key": "meta.file_id", "match": {"value": inp.file_id}}]}
    while True:
        body = {
            "limit": min(inp.limit, 64),
            "with_payload": False,
            "offset": offset,
            "filter": payload_filter,
        }
        r = await client.post(
            f"{settings.QDRANT_URL}/collections/{collection_name}/points/scroll",
            json=body,
        )
        if r.status_code != 200:
            raise HTTPException(502, f"Qdrant scroll failed: {r.text}")
        data = r.json().get("result") or {}
        points = data.get("points") or []
        ids.extend([p.get("id") for p in points if p.get("id") is not None])
        if len(ids) >= inp.limit:
            return {"ids": ids[: inp.limit], "count": len(ids)}
        offset = data.get("next_page_offset")
        if not offset:
            break
    return {"ids": ids, "count": len(ids)}
