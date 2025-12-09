
#!/usr/bin/env python3
import base64
import importlib
import json
import mimetypes
import re
import os
import posixpath
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from nextcloud_client import env_client, NextcloudClient, NextcloudError

from helpers import compute_next_review_at, utcnow_iso
from chunking import chunk_document_with_llm_fallback
from add_context import enrich_chunks_with_context


class DoclingUnavailableError(RuntimeError):
    """Raised when docling is missing at runtime."""


TEXT_FALLBACK_EXTENSIONS = {".txt", ".text"}
ODF_FALLBACK_EXTENSIONS = {".odt", ".ods", ".odp", ".odg", ".odf", ".odm"}

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    cleaned = [c for c in normalized if c.isalnum() or c in {"-", "_"}]
    slug = "".join(cleaned).strip("-_").lower()
    return slug or "doc"


def _optional_import(name: str):
    spec = importlib.util.find_spec(name)
    if spec is None:
        return None
    module = importlib.util.module_from_spec(spec)
    if module is None or spec.loader is None:
        return None
    spec.loader.exec_module(module)
    return module


def _dbg_write(target: Path, content: str):
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content or "", encoding="utf-8")


def pdftotext_extract(file_path: Path, *, timeout: Optional[int] = None) -> str:
    if not file_path.exists():
        return ""
    effective_timeout = PDFTOTEXT_TIMEOUT_S if timeout is None else timeout
    try:
        proc = subprocess.run(
            ["pdftotext", "-enc", "UTF-8", str(file_path), "-"],
            capture_output=True,
            text=True,
            timeout=effective_timeout,
            check=False,
        )
    except FileNotFoundError:
        return ""
    except subprocess.TimeoutExpired:
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout or ""


def ocr_pdf_with_tesseract(
    file_path: Path, *, dpi: Optional[int] = None, max_pages: Optional[int] = None
) -> str:
    pdf2image = _optional_import("pdf2image")
    pytesseract = _optional_import("pytesseract")
    if pdf2image is None or pytesseract is None:
        return ""
    effective_dpi = PDF_OCR_DPI if dpi is None else dpi
    max_pages = PDF_OCR_MAX_PAGES if max_pages is None else max_pages
    last_page = max_pages if max_pages and max_pages > 0 else None
    try:
        images = pdf2image.convert_from_path(
            str(file_path), dpi=effective_dpi, last_page=last_page
        )
    except Exception:
        return ""
    texts: List[str] = []
    for img in images:
        try:
            text = pytesseract.image_to_string(img)
        except Exception:
            continue
        if text and text.strip():
            texts.append(text.strip())
    return "\n\n".join(texts)


def legacy_pdf_fallback_text(file_path: Path) -> str:
    text_candidates: List[str] = []
    base_text = (pdftotext_extract(file_path) or "").strip()
    if base_text:
        text_candidates.append(base_text)
    if ENABLE_OCR:
        ocr_text = (ocr_pdf_with_tesseract(file_path) or "").strip()
        if ocr_text:
            text_candidates.append(ocr_text)
    combined = "\n\n".join(text_candidates).strip()
    return combined

# --- safe logger to avoid crashes during instrumentation ---
_log_counters = {}  # (job_id) -> count

def safe_log(conn, job_id, file_id, step, detail):
    try:
        if not DECISION_LOG_ENABLED:
            return
        if job_id:
            c = _log_counters.get(job_id, 0)
            if c >= DECISION_LOG_MAX_PER_JOB:
                return
            _log_counters[job_id] = c + 1
        log_decision(conn, job_id, file_id, step, detail[:500])  # detail begrenzen
    except Exception:
        pass


# === Config (ENV) ===
DB_PATH              = os.environ.get("DB_PATH", "DocumentDatabase/state.db")
BRAIN_URL            = os.environ.get("BRAIN_URL", "http://192.168.177.151:8080").rstrip("/")
BRAIN_API_KEY        = os.environ.get("BRAIN_API_KEY", "change-me")
BRAIN_COLLECTION     = os.environ.get("BRAIN_COLLECTION", "documents")
BRAIN_CHUNK_TOKENS   = int(os.environ.get("BRAIN_CHUNK_TOKENS", "400"))
BRAIN_OVERLAP_TOKENS = int(os.environ.get("BRAIN_OVERLAP_TOKENS", "80"))
BRAIN_REQUEST_TIMEOUT = float(os.environ.get("BRAIN_REQUEST_TIMEOUT", "120"))
OLLAMA_HOST          = os.environ.get("OLLAMA_HOST", "http://192.168.177.130:11434")
OLLAMA_MODEL         = os.environ.get("OLLAMA_MODEL", "mistral-small3.2:latest")
OLLAMA_MODEL_RELEVANCE = os.environ.get("OLLAMA_MODEL_RELEVANCE", OLLAMA_MODEL)
OLLAMA_MODEL_CHUNKING  = os.environ.get("OLLAMA_MODEL_CHUNKING", OLLAMA_MODEL)
OLLAMA_MODEL_CONTEXT   = os.environ.get("OLLAMA_MODEL_CONTEXT", OLLAMA_MODEL)
RELEVANCE_THRESHOLD  = float(os.environ.get("RELEVANCE_THRESHOLD", "0.55"))
MIN_CHARS            = int(os.environ.get("MIN_CHARS", "200"))
MAX_TEXT_CHARS       = int(os.environ.get("MAX_TEXT_CHARS", "100000"))
MAX_CHARS            = int(os.environ.get("MAX_CHARS", "4000"))
OVERLAP              = int(os.environ.get("OVERLAP", "400"))
MAX_CHUNKS           = int(os.environ.get("MAX_CHUNKS", "200"))
ENABLE_OCR           = os.environ.get("ENABLE_OCR", "0") == "1"
DEBUG                = os.environ.get("DEBUG", "0") == "1"
WORKER_DEBUG_LOGS    = os.environ.get("WORKER_DEBUG_LOGS", os.environ.get("DEBUG", "0")) == "1"
LOCK_TIMEOUT_S       = int(os.environ.get("LOCK_TIMEOUT_S", "600"))
IDLE_SLEEP_S         = float(os.environ.get("IDLE_SLEEP_S", "1.0"))
PDFTOTEXT_TIMEOUT_S  = int(os.environ.get("PDFTOTEXT_TIMEOUT_S", "60"))
PDF_OCR_MAX_PAGES    = int(os.environ.get("PDF_OCR_MAX_PAGES", "20"))
PDF_OCR_DPI          = int(os.environ.get("PDF_OCR_DPI", "300"))

DOCLING_SERVE_URL     = os.environ.get("DOCLING_SERVE_URL", "http://192.168.177.130:5001/v1/convert/file")
DOCLING_SERVE_TIMEOUT  = float(os.environ.get("DOCLING_SERVE_TIMEOUT", "300"))
NEXTCLOUD_DOC_DIR      = os.environ.get("NEXTCLOUD_DOC_DIR", "/RAGdocuments")
NEXTCLOUD_IMAGE_DIR    = os.environ.get("NEXTCLOUD_IMAGE_DIR", "/RAGimages")
NEXTCLOUD_BASE_URL     = os.environ.get("NEXTCLOUD_BASE_URL", "http://192.168.177.133:8080").rstrip("/")
NEXTCLOUD_USER         = os.environ.get("NEXTCLOUD_USER", "andreas")
NEXTCLOUD_TOKEN        = os.environ.get("TOKEN") or os.environ.get("NEXTCLOUD_TOKEN", "")

DECISION_LOG_ENABLED  = os.environ.get("DECISION_LOG_ENABLED", "1") == "1"
DECISION_LOG_MAX_PER_JOB = int(os.environ.get("DECISION_LOG_MAX_PER_JOB", "50"))

WORKER_ID          = f"{os.uname().nodename}-pid{os.getpid()}"

_NEXTCLOUD_CLIENT: NextcloudClient | None = None

def log(msg):
    ts = datetime.utcnow().isoformat(timespec='seconds') + "Z"
    print(f"[worker_id] {ts} {msg}", flush=True)


def log_debug(msg):
    if WORKER_DEBUG_LOGS:
        log(msg)


def get_nextcloud_client() -> NextcloudClient:
    global _NEXTCLOUD_CLIENT
    if _NEXTCLOUD_CLIENT is None:
        client = env_client()
        log(
            f"[nextcloud_client_ready] base_url={NEXTCLOUD_BASE_URL} user={NEXTCLOUD_USER} "
            f"token={'set' if bool(NEXTCLOUD_TOKEN) else 'missing'}"
        )
        _NEXTCLOUD_CLIENT = client
    return _NEXTCLOUD_CLIENT

def log_decision(conn, job_id, file_id, step, detail):
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS decision_log ("
            "id INTEGER PRIMARY KEY,"
            "ts TEXT NOT NULL DEFAULT (datetime('now')),"
            "job_id TEXT,"
            "file_id TEXT,"
            "step TEXT,"
            "detail TEXT)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_decision_log_ts ON decision_log(ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_decision_log_job ON decision_log(job_id)")
        conn.execute(
            "INSERT INTO decision_log (job_id, file_id, step, detail) VALUES (?,?,?,?)",
            (job_id, file_id, step, (detail or "")[:4000]),
        )
        conn.commit()
    except Exception:
        pass

def sqlite_dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

def claim_one(conn):
    """
    Claimt den ältesten fälligen queued-Job atomar.
    Rückgabe: (job_id, file_id) oder (None, None)
    Erwartet globale ENV/Consts: LOCK_TIMEOUT_S, WORKER_ID
    """
    cutoff = int(LOCK_TIMEOUT_S)

    # ISO 8601 -> 'YYYY-MM-DD HH:MM:SS' (trimmt auf 19 Zeichen, ersetzt 'T' durch Leerzeichen)
    def _norm(col):
        return f"datetime(replace(substr({col},1,19), 'T', ' '))"

    # 1) Kandidat bestimmen (geordnet nach enqueue_at, dann job_id)
    row = conn.execute(f"""
        SELECT jobs.job_id, jobs.file_id
          FROM jobs
          JOIN files ON files.id = jobs.file_id
         WHERE jobs.status='queued'
           AND (jobs.run_not_before IS NULL OR {_norm('jobs.run_not_before')} <= datetime('now'))
           AND (jobs.locked_at IS NULL OR {_norm('jobs.locked_at')} < datetime('now', ?))
         ORDER BY files.priority DESC,
                  {_norm('files.next_review_at')},
                  {_norm('jobs.enqueue_at')},
                  jobs.job_id
         LIMIT 1
    """, (f"-{cutoff} seconds",)).fetchone()

    if not row:
        return None, None

    # Row-factory-agnostisch auslesen
    if isinstance(row, dict):
        job_id, file_id = row.get("job_id"), row.get("file_id")
    else:
        try:
            job_id, file_id = row["job_id"], row["file_id"]
        except Exception:
            job_id, file_id = row[0], row[1]

    # 2) Atomar claimen mit identischen Bedingungen (gegen Race Conditions)
    conn.execute(f"""
        UPDATE jobs
           SET status='running',
               locked_at=datetime('now'),
               worker_id=?
         WHERE job_id=?
           AND status='queued'
           AND (run_not_before IS NULL OR {_norm('run_not_before')} <= datetime('now'))
           AND (locked_at IS NULL OR {_norm('locked_at')} < datetime('now', ?))
    """, (WORKER_ID, job_id, f"-{cutoff} seconds"))

    # 3) Erfolg robust ermitteln (SELECT changes() statt rowcount)
    r2 = conn.execute("SELECT COALESCE(changes(),0) AS changes").fetchone()
    if isinstance(r2, dict):
        claimed = int(r2.get("changes", 0))
    else:
        try:
            claimed = int(r2["changes"])
        except Exception:
            claimed = int(r2[0])

    if claimed == 1:
        # Nur loggen, wenn tatsächlich geclaimed wurde
        log_decision(conn, job_id, file_id, "claim", f"claimed by {WORKER_ID}")
        return job_id, file_id

    # Diagnose bei verfehltem Claim (z. B. Race oder Filter)
    safe_log(conn, job_id, file_id, "claim_skip",
             "claim_miss after UPDATE; status/lock/time window changed")
    return None, None


def cleanup_stale(conn):
    cutoff = int(LOCK_TIMEOUT_S)
    cur = conn.execute(
        """
        UPDATE jobs
           SET status='queued', locked_at=NULL, worker_id=NULL
         WHERE status='running' AND locked_at < datetime('now', ?)
        """,
        (f"-{cutoff} seconds",),
    )
    freed = cur.rowcount if hasattr(cur, "rowcount") else 0
    log(f"[cleanup] freed={freed} stale jobs")
    if freed:
        log_decision(conn, None, None, "cleanup", f"freed={freed}")
    return freed

def finish_success(conn, job_id, file_id, status: str):
    if file_id:
        next_review_at, review_reason = compute_next_review_at(status)
        conn.execute(
            """
            UPDATE files
               SET status=?,
                   last_error=NULL,
                   last_checked_at=datetime('now'),
                   next_review_at=?,
                   review_reason=?,
                   should_reingest=0,
                   updated_at=datetime('now')
             WHERE id=?;
            """,
            (status, next_review_at, review_reason, file_id),
        )
        log(
            f"[finish_success] job_id={job_id} file_id={file_id} status={status} next_review_at={next_review_at}"
        )
    else:
        log("[finish_success] file_id missing during update")
    conn.execute("UPDATE jobs SET status='done', locked_at=NULL WHERE job_id=?", (job_id,))
    conn.execute(
        "INSERT INTO decision_log(step, job_id, file_id, detail) VALUES(?,?,?,?)",
        ("done", job_id, file_id, status),
    )
    conn.commit()


def finish_error(conn, job_id, file_id, status: str, msg):
    message = str(msg)[:500]
    log(f"[finish_error] job_id={job_id} file_id={file_id} status={status} message={message}")
    if file_id:
        next_review_at, review_reason = compute_next_review_at(status)
        conn.execute(
            """
            UPDATE files
               SET status=?,
                   last_error=?,
                   last_checked_at=datetime('now'),
                   next_review_at=?,
                   review_reason=?,
                   updated_at=datetime('now')
             WHERE id=?;
            """,
            (status, message, next_review_at, review_reason, file_id),
        )
    conn.execute(
        "UPDATE jobs SET status='failed', last_error=?, locked_at=NULL WHERE job_id=?",
        (message, job_id),
    )
    conn.execute(
        "INSERT INTO decision_log(step, job_id, file_id, detail) VALUES(?,?,?,?)",
        ("error", job_id, file_id, f"{status}: {message}"),
    )
    conn.commit()

def update_file_result(conn, file_id, result_obj):
    conn.execute(
        "UPDATE files SET last_result=?, updated_at=datetime('now') WHERE id=?",
        (json.dumps(result_obj, ensure_ascii=False), file_id)
    )
    conn.commit()



@dataclass
class DoclingChunk:
    text: str
    meta: Dict[str, object]


@dataclass
class DoclingExtraction:
    text: str
    mime_type: Optional[str]
    chunks: List[DoclingChunk]
    images: List[Dict[str, object]]


class DoclingServeIngestor:
    """Interact with docling-serve and persist extracted images."""

    def __init__(
        self,
        *,
        chunk_size: int,
        chunk_overlap: int,
        max_chunks: int,
        service_url: str,
        image_dir: Path,
        nextcloud_client: NextcloudClient | None = None,
        remote_image_dir: str | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunks = max_chunks
        self.service_url = service_url.rstrip("/")
        self.image_dir = ensure_dir(Path(image_dir)) if nextcloud_client is None else Path(image_dir)
        self.nextcloud_client = nextcloud_client
        self.remote_image_dir = remote_image_dir or str(image_dir)

    def extract(self, file_path: Path) -> DoclingExtraction:
        if not file_path.exists():
            raise FileNotFoundError(file_path)

        try:
            with file_path.open("rb") as fp:
                files = {"files": (file_path.name, fp, "application/octet-stream")}
                response = requests.post(self.service_url, files=files, timeout=DOCLING_SERVE_TIMEOUT)
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = self._response_detail(exc.response)
            suffix = f"; {detail}" if detail else ""
            raise DoclingUnavailableError(f"docling-serve request failed: {exc}{suffix}") from exc
        except Exception as exc:
            raise DoclingUnavailableError(f"docling-serve request failed: {exc}") from exc

        try:
            payload = response.json()
        except Exception as exc:
            detail = self._response_detail(response)
            suffix = f"; {detail}" if detail else ""
            raise RuntimeError(f"docling-serve returned invalid JSON{suffix}") from exc

        text = self._extract_text(payload)
        base_slug = slugify(file_path.stem)

        text, inline_images = self._extract_inline_images(text, base_slug)
        images_payload = payload.get("images") or payload.get("media", {}).get("images") or []
        stored_images = self._store_images(images_payload, base_slug)
        if inline_images:
            stored_images.extend(self._store_images(inline_images, base_slug))

        text_with_refs = self._inject_image_refs(text, stored_images)
        mime = payload.get("media_type") or payload.get("mime_type")
        chunks = self._chunk_text(text_with_refs)
        return DoclingExtraction(text=text_with_refs, mime_type=mime, chunks=chunks, images=stored_images)

    def _extract_text(self, payload: Dict[str, object]) -> str:
        # 1) Neues docling-Schema: Text liegt in payload["document"][...]
        doc = payload.get("document")
        if isinstance(doc, dict):
            for key in ("md_content", "text_content", "html_content", "json_content"):
                value = doc.get(key)
                if isinstance(value, str) and value.strip():
                    return value

        # 2) Fallback: ältere / alternative Schemas auf Top-Level
        for key in ("text", "markdown", "content", "body"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value

        # 3) Wenn immer noch nichts gefunden: wie bisher Fehler werfen
        detail = self._payload_detail(payload)
        suffix = f"; {detail}" if detail else ""
        raise RuntimeError(f"docling-serve response did not contain textual content{suffix}")

    def _store_images(self, images_payload, base_slug: str) -> List[Dict[str, object]]:
        stored: List[Dict[str, object]] = []
        if not images_payload:
            return stored

        items = images_payload
        if isinstance(images_payload, dict):
            items = [{"name": name, "data": data} for name, data in images_payload.items()]

        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            raw = self._decode_data(item.get("data") or item.get("content") or item.get("base64") or item.get("payload") or item.get("value"))
            if raw is None:
                continue
            name = item.get("name") or item.get("id")
            mime = item.get("mime") or item.get("content_type") or item.get("type")
            ext = self._extension_from_mime(mime) or (Path(name).suffix if name else "") or ".bin"
            fname = name or f"{base_slug}-{idx}{ext if ext.startswith('.') else '.' + ext}"
            target = self.image_dir / fname
            reference = str(target)
            try:
                if self.nextcloud_client:
                    remote_path = posixpath.normpath(f"/{self.remote_image_dir}/{fname}")
                    log(f"[image_upload] pushing {fname} to Nextcloud {remote_path}")
                    self.nextcloud_client.upload_bytes(remote_path, raw)
                    reference = remote_path
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(raw)
            except Exception as exc:
                log(f"[image_store_failed] {target}: {exc}")
                continue
            stored.append(
                {
                    "label": fname,
                    "path": reference,
                    "reference": reference,
                    "mime": mime or mimetypes.guess_type(fname)[0] or "application/octet-stream",
                }
            )
        return stored

    def _decode_data(self, data: Optional[str]) -> Optional[bytes]:
        if not data:
            return None
        if isinstance(data, bytes):
            return data
        try:
            if data.startswith("data:"):
                _, _, b64_part = data.partition(",")
                return base64.b64decode(b64_part)
            return base64.b64decode(data)
        except Exception:
            return None

    def _extension_from_mime(self, mime: Optional[str]) -> str:
        if not mime:
            return ""
        ext = mimetypes.guess_extension(mime)
        return ext or ""

    def _extract_inline_images(self, text: str, base_slug: str) -> Tuple[str, List[Dict[str, object]]]:
        if not text:
            return text or "", []

        pattern = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<data>data:image/[^)]+)\)")
        matches = list(pattern.finditer(text))
        if not matches:
            return text, []

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        items: List[Dict[str, object]] = []
        item_indices: List[Optional[int]] = []
        for idx, match in enumerate(matches, start=1):
            data_uri = match.group("data")
            if not data_uri.lower().startswith("data:image/"):
                item_indices.append(None)
                continue
            identifier = f"{base_slug}_{timestamp}_{idx}.png"
            items.append({"name": identifier, "data": data_uri, "mime": "image/png"})
            item_indices.append(len(items) - 1)

        stored_images = self._store_images(items, base_slug) if items else []
        last_end = 0
        cleaned_parts: List[str] = []

        for idx, match in enumerate(matches, start=1):
            cleaned_parts.append(text[last_end:match.start()])
            item_idx = item_indices[idx - 1] if idx - 1 < len(item_indices) else None
            stored = stored_images[item_idx] if item_idx is not None and item_idx < len(stored_images) else None
            fallback_label = items[item_idx]["name"] if item_idx is not None and item_idx < len(items) else f"{base_slug}_{timestamp}_{idx}.png"
            target = stored.get("label") if stored else fallback_label
            alt_text = match.group("alt") or "image"
            cleaned_parts.append(f"![{alt_text}]({target})")
            last_end = match.end()

        cleaned_parts.append(text[last_end:])
        return "".join(cleaned_parts), stored_images

    def _inject_image_refs(self, text: str, images: List[Dict[str, object]]) -> str:
        if not images:
            return text or ""
        base = (text or "").rstrip()
        lines = [base] if base else []
        lines.append("")
        lines.append("## Extracted images")
        for img in images:
            label = img.get("label", "image")
            ref = img.get("reference", "")
            lines.append(f"![{label}]({ref})")
        return "\n".join(lines).strip()

    def _chunk_text(self, text: str) -> List[DoclingChunk]:
        chunks: List[DoclingChunk] = []
        if not text:
            return chunks
        step = max(self.chunk_size - self.chunk_overlap, 1)
        for idx, start in enumerate(range(0, len(text), step), start=1):
            if self.max_chunks and idx > self.max_chunks:
                break
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if not chunk_text:
                continue
            chunks.append(DoclingChunk(text=chunk_text, meta={"chunk_index": idx}))
            if end >= len(text):
                break
        if not chunks and text.strip():
            chunks.append(DoclingChunk(text=text.strip(), meta={"chunk_index": 1}))
        return chunks

    def _response_detail(self, response: Optional[requests.Response]) -> str:
        if response is None:
            return ""

        detail: str = ""
        try:
            parsed = response.json()
            if isinstance(parsed, dict):
                detail = str(parsed.get("detail") or parsed.get("error") or parsed)
            else:
                detail = str(parsed)
        except Exception:
            try:
                detail = response.text
            except Exception:
                detail = ""

        detail = (detail or "").strip()
        if len(detail) > 500:
            detail = detail[:500] + "…"
        return detail

    def _payload_detail(self, payload: Optional[Dict[str, object]]) -> str:
        if not payload:
            return ""
        try:
            if isinstance(payload, dict):
                for key in ("detail", "error", "message", "status"):
                    if payload.get(key):
                        return str(payload.get(key))[:500]
            return str(payload)[:500]
        except Exception:
            return ""


_DOCLING_INGESTOR: Optional[DoclingServeIngestor] = None


def get_docling_ingestor() -> DoclingServeIngestor:
    global _DOCLING_INGESTOR
    if _DOCLING_INGESTOR is None:
        nxc = get_nextcloud_client()
        log(f"[docling_ingestor] using Nextcloud images dir {NEXTCLOUD_IMAGE_DIR}")
        _DOCLING_INGESTOR = DoclingServeIngestor(
            chunk_size=MAX_CHARS,
            chunk_overlap=OVERLAP,
            max_chunks=MAX_CHUNKS,
            service_url=DOCLING_SERVE_URL,
            image_dir=Path(tempfile.gettempdir()) / "docling-images",
            nextcloud_client=nxc,
            remote_image_dir=NEXTCLOUD_IMAGE_DIR,
        )
    return _DOCLING_INGESTOR

def ai_score_text(ollama_host, model, text, timeout=600, debug=False, dbg_root: Path=Path("./debug")):
    """
    Bewertet Text via Ollama /api/generate und liefert ein Dict:
      {"is_relevant": bool, "confidence": float, "topics": [str], "summary": str, "visibility": str}

    Keine DB-/Logger-Abhängigkeiten. Debug schreibt optional Dateien in dbg_root.
    """
    import os, time, json, requests
    from datetime import datetime

    # --- Systemprompt & Nutzlast bauen ---
    sys_prompt = (
        "You are a strict, privacy-aware classifier for a personal knowledge base. "
        "Return ONLY compact JSON with keys exactly: "
        '{"is_relevant": true|false, "confidence": number, "topics": [string], '
        '"visibility": "public|private|confidential", "summary": "max 60 words"}.\n'
        "Relevant = helpful for future Q&A of THIS user (operations/ERP/home-lab/finance), "
        "technical how-to, personal notes, key decisions, configs, logs WITH context.\n"
        "Irrelevant = binaries, trivial short logs, duplicates, cache noise, images without text.\n"
        "If content exposes secrets or identifiers -> visibility='confidential'.\n"
        "Output JSON ONLY. No prose."
    )

    max_sample = int(os.environ.get("SCORER_SAMPLE_CHARS", "12000"))
    content_sample = (text or "")[:max_sample]
    user_prompt = f"{sys_prompt}\n---\nCONTENT START\n{content_sample}\nCONTENT END\n"

    # --- Debug-Verzeichnis vorbereiten (optional) ---
    dbg_dir = None
    if debug:
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        try:
            dbg_dir = ensure_dir(dbg_root / f"{ts}_{slugify((content_sample[:80] or 'doc'))}")
            _dbg_write(dbg_dir / "_entered.txt", utcnow_iso())
            _dbg_write(dbg_dir / "lengths.txt", f"content_len={len(content_sample)}")
            _dbg_write(dbg_dir / "prompt.txt", user_prompt)
            _dbg_write(dbg_dir / "content_head.txt", content_sample[:2000])
        except Exception:
            dbg_dir = None  # Debug nicht kritisch

    # --- HTTP-Request an Ollama ---
    url = f"{ollama_host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": user_prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "8192")),
            "top_p": 1.0,
            "repeat_penalty": 1.1,
        },
    }

    t0 = time.time()
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        elapsed = time.time() - t0

        if debug and dbg_dir:
            try:
                _dbg_write(dbg_dir / "http_status.txt", f"{r.status_code}")
                _dbg_write(dbg_dir / "latency_ms.txt", f"{int(elapsed*1000)}")
            except Exception:
                pass

        r.raise_for_status()
        resp = r.json()
        raw = (resp.get("response") or "").strip()

        if debug and dbg_dir:
            try:
                _dbg_write(dbg_dir / "raw_response.json", raw)
            except Exception:
                pass

        # --- JSON der Modellantwort parsen ---
        ai = {}
        try:
            ai = json.loads(raw) if raw else {}
        except Exception as e:
            if debug and dbg_dir:
                _dbg_write(dbg_dir / "json_error.txt", str(e))
            ai = {}

        # --- Felder robust extrahieren ---
        def _to_float(x, default=0.0):
            try:
                return float(x)
            except Exception:
                return default

        result = {
            "is_relevant": bool(ai.get("is_relevant", False)),
            "confidence": _to_float(ai.get("confidence", 0)),
            "topics": ai.get("topics") or [],
            "summary": ai.get("summary") or "",
            "visibility": ai.get("visibility") or "private",
        }

        if debug and dbg_dir:
            try:
                _dbg_write(dbg_dir / "parsed.json", json.dumps(result, indent=2))
                # Optional: Mini-Erklärung (nicht JSON), nur für Dev-Debug
                explain_prompt = (
                    "In one short sentence (max 25 words), explain why the document is or is not relevant to the user's knowledge base. "
                    "Do NOT include JSON, just a sentence."
                )
                r2 = requests.post(
                    url,
                    json={
                        "model": model,
                        "prompt": explain_prompt + "\n---\n" + content_sample[:2000],
                        "stream": False,
                        "options": {"temperature": 0},
                    },
                    timeout=30,
                )
                if r2.ok:
                    _dbg_write(dbg_dir / "why.txt", (r2.json().get("response") or "").strip())
            except Exception:
                pass

        return result

    except Exception as e:
        if debug and dbg_dir:
            try:
                _dbg_write(dbg_dir / "fatal_error.txt", str(e))
            except Exception:
                pass
        # Fallback-Ergebnis bei Fehlern
        return {
            "is_relevant": False,
            "confidence": 0.0,
            "topics": [],
            "summary": "",
            "visibility": "private",
        }



def brain_ingest_text(
    brain_url,
    api_key,
    text,
    meta,
    *,
    chunk_tokens: int | None = None,
    overlap_tokens: int | None = None,
    collection: str | None = None,
    timeout: float = 30,
):
    from datetime import datetime
    from pathlib import Path
    import os, json, requests, sqlite3

    url = brain_url.rstrip("/") + "/ingest/text"
    headers = {"x-api-key": api_key, "content-type": "application/json"}
    payload = {"text": text, "meta": meta}
    if chunk_tokens:
        payload["chunk_tokens"] = int(chunk_tokens)
    if overlap_tokens is not None:
        payload["overlap_tokens"] = int(overlap_tokens)
    if collection:
        payload["collection"] = collection

    # Optional: pre-log in DB for Nachvollziehbarkeit
    try:
        # Nur wenn eine globale conn vorhanden ist – sonst überspringen
        if 'conn' in globals():
            try:
                globals()['log_decision'](globals()['conn'], meta.get('job_id','?'), meta.get('file_id','?'),
                                          "pre_brain_post",
                                          f"url={url} len={len(text)} chunk_tokens={payload.get('chunk_tokens')} overlap_tokens={payload.get('overlap_tokens')}")
            except Exception:
                pass
    except Exception:
        pass

    dbg_on = os.environ.get("DEBUG_BRAIN") == "1"
    dbg_dir = Path(os.environ.get("BRAIN_DEBUG_DIR", "/opt/ct109-ingest/brain-debug"))
    if dbg_on:
        try:
            dbg_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            dbg_on = False

    if dbg_on:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        base = dbg_dir / f"{ts}-{os.getpid()}"
        req_dump = {
            "url": url,
            "headers": {"x-api-key": "***redacted***", "content-type": "application/json"},
            "meta": meta,
            "text_len": len(text),
            "text_head": text[:1000],
        }
        (base.with_suffix(".request.json")).write_text(json.dumps(req_dump, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if dbg_on:
            resp_dump = {"status": r.status_code, "ok": r.ok, "body_head": r.text[:2000]}
            (base.with_suffix(".response.json")).write_text(json.dumps(resp_dump, ensure_ascii=False, indent=2), encoding="utf-8")
        r.raise_for_status()
        return r.json()
    except Exception as e:
        if dbg_on:
            from traceback import format_exc
            (base.with_suffix(".error.json")).write_text(format_exc(), encoding="utf-8")
        raise
    except Exception as e:
        if dbg_on:
            err_dump = {"error": repr(e)}
            (base.with_suffix(".error.json")).write_text(json.dumps(err_dump, ensure_ascii=False, indent=2), encoding="utf-8")
        raise



# === Worker logic ===

def process_one(conn, job_id, file_id):
    """Run a full ingest pipeline for a single file (download, extract, chunk, context, and upload to Qdrant)."""
    from pathlib import Path

    # --- Helper für row-factory-agnostischen Zugriff (dict/sqlite3.Row/tuple) ---
    def _get(row, key_or_idx, fallback=None):
        if row is None:
            return fallback
        if isinstance(row, dict):
            return row.get(key_or_idx, fallback)
        try:
            return row[key_or_idx]
        except Exception:
            try:
                return row[int(key_or_idx)]
            except Exception:
                return fallback

    log(f"[process_start] job_id={job_id} file_id={file_id}")
    # 1) Pfad aus files
    row = conn.execute("SELECT path, size, mtime, priority FROM files WHERE id=?", (file_id,)).fetchone()
    if not row:
        finish_error(conn, job_id, file_id, "error_missing_db_record", "file_id not found in files")
        return
    path = _get(row, "path", _get(row, 0))
    if not path:
        finish_error(conn, job_id, file_id, "error_missing_path", "path column missing/empty")
        return
    original_path = str(path)
    p = Path(path)
    temp_file: Path | None = None
    size_hint = _get(row, "size")
    log(
        f"[process_path] job_id={job_id} file_id={file_id} path={p} size={size_hint} priority={_get(row, 'priority')}"
    )
    safe_log(conn, job_id, file_id, "path_ok", str(p))

    if not p.exists():
        try:
            client = get_nextcloud_client()
            log(
                f"[download_missing] job_id={job_id} file_id={file_id} path={path} base_url={NEXTCLOUD_BASE_URL}"
            )
            temp_file = client.download_to_temp(path, suffix=p.suffix)
            p = temp_file
            safe_log(conn, job_id, file_id, "downloaded", f"temp={p}")
        except Exception as exc:
            log(f"[download_missing_error] job_id={job_id} path={path} err={exc}")
            update_file_result(conn, file_id, {"accepted": False, "error": "missing_file"})
            finish_error(conn, job_id, file_id, "error_missing_file", f"{exc}")
            return

    # Snapshot der Gates/ENV
    try:
        gate_msg = (
            f"MIN_CHARS={MIN_CHARS} "
            f"RELEVANCE_THRESHOLD={RELEVANCE_THRESHOLD} "
            f"MAX_TEXT_CHARS={MAX_TEXT_CHARS} "
            f"MAX_CHUNKS={MAX_CHUNKS} OVERLAP={OVERLAP} "
            f"OLLAMA_HOST={'set' if OLLAMA_HOST else 'unset'} "
            f"OLLAMA_MODEL={OLLAMA_MODEL!s} "
            f"MODEL_RELEVANCE={OLLAMA_MODEL_RELEVANCE!s} "
            f"MODEL_CHUNKING={OLLAMA_MODEL_CHUNKING!s} "
            f"MODEL_CONTEXT={OLLAMA_MODEL_CONTEXT!s} "
            f"BRAIN_URL={'set' if BRAIN_URL else 'unset'} "
            f"KEY={'yes' if BRAIN_API_KEY else 'no'}"
        )
        safe_log(conn, job_id, file_id, "gate_snapshot", gate_msg)
    except Exception:
        pass

    # 2) Existiert Datei?
    if not p.exists():
        update_file_result(conn, file_id, {"accepted": False, "error": "missing_file"})
        finish_error(conn, job_id, file_id, "error_missing_file", "file missing on disk")
        return

    try:
        size = p.stat().st_size
        log(f"[extract_start] job_id={job_id} file_id={file_id} path={p} size={size}")
        ingestor = get_docling_ingestor()

        extraction: Optional[DoclingExtraction] = None
        docling_error: Optional[Exception] = None
        docling_source = "docling-serve"
        try:
            extraction = ingestor.extract(p)
        except Exception as e:
            docling_error = e
            log(f"[extract_error] job_id={job_id} file_id={file_id} path={p} err={e}")

        if extraction is None and p.suffix.lower() == ".pdf":
            log(
                f"[extract_fallback] job_id={job_id} file_id={file_id} using legacy PDF fallback due to {docling_error}"
            )
            safe_log(conn, job_id, file_id, "docling_pdf_fallback", str(docling_error))
            try:
                fallback_text = legacy_pdf_fallback_text(p)
            except Exception as e:
                fallback_text = ""
                if docling_error is None:
                    docling_error = e
            if fallback_text:
                log(
                    f"[extract_fallback_success] job_id={job_id} file_id={file_id} text_len={len(fallback_text)}"
                )
                chunks = ingestor._chunk_text(fallback_text)
                extraction = DoclingExtraction(
                    text=fallback_text,
                    mime_type="text/plain",
                    chunks=chunks,
                    images=[],
                )
                docling_source = "pdf_fallback"

        if extraction is None:
            update_file_result(
                conn,
                file_id,
                {"accepted": False, "skipped": "extract_failed", "error": str(docling_error)},
            )
            finish_error(conn, job_id, file_id, "error_extract_failed", f"docling_failed: {docling_error}")
            return

        clean = (extraction.text or "").strip()
        detected = extraction.mime_type or "docling-serve"
        log(
            f"[extract_ok] job_id={job_id} file_id={file_id} mime={detected} chars={len(clean)} chunks={len(extraction.chunks)} source={docling_source}"
        )
        log_decision(conn, job_id, file_id, "sniff", f"mime={detected}; size={size}; source={docling_source}")
        log_decision(
            conn,
            job_id,
            file_id,
            "extract_ok",
            f"chars={len(clean)} docling_chunks={len(extraction.chunks)} source={docling_source}",
        )

        if not clean or len(clean) < MIN_CHARS:
            update_file_result(conn, file_id, {"accepted": False, "skipped": "too_short", "chars": len(clean)})
            log_decision(conn, job_id, file_id, "threshold", f"skip: too_short chars={len(clean)} < MIN_CHARS={MIN_CHARS}")
            finish_success(conn, job_id, file_id, "skipped_too_short")
            return

        if len(clean) > MAX_TEXT_CHARS:
            clean = clean[:MAX_TEXT_CHARS]
            log_decision(conn, job_id, file_id, "threshold", f"cap: truncated to {MAX_TEXT_CHARS} chars")

        try:
            if not OLLAMA_HOST:
                safe_log(conn, job_id, file_id, "no_ollama", "OLLAMA_HOST not set -> skipping AI score")
                raise RuntimeError("OLLAMA disabled")
            log(
                f"[ai_score_start] job_id={job_id} file_id={file_id} len={len(clean)} host={OLLAMA_HOST} model={OLLAMA_MODEL_RELEVANCE}"
            )
            safe_log(conn, job_id, file_id, "pre_ai", f"len={len(clean)} host={OLLAMA_HOST} model={OLLAMA_MODEL_RELEVANCE}")
            ai = ai_score_text(
                OLLAMA_HOST,
                OLLAMA_MODEL_RELEVANCE,
                clean,
                timeout=600,
                debug=DEBUG,
                dbg_root=Path("./debug"),
            )
            is_rel = bool(ai.get("is_relevant"))
            conf   = float(ai.get("confidence", 0.0))
            log(f"[ai_score_result] job_id={job_id} file_id={file_id} is_rel={is_rel} conf={conf}")
            safe_log(conn, job_id, file_id, "score", f"is_rel={is_rel} conf={conf}")
        except Exception as e:
            update_file_result(conn, file_id, {"accepted": False, "error": f"ai_scoring_failed: {e}"})
            finish_error(conn, job_id, file_id, "error_ai_scoring", f"ai_scoring_failed: {e}")
            return

        if not is_rel or conf < RELEVANCE_THRESHOLD:
            log(
                f"[ai_threshold_skip] job_id={job_id} file_id={file_id} is_rel={is_rel} conf={conf} threshold={RELEVANCE_THRESHOLD}"
            )
            update_file_result(conn, file_id, {"accepted": False, "ai": ai, "reason": "below_threshold", "threshold": RELEVANCE_THRESHOLD})
            log_decision(conn, job_id, file_id, "threshold", f"skip: is_relevant={is_rel} conf={conf} < RELEVANCE_THRESHOLD={RELEVANCE_THRESHOLD}")
            finish_success(conn, job_id, file_id, "ai_not_relevant")
            return

        log(
            f"[chunking_start] job_id={job_id} file_id={file_id} len={len(clean)} max_chunks={MAX_CHUNKS} overlap={OVERLAP}"
        )
        safe_log(
            conn,
            job_id,
            file_id,
            "chunking_request",
            f"model={OLLAMA_MODEL_CHUNKING} tokens={BRAIN_CHUNK_TOKENS} overlap_chars={OVERLAP} max_chunks={MAX_CHUNKS} chars={len(clean)}",
        )
        chunk_texts, chunk_meta = chunk_document_with_llm_fallback(
            clean,
            ollama_host=OLLAMA_HOST,
            model=OLLAMA_MODEL_CHUNKING,
            target_tokens=BRAIN_CHUNK_TOKENS,
            overlap_chars=OVERLAP,
            max_chunks=MAX_CHUNKS,
            timeout=BRAIN_REQUEST_TIMEOUT,
            debug=DEBUG,
            return_debug=True,
        )

        chunk_source = "llm_chunking"
        if chunk_meta.get("chunk_source") == "fallback":
            chunk_source = "llm_fallback"
        if chunk_meta:
            safe_log(
                conn,
                job_id,
                file_id,
                "chunking_meta",
                json.dumps(chunk_meta, ensure_ascii=False)[:500],
            )
            log_debug(
                f"[chunking_meta] job_id={job_id} file_id={file_id} llm={chunk_meta.get('llm_info')} fallback={chunk_meta.get('fallback_used')}"
            )

        if not chunk_texts and extraction.chunks:
            chunk_texts = [c.text for c in extraction.chunks]
            chunk_source = "docling_fallback"
            log_debug(
                f"[chunking_docling_fallback] job_id={job_id} file_id={file_id} using {len(chunk_texts)} docling chunks"
            )
        elif not chunk_texts:
            chunk_texts = [clean]
            chunk_source = "single_block"
            log_debug(
                f"[chunking_single_block] job_id={job_id} file_id={file_id} chars={len(clean)}"
            )
        log(
            f"[chunking_done] job_id={job_id} file_id={file_id} chunks={len(chunk_texts)} source={chunk_source}"
        )
        log_debug(
            f"[chunking_detail] job_id={job_id} file_id={file_id} chunk_lengths={[len(c) for c in chunk_texts]}"
        )
        try:
            safe_log(
                conn,
                job_id,
                file_id,
                "chunking_lengths",
                str([len(c) for c in chunk_texts])[:500],
            )
        except Exception:
            pass

        chunks: List[DoclingChunk] = []
        for idx, chunk_text in enumerate(chunk_texts, 1):
            meta: Dict[str, object] = {
                "chunk_index": idx,
                "chunks_total": len(chunk_texts),
            }
            if chunk_source == "docling_fallback" and idx - 1 < len(extraction.chunks):
                original_meta = extraction.chunks[idx - 1].meta
                for key in ("page_start", "page_end", "section"):
                    if key in original_meta:
                        meta[key] = original_meta[key]
            chunks.append(DoclingChunk(text=chunk_text, meta=meta))
            log_debug(
                f"[chunk_built] job_id={job_id} file_id={file_id} idx={idx}/{len(chunk_texts)} len={len(chunk_text)} meta={meta}"
            )

        try:
            enriched = enrich_chunks_with_context(
                document=clean,
                chunks=[c.text for c in chunks],
                ollama_host=OLLAMA_HOST,
                model=OLLAMA_MODEL_CONTEXT,
                timeout=BRAIN_REQUEST_TIMEOUT,
                debug=DEBUG,
            )
            if enriched and len(enriched) == len(chunks):
                for chunk, enriched_text in zip(chunks, enriched):
                    chunk.text = enriched_text
                chunk_source += "+context"
        except Exception as e:
            safe_log(conn, job_id, file_id, "context_fallback", f"{e}")

        log(
            f"[chunk_plan] job_id={job_id} file_id={file_id} chunks={len(chunks)} source={chunk_source} overlap={OVERLAP}"
        )
        log_decision(
            conn,
            job_id,
            file_id,
            "chunk_plan",
            f"chunks={len(chunks)} via {chunk_source} overlap={OVERLAP}",
        )

        errors = []
        for idx, chunk in enumerate(chunks, 1):
            chunk_meta = {
                "source": "RAGIngestQdrant",
                "path": original_path,
                "document_name": posixpath.basename(original_path.rstrip("/")) or p.name,
                "chunk_index": chunk.meta.get("chunk_index", idx),
                "chunks_total": len(chunks),
                "job_id": job_id,
                "file_id": file_id,
            }
            for key in ("page_start", "page_end", "section"):
                value = chunk.meta.get(key)
                if value is not None:
                    chunk_meta[key] = value
            try:
                if not BRAIN_URL:
                    safe_log(conn, job_id, file_id, "no_brain_url", "BRAIN_URL not set -> skipping POST")
                    raise RuntimeError("Brain URL not configured")
                log(
                    f"[brain_post] job_id={job_id} file_id={file_id} chunk={idx}/{len(chunks)} len={len(chunk.text)}"
                )
                log_debug(
                    f"[brain_post_payload] job_id={job_id} file_id={file_id} idx={idx} meta={chunk_meta}"
                )
                safe_log(
                    conn,
                    job_id,
                    file_id,
                    "pre_brain_post",
                    f"chunk={idx}/{len(chunks)} len={len(chunk.text)} url={BRAIN_URL}",
                )
                ok = brain_ingest_text(
                    BRAIN_URL,
                    BRAIN_API_KEY,
                    chunk.text,
                    meta=chunk_meta,
                    collection=BRAIN_COLLECTION,
                    timeout=BRAIN_REQUEST_TIMEOUT,
                )
                if ok:
                    safe_log(conn, job_id, file_id, "brain_ok", f"chunk={idx}")
                if not ok:
                    errors.append(f"chunk {idx} failed")
            except Exception as e:
                safe_log(conn, job_id, file_id, "brain_err", f"chunk={idx} err={e}")
                log(
                    f"[brain_error] job_id={job_id} file_id={file_id} chunk={idx}/{len(chunks)} err={e}"
                )
                errors.append(f"chunk {idx} error: {e}")

        result = {
            "accepted": len(errors) == 0,
            "ai": ai,
            "chunks": len(chunks),
            "docling": {
                "mime": extraction.mime_type,
                "source": docling_source,
                "chunks": [chunk.meta for chunk in chunks],
                "images": extraction.images,
            },
            "images": extraction.images,
        }
        if errors:
            result["errors"] = errors
            update_file_result(conn, file_id, result)
            finish_error(conn, job_id, file_id, "error_brain_ingest", "; ".join(errors)[:500])
            return

        update_file_result(conn, file_id, result)
        finish_success(conn, job_id, file_id, "vectorized")
    finally:
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass


def main():
    # Robust SQLite connection: timeout + autocommit and WAL/busy settings
    db_path = Path(DB_PATH)
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    conn = sqlite3.connect(str(db_path), timeout=30, isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
    except Exception:
        pass
    conn.row_factory = sqlite_dict_factory

    log(
        f"[startup] DB={DB_PATH} brain_url={BRAIN_URL} key_present={bool(BRAIN_API_KEY)} "
        f"threshold={RELEVANCE_THRESHOLD} worker_id={WORKER_ID}"
    )
    try:
        while True:
            log("[loop] tick")
            cleanup_stale(conn)
            job_id, file_id = claim_one(conn)
            log(f"[claim] job_id={job_id} file_id={file_id}")
            if not job_id:
                log("no queued jobs found - shutting down")
                break
            try:
                process_one(conn, job_id, file_id)
            except Exception as e:
                traceback.print_exc()
                finish_error(conn, job_id, file_id, "error_fatal", f"fatal: {e}")
    except KeyboardInterrupt:
        log("stopping...")
    finally:
        try:
            conn.close()
        except Exception:
            pass
        log("worker stopped")

if __name__ == "__main__":
    main()
