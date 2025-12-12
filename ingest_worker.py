
#!/usr/bin/env python3
import json
import logging
import os
import posixpath
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, Future

import requests
from nextcloud_client import env_client, NextcloudClient, NextcloudError

import initENV
from add_context import enrich_chunks_with_context
from chunking import chunk_document_with_llm_fallback
from helpers import compute_next_review_at, ensure_db, utcnow_iso
from prompt_store import get_prompt
from text_extraction import (
    DoclingChunk,
    ExtractionFailed,
    ExtractionOutcome,
    ensure_dir,
    extract_document,
    slugify,
)



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
DB_PATH = initENV.DB_PATH
BRAIN_URL = initENV.BRAIN_URL
BRAIN_API_KEY = initENV.BRAIN_API_KEY
BRAIN_COLLECTION = initENV.BRAIN_COLLECTION
BRAIN_CHUNK_TOKENS = initENV.BRAIN_CHUNK_TOKENS
BRAIN_REQUEST_TIMEOUT = initENV.BRAIN_REQUEST_TIMEOUT
OLLAMA_HOST = initENV.OLLAMA_HOST
OLLAMA_MODEL = initENV.OLLAMA_MODEL
OLLAMA_MODEL_RELEVANCE = initENV.OLLAMA_MODEL_RELEVANCE
OLLAMA_MODEL_CHUNKING = initENV.OLLAMA_MODEL_CHUNKING
OLLAMA_MODEL_CONTEXT = initENV.OLLAMA_MODEL_CONTEXT
RELEVANCE_THRESHOLD = initENV.RELEVANCE_THRESHOLD
MIN_CHARS = initENV.MIN_CHARS
MAX_TEXT_CHARS = initENV.MAX_TEXT_CHARS
MAX_CHARS = initENV.MAX_CHARS
OVERLAP = initENV.OVERLAP
MAX_CHUNKS = initENV.MAX_CHUNKS
ENABLE_OCR = initENV.ENABLE_OCR
DEBUG = initENV.DEBUG
LOCK_TIMEOUT_S = initENV.LOCK_TIMEOUT_S
IDLE_SLEEP_S = initENV.IDLE_SLEEP_S
PDFTOTEXT_TIMEOUT_S = initENV.PDFTOTEXT_TIMEOUT_S
PDF_OCR_MAX_PAGES = initENV.PDF_OCR_MAX_PAGES
PDF_OCR_DPI = initENV.PDF_OCR_DPI
RETRY_PRIORITY_ERROR = 90

DOCLING_SERVE_URL = initENV.DOCLING_SERVE_URL
DOCLING_SERVE_TIMEOUT = initENV.DOCLING_SERVE_TIMEOUT
DOCLING_MAX_WORKERS = initENV.DOCLING_MAX_WORKERS
NEXTCLOUD_DOC_DIR = initENV.NEXTCLOUD_DOC_DIR
NEXTCLOUD_IMAGE_DIR = initENV.NEXTCLOUD_IMAGE_DIR
NEXTCLOUD_BASE_URL = initENV.NEXTCLOUD_BASE_URL
NEXTCLOUD_USER = initENV.NEXTCLOUD_USER
NEXTCLOUD_TOKEN = initENV.NEXTCLOUD_TOKEN

DECISION_LOG_ENABLED = initENV.DECISION_LOG_ENABLED
DECISION_LOG_MAX_PER_JOB = initENV.DECISION_LOG_MAX_PER_JOB

WORKER_ID = f"{os.uname().nodename}-pid{os.getpid()}"

LOG_PATH = Path("logs") / "scan_scheduler.log"

_LOGGER: Optional[logging.Logger] = None


@dataclass
class DoclingAsyncTaskState:
    task_id: str
    file_name: str
    start_time: float
    deadline: float
    attempts: int = 0
    last_error: Exception | None = None
    next_interval: float = 0.0


@dataclass
class ExtractionStageResult:
    job_id: str
    file_id: str
    extraction_outcome: ExtractionOutcome
    clean_text: str
    doc_dbg_dir: Path | None


class DoclingAsyncManager:
    def __init__(self, max_parallel: int = 5):
        self.max_parallel = max_parallel
        self._semaphore = threading.Semaphore(max_parallel)
        self._lock = threading.Lock()
        self._active: Dict[str, DoclingAsyncTaskState] = {}
        self._queued: List[DoclingAsyncTaskState] = []

    @contextmanager
    def slot(self, file_name: str, deadline: float, initial_interval: float):
        state = DoclingAsyncTaskState(
            task_id=f"{file_name}-{time.time_ns()}",
            file_name=file_name,
            start_time=time.time(),
            deadline=deadline,
            next_interval=initial_interval,
        )
        with self._lock:
            self._queued.append(state)
            queued_names = [s.file_name for s in self._queued]
        log_debug(
            f"[docling_async_queue] queued={queued_names} active={len(self._active)} limit={self.max_parallel}"
        )

        self._semaphore.acquire()
        with self._lock:
            self._queued = [s for s in self._queued if s.task_id != state.task_id]
            self._active[state.task_id] = state
            active = len(self._active)
        log_debug(
            f"[docling_async_slot_acquired] task_id={state.task_id} file={file_name} "
            f"active={active} limit={self.max_parallel}"
        )
        try:
            yield state
        finally:
            with self._lock:
                self._active.pop(state.task_id, None)
                active = len(self._active)
            self._semaphore.release()
            log_debug(
                f"[docling_async_slot_released] task_id={state.task_id} file={file_name} active={active}"
            )


_DOC_EXTRACTION_MANAGER = DoclingAsyncManager(max_parallel=DOCLING_MAX_WORKERS)


def _configure_debug_logger() -> Optional[logging.Logger]:
    if not DEBUG:
        return None
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log_path = LOG_PATH.resolve()
        logger = logging.getLogger("ingest_worker")
        # Avoid duplicate handlers if multiple modules configure the same logger
        if not any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", None) == str(log_path)
            for h in logger.handlers
        ):
            logger.setLevel(logging.INFO)
            handler = logging.FileHandler(log_path, encoding="utf-8")
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s [worker] [%(name)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(handler)
        logger.propagate = False
        return logger
    except Exception:
        return None


_LOGGER = _configure_debug_logger()

_NEXTCLOUD_CLIENT: NextcloudClient | None = None

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [worker] [{WORKER_ID}] {msg}"
    print(line, flush=True)
    if DEBUG and _LOGGER is not None:
        try:
            _LOGGER.info(f"[{WORKER_ID}] {msg}")
        except Exception:
            pass


def create_db_conn():
    db_path = Path(DB_PATH)
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    conn = sqlite3.connect(str(db_path), timeout=30, isolation_level=None, check_same_thread=False)
    ensure_db(conn)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
    except Exception:
        pass
    conn.row_factory = sqlite_dict_factory
    return conn


def log_debug(msg):
    if DEBUG:
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


def next_error_count(conn, file_id: str) -> int:
    cur = conn.execute(
        "SELECT COALESCE(error_count, 0) FROM files WHERE id=?",
        (file_id,),
    )
    row = cur.fetchone()
    current = 0
    if row:
        try:
            current = int(row[0])
        except Exception:
            current = 0
    return current + 1

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
                   updated_at=datetime('now'),
                   error_count=0
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
        error_count = next_error_count(conn, file_id)
        next_review_at, review_reason = compute_next_review_at(status, error_count=error_count)
        conn.execute(
            """
            UPDATE files
               SET status=?,
                   last_error=?,
                   last_checked_at=datetime('now'),
                   next_review_at=?,
                   review_reason=?,
                   updated_at=datetime('now'),
                   error_count=?,
                   priority=?
             WHERE id=?;
            """,
            (
                status,
                message,
                next_review_at,
                review_reason,
                error_count,
                RETRY_PRIORITY_ERROR,
                file_id,
            ),
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

def replace_file_images(conn, file_id: str, images: List[Dict[str, object]] | None):
    """Persist the latest image references for a file (idempotent)."""
    try:
        log(f"[image_debug] file_id={file_id} images_count={len(images or [])}")
        ensure_db(conn)
        conn.execute("DELETE FROM images WHERE file_id=?", (file_id,))
        for img in images or []:
            reference = img.get("reference") or img.get("path") or ""
            if not reference:
                continue
            conn.execute(
                "INSERT INTO images (file_id, label, reference, mime) VALUES (?,?,?,?)",
                (
                    file_id,
                    img.get("label"),
                    reference,
                    img.get("mime"),
                ),
            )
        conn.commit()
    except Exception as exc:
        log(f"[image_record_failed] file_id={file_id} err={exc}")
        traceback.print_exc()

def ai_score_text(
    ollama_host,
    model,
    text,
    timeout=600,
    debug=False,
    dbg_root: Path = Path("./debug"),
    dbg_dir: Optional[Path] = None,
):
    """
    Bewertet Text via Ollama /api/generate und liefert ein Dict:
      {"is_relevant": bool, "confidence": float, "topics": [str], "summary": str, "visibility": str}

    Keine DB-/Logger-Abhängigkeiten. Debug schreibt optional Dateien in dbg_root.
    """
    import os, time, json, requests
    from datetime import datetime

    # --- Systemprompt & Nutzlast bauen ---
    sys_prompt = get_prompt("relevance")

    max_sample = int(os.environ.get("SCORER_SAMPLE_CHARS", "12000"))
    content_sample = (text or "")[:max_sample]
    user_prompt = f"{sys_prompt}\n---\nCONTENT START\n{content_sample}\nCONTENT END\n"

    # --- Debug-Verzeichnis vorbereiten (optional) ---
    dbg_active_dir: Optional[Path] = None
    if debug:
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        target_dir = dbg_dir or (dbg_root / f"{ts}_{slugify((content_sample[:80] or 'doc'))}")
        try:
            dbg_active_dir = ensure_dir(target_dir)
            _dbg_write(dbg_active_dir / "_entered.txt", utcnow_iso())
            _dbg_write(dbg_active_dir / "lengths.txt", f"content_len={len(content_sample)}")
            _dbg_write(dbg_active_dir / "prompt.txt", user_prompt)
            _dbg_write(dbg_active_dir / "content_head.txt", content_sample[:2000])
        except Exception:
            dbg_active_dir = None  # Debug nicht kritisch

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

        if debug and dbg_active_dir:
            try:
                _dbg_write(dbg_active_dir / "http_status.txt", f"{r.status_code}")
                _dbg_write(dbg_active_dir / "latency_ms.txt", f"{int(elapsed*1000)}")
            except Exception:
                pass

        r.raise_for_status()
        resp = r.json()
        raw = (resp.get("response") or "").strip()

        if debug and dbg_active_dir:
            try:
                _dbg_write(dbg_active_dir / "raw_response.json", raw)
            except Exception:
                pass

        # --- JSON der Modellantwort parsen ---
        ai = {}
        try:
            ai = json.loads(raw) if raw else {}
        except Exception as e:
            if debug and dbg_active_dir:
                _dbg_write(dbg_active_dir / "json_error.txt", str(e))
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

        if debug and dbg_active_dir:
            try:
                _dbg_write(dbg_active_dir / "parsed.json", json.dumps(result, indent=2))
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
                    _dbg_write(dbg_active_dir / "why.txt", (r2.json().get("response") or "").strip())
            except Exception:
                pass

        return result

    except Exception as e:
        if debug and dbg_active_dir:
            try:
                _dbg_write(dbg_active_dir / "fatal_error.txt", str(e))
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
    debug: bool = False,
    dbg_dir: Optional[Path] = None,
    dbg_root: Path = Path("./debug"),
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

    dbg_on = debug
    dbg_active_dir: Optional[Path] = None
    dbg_base: Optional[Path] = None
    if dbg_on:
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        target_dir = dbg_dir or (dbg_root / f"{ts}_{slugify(((text or '')[:80] or 'doc'))}")
        try:
            dbg_active_dir = ensure_dir(target_dir)
            chunk_label = meta.get("chunk_index")
            chunk_suffix = f"chunk{chunk_label}" if chunk_label is not None else f"chunk-{os.getpid()}"
            dbg_base = dbg_active_dir / chunk_suffix
            req_dump = {
                "url": url,
                "headers": {"x-api-key": "***redacted***", "content-type": "application/json"},
                "meta": meta,
                "text_len": len(text),
                "text_head": text[:1000],
            }
            (dbg_base.with_suffix(".request.json")).write_text(
                json.dumps(req_dump, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            dbg_on = False

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if dbg_on and dbg_base:
            resp_dump = {"status": r.status_code, "ok": r.ok, "body_head": r.text[:2000]}
            (dbg_base.with_suffix(".response.json")).write_text(
                json.dumps(resp_dump, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        if dbg_on and dbg_base:
            from traceback import format_exc
            (dbg_base.with_suffix(".error.json")).write_text(format_exc(), encoding="utf-8")
        raise
    except Exception as e:
        if dbg_on and dbg_base:
            err_dump = {"error": repr(e)}
            (dbg_base.with_suffix(".error.json")).write_text(
                json.dumps(err_dump, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        raise



# === Worker logic ===


def _get_field(row, key_or_idx, fallback=None):
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



def run_extraction_stage(conn, job_id: str, file_id: str) -> ExtractionStageResult | None:
    from pathlib import Path

    log(f"[process_start] job_id={job_id} file_id={file_id}")
    row = conn.execute("SELECT path, size, mtime, priority FROM files WHERE id=?", (file_id,)).fetchone()
    if not row:
        finish_error(conn, job_id, file_id, "error_missing_db_record", "file_id not found in files")
        return None
    path = _get_field(row, "path", _get_field(row, 0))
    if not path:
        finish_error(conn, job_id, file_id, "error_missing_path", "path column missing/empty")
        return None
    original_path = str(path)
    p = Path(path)
    temp_file: Path | None = None
    size_hint = _get_field(row, "size")
    log(f"[process_path] job_id={job_id} file_id={file_id} path={p} size={size_hint} priority={_get_field(row, 'priority')}")
    safe_log(conn, job_id, file_id, "path_ok", str(p))
    log_decision(conn, job_id, file_id, "docling_queue", "pending docling conversion")

    if not p.exists():
        try:
            client = get_nextcloud_client()
            log(f"[download_missing] job_id={job_id} file_id={file_id} path={path} base_url={NEXTCLOUD_BASE_URL}")
            temp_file = client.download_to_temp(path, suffix=p.suffix)
            p = temp_file
            safe_log(conn, job_id, file_id, "downloaded", f"temp={p}")
        except Exception as exc:
            log(f"[download_missing_error] job_id={job_id} path={path} err={exc}")
            update_file_result(conn, file_id, {"accepted": False, "error": "missing_file"})
            finish_error(conn, job_id, file_id, "error_missing_file", f"{exc}")
            return None

    try:
        log_decision(conn, job_id, file_id, "docling_extracting", "sending to docling-serve")
        extraction_outcome = extract_document(
            p,
            job_id=job_id,
            file_id=file_id,
            original_path=original_path,
            async_slot_provider=_DOC_EXTRACTION_MANAGER.slot,
        )
        extraction = extraction_outcome.extraction
        docling_source = extraction_outcome.source
        debug_dir = extraction_outcome.debug_dir
        clean = (extraction.text or "").strip()
        detected = extraction.mime_type or "docling-serve"
        log_decision(conn, job_id, file_id, "sniff", f"mime={detected}; source={docling_source}")
        log_decision(conn, job_id, file_id, "extract_ok", f"chars={len(clean)} docling_chunks={len(extraction.chunks)} source={docling_source}")
    except ExtractionFailed as exc:
        update_file_result(conn, file_id, {"accepted": False, "skipped": "extract_failed", "error": str(exc)})
        finish_error(conn, job_id, file_id, "error_extract_failed", str(exc))
        return None
    finally:
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass

    if not clean or len(clean) < MIN_CHARS:
        update_file_result(conn, file_id, {"accepted": False, "skipped": "too_short", "chars": len(clean)})
        log_decision(conn, job_id, file_id, "threshold", f"skip: too_short chars={len(clean)} < MIN_CHARS={MIN_CHARS}")
        finish_success(conn, job_id, file_id, "skipped_too_short")
        return None

    if len(clean) > MAX_TEXT_CHARS:
        clean = clean[:MAX_TEXT_CHARS]
        log_decision(conn, job_id, file_id, "threshold", f"cap: truncated to {MAX_TEXT_CHARS} chars")

    doc_dbg_dir: Optional[Path] = None
    if DEBUG and debug_dir:
        try:
            doc_dbg_dir = ensure_dir(debug_dir / "ai_score")
        except Exception:
            doc_dbg_dir = None

    log_decision(conn, job_id, file_id, "extraction_complete", f"chars={len(clean)}")

    return ExtractionStageResult(job_id, file_id, extraction_outcome, clean, doc_dbg_dir)



def run_post_extraction_pipeline(conn, stage: ExtractionStageResult):
    job_id = stage.job_id
    file_id = stage.file_id
    extraction_outcome = stage.extraction_outcome
    extraction = extraction_outcome.extraction
    clean = stage.clean_text
    doc_dbg_dir = stage.doc_dbg_dir

    try:
        if not OLLAMA_HOST:
            safe_log(conn, job_id, file_id, "no_ollama", "OLLAMA_HOST not set -> skipping AI score")
            raise RuntimeError("OLLAMA disabled")
        log_decision(conn, job_id, file_id, "stage_relevance", "running relevance check")
        log(f"[ai_score_start] job_id={job_id} file_id={file_id} len={len(clean)} host={OLLAMA_HOST} model={OLLAMA_MODEL_RELEVANCE}")
        safe_log(conn, job_id, file_id, "pre_ai", f"len={len(clean)} host={OLLAMA_HOST} model={OLLAMA_MODEL_RELEVANCE}")
        ai = ai_score_text(
            OLLAMA_HOST,
            OLLAMA_MODEL_RELEVANCE,
            clean,
            timeout=600,
            debug=DEBUG,
            dbg_root=Path("./debug"),
            dbg_dir=doc_dbg_dir,
        )
        is_rel = bool(ai.get("is_relevant"))
        conf = float(ai.get("confidence", 0.0))
        log(f"[ai_score_result] job_id={job_id} file_id={file_id} is_rel={is_rel} conf={conf}")
        safe_log(conn, job_id, file_id, "score", f"is_rel={is_rel} conf={conf}")
    except Exception as e:
        update_file_result(conn, file_id, {"accepted": False, "error": f"ai_scoring_failed: {e}"})
        finish_error(conn, job_id, file_id, "error_ai_scoring", f"ai_scoring_failed: {e}")
        return

    if not is_rel or conf < RELEVANCE_THRESHOLD:
        log(f"[ai_threshold_skip] job_id={job_id} file_id={file_id} is_rel={is_rel} conf={conf} threshold={RELEVANCE_THRESHOLD}")
        update_file_result(conn, file_id, {"accepted": False, "ai": ai, "reason": "below_threshold", "threshold": RELEVANCE_THRESHOLD})
        log_decision(conn, job_id, file_id, "threshold", f"skip: is_relevant={is_rel} conf={conf} < RELEVANCE_THRESHOLD={RELEVANCE_THRESHOLD}")
        finish_success(conn, job_id, file_id, "ai_not_relevant")
        return

    try:
        extraction_outcome.upload_images()
        replace_file_images(conn, file_id, extraction.images)
    except Exception as exc:
        update_file_result(conn, file_id, {"accepted": False, "error": f"image_upload_failed: {exc}"})
        finish_error(conn, job_id, file_id, "error_image_upload", f"image_upload_failed: {exc}")
        return

    log(f"[chunking_start] job_id={job_id} file_id={file_id} len={len(clean)} max_chunks={MAX_CHUNKS} overlap={OVERLAP}")
    log_decision(conn, job_id, file_id, "stage_chunking", "chunking text")
    safe_log(conn, job_id, file_id, "chunking_request", f"model={OLLAMA_MODEL_CHUNKING} tokens={BRAIN_CHUNK_TOKENS} overlap_chars={OVERLAP} max_chunks={MAX_CHUNKS} chars={len(clean)}")
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
        safe_log(conn, job_id, file_id, "chunking_meta", json.dumps(chunk_meta, ensure_ascii=False)[:500])
        log_debug(f"[chunking_meta] job_id={job_id} file_id={file_id} llm={chunk_meta.get('llm_info')} fallback={chunk_meta.get('fallback_used')}")

    if not chunk_texts and extraction.chunks:
        chunk_texts = [c.text for c in extraction.chunks]
        chunk_source = "docling_fallback"
        safe_log(conn, job_id, file_id, "chunking_fallback", "used docling chunks due to empty LLM output")

    log(f"[add_context_start] job_id={job_id} file_id={file_id} chunks={len(chunk_texts)}")
    log_decision(conn, job_id, file_id, "stage_context", "adding context")
    chunks = enrich_chunks_with_context(
        document=clean,
        chunks=chunk_texts,
        ollama_host=OLLAMA_HOST,
        model=OLLAMA_MODEL_CONTEXT,
        timeout=BRAIN_REQUEST_TIMEOUT,
        debug=DEBUG,
    )

    errors = []
    for idx, chunk in enumerate(chunks):
        if chunk.error:
            errors.append(f"chunk {idx} error: {chunk.error}")

    if errors:
        update_file_result(conn, file_id, {"accepted": False, "errors": errors})
        finish_error(conn, job_id, file_id, "error_chunk_context", "; ".join(errors)[:500])
        return

    try:
        log_decision(conn, job_id, file_id, "stage_embedding", "sending chunks to brain")
        chunk_outcome = []
        for idx, chunk in enumerate(chunks):
            chunk_meta = {
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
                log(f"[brain_post] job_id={job_id} file_id={file_id} chunk={idx}/{len(chunks)} len={len(chunk.text)}")
                log_debug(f"[brain_post_payload] job_id={job_id} file_id={file_id} idx={idx} meta={chunk_meta}")
                safe_log(conn, job_id, file_id, "pre_brain_post", f"chunk={idx}/{len(chunks)} len={len(chunk.text)} url={BRAIN_URL}")
                ok = brain_ingest_text(
                    BRAIN_URL,
                    BRAIN_API_KEY,
                    chunk.text,
                    meta=chunk_meta,
                    collection=BRAIN_COLLECTION,
                    timeout=BRAIN_REQUEST_TIMEOUT,
                    debug=DEBUG,
                    dbg_dir=stage.doc_dbg_dir,
                    dbg_root=Path("./debug"),
                )
                if ok:
                    safe_log(conn, job_id, file_id, "brain_ok", f"chunk={idx}")
                if not ok:
                    errors.append(f"chunk {idx} failed")
            except Exception as e:
                safe_log(conn, job_id, file_id, "brain_err", f"chunk={idx} err={e}")
                log(f"[brain_error] job_id={job_id} file_id={file_id} chunk={idx}/{len(chunks)} err={e}")
                errors.append(f"chunk {idx} error: {e}")
            chunk_outcome.append(chunk_meta)
    except Exception as exc:
        errors.append(str(exc))

    sanitized_images = extraction_outcome.sanitized_images()
    result = {
        "accepted": len(errors) == 0,
        "ai": ai,
        "chunks": len(chunks),
        "docling": {
            "mime": extraction.mime_type,
            "source": extraction_outcome.source,
            "chunks": chunk_outcome if chunk_outcome else [c.meta for c in chunks],
            "images": sanitized_images,
        },
        "images": sanitized_images,
    }
    if errors:
        result["errors"] = errors
        update_file_result(conn, file_id, result)
        finish_error(conn, job_id, file_id, "error_brain_ingest", "; ".join(errors)[:500])
        return

    update_file_result(conn, file_id, result)
    finish_success(conn, job_id, file_id, "vectorized")



def process_one(conn, job_id, file_id, stage: ExtractionStageResult | None = None):
    active_stage = stage or run_extraction_stage(conn, job_id, file_id)
    if active_stage is None:
        return
    run_post_extraction_pipeline(conn, active_stage)


def _start_pipeline(pipeline_executor: ThreadPoolExecutor, stage: ExtractionStageResult):
    def _pipeline_task():
        local_conn = create_db_conn()
        try:
            log(f"[pipeline_start] job_id={stage.job_id} file_id={stage.file_id}")
            try:
                log_decision(
                    local_conn,
                    stage.job_id,
                    stage.file_id,
                    "pipeline_start",
                    "starting relevance->embedding",
                )
            except Exception:
                pass
            process_one(local_conn, stage.job_id, stage.file_id, stage)
        finally:
            try:
                local_conn.close()
            except Exception:
                pass

    return pipeline_executor.submit(_pipeline_task)


def main():
    conn = create_db_conn()
    executor = ThreadPoolExecutor(max_workers=DOCLING_MAX_WORKERS)
    pipeline_executor = ThreadPoolExecutor(max_workers=1)
    futures: dict[Future, tuple[str, str]] = {}
    ready: list[ExtractionStageResult] = []
    pipeline_future: Future | None = None

    def submit_if_slot_available():
        while len(futures) < DOCLING_MAX_WORKERS:
            job_id, file_id = claim_one(conn)
            if not job_id:
                break

            def _task(jid=job_id, fid=file_id):
                local_conn = create_db_conn()
                try:
                    return run_extraction_stage(local_conn, jid, fid)
                finally:
                    try:
                        local_conn.close()
                    except Exception:
                        pass

            futures[executor.submit(_task)] = (job_id, file_id)
            log(f"[docling_dispatch] job_id={job_id} file_id={file_id} active={len(futures)} limit={DOCLING_MAX_WORKERS}")

    log(
        f"[startup] DB={DB_PATH} brain_url={BRAIN_URL} key_present={bool(BRAIN_API_KEY)} "
        f"threshold={RELEVANCE_THRESHOLD} worker_id={WORKER_ID} docling_workers={DOCLING_MAX_WORKERS}"
    )
    try:
        while True:
            cleanup_stale(conn)
            submit_if_slot_available()

            done = [f for f in list(futures) if f.done()]
            for fut in done:
                job_meta = futures.pop(fut)
                try:
                    result = fut.result()
                    if result:
                        ready.append(result)
                except Exception as exc:
                    log(f"[docling_task_error] job={job_meta} err={exc}")

            if pipeline_future and pipeline_future.done():
                try:
                    pipeline_future.result()
                except Exception as exc:
                    log(f"[pipeline_task_error] err={exc}")
                pipeline_future = None

            if pipeline_future is None and ready:
                stage = ready.pop(0)
                pipeline_future = _start_pipeline(pipeline_executor, stage)
                continue

            if futures or ready or pipeline_future:
                time.sleep(IDLE_SLEEP_S)
                continue

            submit_if_slot_available()
            if not futures and not ready and pipeline_future is None:
                log("no queued jobs found - shutting down")
                break
    except KeyboardInterrupt:
        log("stopping...")
    finally:
        executor.shutdown(wait=False)
        pipeline_executor.shutdown(wait=False)
        try:
            conn.close()
        except Exception:
            pass
        log("worker stopped")

if __name__ == "__main__":
    main()
