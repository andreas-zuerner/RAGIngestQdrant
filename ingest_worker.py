
#!/usr/bin/env python3
import html
import importlib
import inspect
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
import unicodedata
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import requests

from helpers import compute_next_review_at, utcnow_iso


class DoclingUnavailableError(RuntimeError):
    """Raised when docling is missing at runtime."""


DOCLING_AVAILABLE = True
TEXT_FALLBACK_EXTENSIONS = {".txt", ".text"}
ODF_FALLBACK_EXTENSIONS = {".odt", ".ods", ".odp", ".odg", ".odf", ".odm"}

try:  # pragma: no cover - import-time wiring
    from docling.document_converter import DocumentConverter

    try:
        from docling.pipeline.standard_pipeline import StandardPipelineConfig
    except Exception:  # pragma: no cover - optional config module
        StandardPipelineConfig = None

    try:
        from docling.chunking import HybridChunker, HybridChunkerConfig
    except Exception:  # pragma: no cover - optional chunking module
        HybridChunker = None
        HybridChunkerConfig = None

except Exception as exc:  # pragma: no cover - executed only without docling
    DocumentConverter = None
    StandardPipelineConfig = None
    HybridChunker = None
    HybridChunkerConfig = None
    DOCLING_AVAILABLE = False
    DOCLING_IMPORT_ERROR = exc
else:
    DOCLING_IMPORT_ERROR = None

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
DB_PATH            = os.environ.get("DB_PATH", "DocumentDatabase/state.db")
BRAIN_URL          = os.environ.get("BRAIN_URL", "http://192.168.177.151:8080").rstrip("/")
BRAIN_API_KEY      = os.environ.get("BRAIN_API_KEY", "change-me")
BRAIN_CHUNK_TOKENS = int(os.environ.get("BRAIN_CHUNK_TOKENS", "400"))
BRAIN_OVERLAP_TOKENS = int(os.environ.get("BRAIN_OVERLAP_TOKENS", "80"))
BRAIN_REQUEST_TIMEOUT = float(os.environ.get("BRAIN_REQUEST_TIMEOUT", "120"))
OLLAMA_HOST        = os.environ.get("OLLAMA_HOST", "http://192.168.177.130:11434")
OLLAMA_MODEL       = os.environ.get("OLLAMA_MODEL", "mistral-small3.2:latest")
RELEVANCE_THRESHOLD= float(os.environ.get("RELEVANCE_THRESHOLD", "0.55"))
MIN_CHARS          = int(os.environ.get("MIN_CHARS", "200"))
MAX_TEXT_CHARS     = int(os.environ.get("MAX_TEXT_CHARS", "100000"))
MAX_CHARS          = int(os.environ.get("MAX_CHARS", "4000"))
OVERLAP            = int(os.environ.get("OVERLAP", "400"))
MAX_CHUNKS         = int(os.environ.get("MAX_CHUNKS", "200"))
ENABLE_OCR         = os.environ.get("ENABLE_OCR", "0") == "1"
DEBUG              = os.environ.get("DEBUG", "0") == "1"
LOCK_TIMEOUT_S     = int(os.environ.get("LOCK_TIMEOUT_S", "600"))
IDLE_SLEEP_S       = float(os.environ.get("IDLE_SLEEP_S", "1.0"))
PDFTOTEXT_TIMEOUT_S = int(os.environ.get("PDFTOTEXT_TIMEOUT_S", "60"))
PDF_OCR_MAX_PAGES   = int(os.environ.get("PDF_OCR_MAX_PAGES", "20"))
PDF_OCR_DPI         = int(os.environ.get("PDF_OCR_DPI", "300"))

DECISION_LOG_ENABLED  = os.environ.get("DECISION_LOG_ENABLED", "1") == "1"
DECISION_LOG_MAX_PER_JOB = int(os.environ.get("DECISION_LOG_MAX_PER_JOB", "50"))

WORKER_ID          = f"{os.uname().nodename}-pid{os.getpid()}"

def log(msg):
    ts = datetime.utcnow().isoformat(timespec='seconds') + "Z"
    print(f"[worker_id] {ts} {msg}", flush=True)

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
        """, (f"-{cutoff} seconds",)
    )
    freed = cur.rowcount if hasattr(cur, "rowcount") else 0
    print(f"{freed}", flush=True)
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
        print(f" Successfully finished file_id={file_id}", flush=True)
    else:
        print("file_id ist leer", flush=True)
    conn.execute("UPDATE jobs SET status='done', locked_at=NULL WHERE job_id=?", (job_id,))
    conn.execute(
        "INSERT INTO decision_log(step, job_id, file_id, detail) VALUES(?,?,?,?)",
        ("done", job_id, file_id, status),
    )
    conn.commit()


def finish_error(conn, job_id, file_id, status: str, msg):
    message = str(msg)[:500]
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


class DoclingIngestor:
    """Wrapper around docling's converter and hybrid chunker."""

    def __init__(self, *, enable_ocr: bool, chunk_size: int, chunk_overlap: int, max_chunks: int):
        if not DOCLING_AVAILABLE:
            raise DoclingUnavailableError(
                "docling is not installed or provides an incompatible API. Install or upgrade it via 'pip install docling' to run the ingest worker"
                + (f" (import error: {DOCLING_IMPORT_ERROR})" if DOCLING_IMPORT_ERROR else "")
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunks = max_chunks
        self.converter = self._build_converter(enable_ocr)
        self.chunker = self._build_chunker(chunk_size, chunk_overlap)

    def _build_converter(self, enable_ocr: bool):
        if StandardPipelineConfig is None:
            return DocumentConverter()
        config = StandardPipelineConfig()
        for attr in ("enable_image_ocr", "enable_ocr"):
            if hasattr(config, attr):
                setattr(config, attr, bool(enable_ocr))
        try:
            return DocumentConverter(pipeline_config=config)
        except TypeError:
            return DocumentConverter()

    def _build_chunker(self, chunk_size: int, chunk_overlap: int):
        if HybridChunker is None:
            return None
        cfg = None
        if HybridChunkerConfig is not None:
            cfg_kwargs = {}
            try:
                sig = inspect.signature(HybridChunkerConfig)
                if "chunk_size" in sig.parameters:
                    cfg_kwargs["chunk_size"] = chunk_size
                if "chunk_overlap" in sig.parameters:
                    cfg_kwargs["chunk_overlap"] = chunk_overlap
                if "overlap" in sig.parameters and "chunk_overlap" not in cfg_kwargs:
                    cfg_kwargs["overlap"] = chunk_overlap
                if "max_chunks" in sig.parameters:
                    cfg_kwargs["max_chunks"] = self.max_chunks
            except (TypeError, ValueError):
                cfg_kwargs = {}
            try:
                cfg = HybridChunkerConfig(**cfg_kwargs)
            except Exception:
                cfg = None
        try:
            return HybridChunker(cfg) if cfg is not None else HybridChunker()
        except Exception:
            return None
            
    def _normalize_word_template(self, file_path: Path, suffix: str) -> Optional[Path]:
        if suffix not in {".dotm", ".dotx"}:
            return None

        try:
            with zipfile.ZipFile(file_path, "r") as zin:
                if "[Content_Types].xml" not in zin.namelist():
                    return None

                content_types = zin.read("[Content_Types].xml")
                replacements = (
                    (
                        b"application/vnd.ms-word.template.macroEnabledTemplate.main+xml",
                        b"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml",
                    ),
                    (
                        b"application/vnd.openxmlformats-officedocument.wordprocessingml.template.main+xml",
                        b"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml",
                    ),
                )
                new_content = content_types
                changed = False
                for old, new in replacements:
                    if old in new_content:
                        new_content = new_content.replace(old, new)
                        changed = True

                if not changed:
                    return None

                tmp_dir = ensure_dir(Path(tempfile.gettempdir()) / "brain-docling-word")
                tmp_path = tmp_dir / f"{slugify(file_path.stem)}.docx"

                with zipfile.ZipFile(tmp_path, "w") as zout:
                    for item in zin.infolist():
                        data = zin.read(item.filename)
                        if item.filename == "[Content_Types].xml":
                            data = new_content
                        zout.writestr(item, data)

                return tmp_path
        except Exception:
            return None

    def extract(self, file_path: Path) -> DoclingExtraction:
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        input_obj = self._build_conversion_input(file_path)
        result = self.converter.convert(input_obj)
        document = (
            getattr(result, "document", None)
            or getattr(result, "output_document", None)
            or getattr(result, "output", None)
        )
        if document is None:
            raise RuntimeError("docling returned no document")
        mime = getattr(result, "media_type", None) or getattr(document, "media_type", None)
        return self._build_extraction(document, mime)

    def extract_markdown(self, markdown_text: str, *, original_name: str = "doc.md") -> DoclingExtraction:
        if not markdown_text or not markdown_text.strip():
            raise RuntimeError("markdown text for docling fallback is empty")
        tmp_dir = ensure_dir(Path(tempfile.gettempdir()) / "brain-docling-markdown")
        tmp_path = tmp_dir / f"{slugify(original_name)}.md"
        tmp_path.write_text(markdown_text, encoding="utf-8")
        result = self.converter.convert(str(tmp_path))
        document = (
            getattr(result, "document", None)
            or getattr(result, "output_document", None)
            or getattr(result, "output", None)
        )
        if document is None:
            raise RuntimeError("docling returned no document from markdown fallback")
        mime = getattr(result, "media_type", None) or getattr(document, "media_type", None) or "text/markdown"
        return self._build_extraction(document, mime)

    def _build_extraction(self, document, mime: Optional[str]) -> DoclingExtraction:
        chunks = self._chunk_document(document)
        if not chunks:
            doc_text = self._document_to_text(document)
            if not doc_text.strip():
                raise RuntimeError("docling produced no textual output")
            chunk = DoclingChunk(text=doc_text, meta={"chunk_index": 1})
            return DoclingExtraction(text=doc_text, mime_type=mime, chunks=[chunk])

        processed: List[DoclingChunk] = []
        for idx, chunk in enumerate(chunks, start=1):
            if self.max_chunks and idx > self.max_chunks:
                break
            chunk_text = self._chunk_to_text(chunk)
            if not chunk_text or not chunk_text.strip():
                continue
            meta = {"chunk_index": idx}
            meta.update(self._page_hint(chunk))
            processed.append(DoclingChunk(text=chunk_text, meta=meta))

        if not processed:
            doc_text = self._document_to_text(document)
            if not doc_text.strip():
                raise RuntimeError("docling produced empty chunks")
            processed.append(DoclingChunk(text=doc_text, meta={"chunk_index": 1}))

        combined_text = self._document_to_text(document, processed)
        if not combined_text.strip():
            combined_text = "\n\n".join(chunk.text for chunk in processed)

        return DoclingExtraction(text=combined_text, mime_type=mime, chunks=processed)

    def _build_conversion_input(self, file_path: Path):
        """
        Build a Docling conversion input for the given file path.

        With newer Docling versions, `DocumentConverter.convert` typically
        accepts a path (string or Path) directly.
        For text and ODF files we first wrap/convert to HTML and write
        a temporary .html file, then pass its path to Docling.
        """
        suffix = file_path.suffix.lower()

        # Normalize Word templates (.dotm/.dotx) to plain .docx by adjusting
        # the content type. The python-docx dependency used by docling rejects
        # the macro/template content types, even though the payload is
        # otherwise compatible with DOCX. We rewrite the Content_Types entry
        # into a temporary copy so docling can ingest the document.
        normalized_word = self._normalize_word_template(file_path, suffix)
        if normalized_word is not None:
            return str(normalized_word)
        
        if suffix in TEXT_FALLBACK_EXTENSIONS:
            html_text = self._wrap_plain_text_html(file_path)
            if html_text:
                return self._input_from_html(html_text, original_name=file_path.name)
        if suffix in ODF_FALLBACK_EXTENSIONS:
            html_text = self._odf_to_html(file_path)
            if html_text:
                return self._input_from_html(html_text, original_name=file_path.name)
        # Standard-Fall: Originaldatei als Quelle verwenden
        return str(file_path)

    def _input_from_html(self, html_text: str, *, original_name: str):
        """
        Writes the given HTML into a temporary file and returns its path
        as a string. This path can be passed directly to Docling's
        DocumentConverter.convert().
        """
        if not html_text:
            return None
        html_text = html_text if isinstance(html_text, str) else str(html_text)
        tmp_dir = ensure_dir(Path(tempfile.gettempdir()) / "brain-docling-html")
        tmp_name = f"{slugify(original_name)}.html"
        tmp_path = tmp_dir / tmp_name
        tmp_path.write_text(html_text, encoding="utf-8")
        return str(tmp_path)

    def _wrap_plain_text_html(self, file_path: Path) -> str:
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""
        escaped = html.escape(text)
        return f"<html><body><pre>{escaped}</pre></body></html>"

    def _odf_to_html(self, file_path: Path) -> str:
        try:
            with zipfile.ZipFile(file_path) as zf:
                content = zf.read("content.xml")
        except Exception:
            return ""
        try:
            root = ET.fromstring(content)
        except Exception:
            return ""
        paragraphs: List[str] = []
        for elem in root.iter():
            local = elem.tag.split("}")[-1]
            if local.lower() in {"p", "h", "h1", "h2", "h3", "h4", "h5", "h6", "title"}:
                text = "".join(elem.itertext()).strip()
                if text:
                    paragraphs.append(f"<p>{html.escape(text)}</p>")
        if not paragraphs:
            fallback = "".join(root.itertext()).strip()
            if fallback:
                paragraphs.append(f"<p>{html.escape(fallback)}</p>")
        if not paragraphs:
            return ""
        body = "\n".join(paragraphs)
        return f"<html><body>{body}</body></html>"

    def _chunk_document(self, document):
        if self.chunker is None:
            return self._fallback_chunk_document(document)
        chunk_method = getattr(self.chunker, "chunk", None) or getattr(self.chunker, "chunk_document", None)
        if not callable(chunk_method):
            return self._fallback_chunk_document(document)
        try:
            chunks = chunk_method(document)
            if chunks:
                return chunks
        except Exception:
            pass
        return self._fallback_chunk_document(document)

    def _fallback_chunk_document(self, document) -> List[DoclingChunk]:
        text = self._document_to_text(document)
        if not text:
            return []
        chunks: List[DoclingChunk] = []
        step = max(self.chunk_size - self.chunk_overlap, 1)
        idx = 0
        for start in range(0, len(text), step):
            if self.max_chunks and idx >= self.max_chunks:
                break
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if not chunk_text:
                continue
            idx += 1
            chunks.append(DoclingChunk(text=chunk_text, meta={"chunk_index": idx}))
            if end >= len(text):
                break
        return chunks

    def _chunk_to_text(self, chunk) -> str:
        for attr in ("text", "content"):
            value = getattr(chunk, attr, None)
            if isinstance(value, str):
                return value
            if callable(value):
                try:
                    out = value()
                    if isinstance(out, str):
                        return out
                except Exception:
                    continue
        if hasattr(chunk, "to_text"):
            try:
                out = chunk.to_text()
                if isinstance(out, str):
                    return out
            except Exception:
                pass
        return str(chunk)

    def _page_hint(self, chunk) -> Dict[str, object]:
        hint: Dict[str, object] = {}
        page_span = getattr(chunk, "page_span", None)
        if page_span:
            for key in ("start", "start_page", "start_page_number"):
                value = getattr(page_span, key, None)
                if isinstance(value, int):
                    hint.setdefault("page_start", value)
                    break
            for key in ("end", "end_page", "end_page_number"):
                value = getattr(page_span, key, None)
                if isinstance(value, int):
                    hint.setdefault("page_end", value)
                    break
        section = getattr(chunk, "section", None)
        if isinstance(section, str) and section.strip():
            hint["section"] = section.strip()
        return hint

    def _document_to_text(self, document, chunks: Optional[List[DoclingChunk]] = None) -> str:
        for attr in ("export_text", "export_to_text", "to_text", "get_text"):
            func = getattr(document, attr, None)
            if callable(func):
                try:
                    text = func()
                    if isinstance(text, str) and text.strip():
                        return text
                except Exception:
                    continue
        if chunks:
            return "\n\n".join(chunk.text for chunk in chunks)
        return ""


_DOCLING_INGESTOR: Optional[DoclingIngestor] = None


def get_docling_ingestor() -> DoclingIngestor:
    global _DOCLING_INGESTOR
    if _DOCLING_INGESTOR is None:
        _DOCLING_INGESTOR = DoclingIngestor(
            enable_ocr=ENABLE_OCR,
            chunk_size=MAX_CHARS,
            chunk_overlap=OVERLAP,
            max_chunks=MAX_CHUNKS,
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
    """Verarbeitet genau einen Job end-to-end (extract -> score -> chunk -> brain)."""
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

    # 1) Pfad aus files
    row = conn.execute("SELECT path FROM files WHERE id=?", (file_id,)).fetchone()
    if not row:
        finish_error(conn, job_id, file_id, "error_missing_db_record", "file_id not found in files")
        return
    path = _get(row, "path", _get(row, 0))
    if not path:
        finish_error(conn, job_id, file_id, "error_missing_path", "path column missing/empty")
        return
    p = Path(path)
    safe_log(conn, job_id, file_id, "path_ok", str(p))

    # Snapshot der Gates/ENV
    try:
        gate_msg = (
            f"MIN_CHARS={MIN_CHARS} "
            f"RELEVANCE_THRESHOLD={RELEVANCE_THRESHOLD} "
            f"MAX_TEXT_CHARS={MAX_TEXT_CHARS} "
            f"MAX_CHUNKS={MAX_CHUNKS} OVERLAP={OVERLAP} "
            f"OLLAMA_HOST={'set' if OLLAMA_HOST else 'unset'} "
            f"OLLAMA_MODEL={OLLAMA_MODEL!s} "
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

    # 3) Text extrahieren via docling (+ PDF-OCR-Fallback)
    size = p.stat().st_size
    try:
        ingestor = get_docling_ingestor()
    except DoclingUnavailableError as e:
        finish_error(conn, job_id, file_id, "error_missing_dependency", str(e))
        return

    extraction: Optional[DoclingExtraction] = None
    docling_error: Optional[Exception] = None
    docling_source = "docling"
    try:
        extraction = ingestor.extract(p)
    except Exception as e:
        docling_error = e

    if extraction is None and p.suffix.lower() == ".pdf":
        safe_log(conn, job_id, file_id, "docling_pdf_fallback", str(docling_error))
        try:
            fallback_text = legacy_pdf_fallback_text(p)
            if not fallback_text.strip():
                raise RuntimeError("legacy_pdf_fallback produced no text")
            try:
                extraction = ingestor.extract_markdown(fallback_text, original_name=p.name)
                docling_source = "pdf_ocr_fallback"
                log_decision(
                    conn,
                    job_id,
                    file_id,
                    "docling_fallback",
                    f"pdf_ocr_fallback len={len(fallback_text)} original_error={docling_error}",
                )
            except Exception as markdown_error:
                # If Docling cannot process the markdown fallback, fall back to a
                # minimal extraction that bypasses Docling entirely so that we
                # can still ingest text obtained via pdftotext/ocr.
                chunk = DoclingChunk(text=fallback_text, meta={"chunk_index": 1})
                extraction = DoclingExtraction(
                    text=fallback_text,
                    mime_type="text/plain",
                    chunks=[chunk],
                )
                docling_source = "pdf_text_fallback"
                log_decision(
                    conn,
                    job_id,
                    file_id,
                    "docling_fallback",
                    f"pdf_text_fallback len={len(fallback_text)} original_error={docling_error}; markdown_error={markdown_error}",
                )
        except Exception as e:
            if docling_error is None:
                docling_error = e
            else:
                docling_error = RuntimeError(f"{docling_error}; pdf_fallback_failed: {e}")

    if extraction is None:
        update_file_result(
            conn,
            file_id,
            {"accepted": False, "skipped": "extract_failed", "error": str(docling_error)},
        )
        finish_error(conn, job_id, file_id, "error_extract_failed", f"docling_failed: {docling_error}")
        return

    clean = (extraction.text or "").strip()
    detected = extraction.mime_type or "docling"
    log_decision(conn, job_id, file_id, "sniff", f"mime={detected}; size={size}; source={docling_source}")
    log_decision(
        conn,
        job_id,
        file_id,
        "extract_ok",
        f"chars={len(clean)} docling_chunks={len(extraction.chunks)} source={docling_source}",
    )

    # 4) Zu kurz? -> freundlich beenden
    if not clean or len(clean) < MIN_CHARS:
        update_file_result(conn, file_id, {"accepted": False, "skipped": "too_short", "chars": len(clean)})
        log_decision(conn, job_id, file_id, "threshold", f"skip: too_short chars={len(clean)} < MIN_CHARS={MIN_CHARS}")
        finish_success(conn, job_id, file_id, "skipped_too_short")
        return

    # 5) Sehr lange Texte kappen
    if len(clean) > MAX_TEXT_CHARS:
        clean = clean[:MAX_TEXT_CHARS]
        log_decision(conn, job_id, file_id, "threshold", f"cap: truncated to {MAX_TEXT_CHARS} chars")

    # 6) AI-Scoring via Ollama (falls konfiguriert)
    try:
        if not OLLAMA_HOST:
            safe_log(conn, job_id, file_id, "no_ollama", "OLLAMA_HOST not set -> skipping AI score")
            raise RuntimeError("OLLAMA disabled")
        safe_log(conn, job_id, file_id, "pre_ai", f"len={len(clean)} host={OLLAMA_HOST} model={OLLAMA_MODEL}")
        ai = ai_score_text(OLLAMA_HOST, OLLAMA_MODEL, clean, timeout=600, debug=DEBUG, dbg_root=Path("./debug"))
        is_rel = bool(ai.get("is_relevant"))
        conf   = float(ai.get("confidence", 0.0))
        safe_log(conn, job_id, file_id, "score", f"is_rel={is_rel} conf={conf}")
    except Exception as e:
        update_file_result(conn, file_id, {"accepted": False, "error": f"ai_scoring_failed: {e}"})
        finish_error(conn, job_id, file_id, "error_ai_scoring", f"ai_scoring_failed: {e}")
        return

    # 7) Gate prüfen
    if not is_rel or conf < RELEVANCE_THRESHOLD:
        update_file_result(conn, file_id, {"accepted": False, "ai": ai, "reason": "below_threshold", "threshold": RELEVANCE_THRESHOLD})
        log_decision(conn, job_id, file_id, "threshold", f"skip: is_relevant={is_rel} conf={conf} < RELEVANCE_THRESHOLD={RELEVANCE_THRESHOLD}")
        finish_success(conn, job_id, file_id, "ai_not_relevant")
        return

    # 8) Chunking durch docling
    docling_chunks = extraction.chunks
    log_decision(
        conn,
        job_id,
        file_id,
        "chunk_plan",
        f"chunks={len(docling_chunks)} via docling hybrid overlap={OVERLAP}",
    )

    # 9) Brain-Ingest pro Chunk
    errors = []
    for idx, chunk in enumerate(docling_chunks, 1):
        chunk_meta = {
            "source": "ct108",
            "path": str(p),
            "chunk_index": chunk.meta.get("chunk_index", idx),
            "chunks_total": len(docling_chunks),
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
            safe_log(
                conn,
                job_id,
                file_id,
                "pre_brain_post",
                f"chunk={idx}/{len(docling_chunks)} len={len(chunk.text)} url={BRAIN_URL}",
            )
            ok = brain_ingest_text(
                BRAIN_URL,
                BRAIN_API_KEY,
                chunk.text,
                meta=chunk_meta,
                chunk_tokens=BRAIN_CHUNK_TOKENS,
                overlap_tokens=BRAIN_OVERLAP_TOKENS,
                timeout=BRAIN_REQUEST_TIMEOUT,
            )
            if ok:
                safe_log(conn, job_id, file_id, "brain_ok", f"chunk={idx}")
            if not ok:
                errors.append(f"chunk {idx} failed")
        except Exception as e:
            safe_log(conn, job_id, file_id, "brain_err", f"chunk={idx} err={e}")
            errors.append(f"chunk {idx} error: {e}")

    # 10) Ergebnis persistieren
    result = {
        "accepted": len(errors) == 0,
        "ai": ai,
        "chunks": len(docling_chunks),
        "docling": {
            "mime": extraction.mime_type,
            "source": docling_source,
            "chunks": [chunk.meta for chunk in docling_chunks],
        },
    }
    if errors:
        result["errors"] = errors
        update_file_result(conn, file_id, result)
        finish_error(conn, job_id, file_id, "error_brain_ingest", "; ".join(errors)[:500])
        return

    update_file_result(conn, file_id, result)
    finish_success(conn, job_id, file_id, "vectorized")


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

    log(f"DB={DB_PATH} brain_url={BRAIN_URL} key_present={bool(BRAIN_API_KEY)} threshold={RELEVANCE_THRESHOLD}")
    try:
        while True:
            # print(f"Start while", flush=True)
            cleanup_stale(conn)
            # print(f"cleanup", flush=True)
            job_id, file_id = claim_one(conn)
            print(f"Starting job_id={job_id}, file_id={file_id}", flush=True)
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
