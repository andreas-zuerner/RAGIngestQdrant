#!/usr/bin/env python3
"""Text extraction helpers for ingest_worker.

This module encapsulates all logic for preparing documents for docling-serve,
handling synchronous and asynchronous endpoints, and extracting text and
images. Logging and debugging is handled locally so ingest_worker only needs to
call :func:`extract_document` and work with the returned data.
"""

from __future__ import annotations

import base64
import importlib
import json
import logging
import mimetypes
import os
import posixpath
import re
import shutil
import subprocess
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager, nullcontext

import requests
from nextcloud_client import NextcloudClient, env_client

import initENV


class DoclingUnavailableError(RuntimeError):
    """Raised when docling is missing at runtime."""


class ExtractionFailed(RuntimeError):
    """Raised when the extraction pipeline cannot produce text."""


CONVERT_BEFORE_DOCLING = {".xls", ".doc", ".odf", ".odp", ".odt", ".odg", ".ods", ".odm"}
SUPPORTED_EXTENSIONS = {
    ".docx",
    ".dotx",
    ".docm",
    ".dotm",
    ".pptx",
    ".potx",
    ".ppsx",
    ".pptm",
    ".potm",
    ".ppsm",
    ".pdf",
    ".md",
    ".html",
    ".htm",
    ".xhtml",
    ".xml",
    ".nxml",
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
    ".adoc",
    ".asciidoc",
    ".asc",
    ".csv",
    ".xlsx",
    ".xlsm",
    ".txt",
    ".json",
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".flac",
    ".mp4",
    ".avi",
    ".mov",
    ".vtt",
}
SUPPORTED_EXTENSIONS |= CONVERT_BEFORE_DOCLING


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


def convert_for_docling(file_path: Path) -> Tuple[Path, Optional[Path]]:
    """
    Convert office-style documents to PDF before sending them to docling-serve.

    Returns a tuple of (path_to_use, temp_dir_to_cleanup). When conversion fails
    or is not required, the original path is returned and the cleanup path is None.
    """

    ext = file_path.suffix.lower()
    if ext not in CONVERT_BEFORE_DOCLING:
        return file_path, None

    soffice_candidates = [shutil.which("soffice"), shutil.which("libreoffice")]
    soffice = next((c for c in soffice_candidates if c), None)
    if soffice is None:
        log(
            f"[convert_skip_no_soffice] path={file_path} ext={ext} "
            f"PATH={os.environ.get('PATH')} candidates={soffice_candidates}"
        )
        return file_path, None

    temp_dir: Optional[Path] = None
    try:
        temp_dir = Path(tempfile.mkdtemp(prefix="docling-convert-"))
        cmd = [
            soffice,
            "--headless",
            "--nologo",
            "--nolockcheck",
            "--nodefault",
            "--norestore",
            "--invisible",
            "--convert-to",
            "pdf:writer_pdf_Export",
            "--outdir",
            str(temp_dir),
            str(file_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if proc.returncode != 0:
            log(
                f"[convert_failed] source={file_path} rc={proc.returncode} "
                f"stdout={proc.stdout.strip()!r} stderr={proc.stderr.strip()!r}"
            )
            shutil.rmtree(temp_dir, ignore_errors=True)
            return file_path, None

        candidate = temp_dir / f"{file_path.stem}.pdf"
        if not candidate.exists():
            pdfs = list(temp_dir.glob("*.pdf"))
            candidate = pdfs[0] if pdfs else candidate

        if candidate.exists():
            log(f"[convert_success] source={file_path} target={candidate}")
            return candidate, temp_dir

        log(f"[convert_missing_output] source={file_path} dir={temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return file_path, None
    except Exception as exc:
        log(f"[convert_error] source={file_path} err={exc}")
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return file_path, None


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


_LOGGER: Optional[logging.Logger] = None


def _configure_file_logger() -> Optional[logging.Logger]:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log_path = LOG_PATH.resolve()
        logger = logging.getLogger("scan")
        if not any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", None) == str(log_path)
            for h in logger.handlers
        ):
            logger.setLevel(logging.DEBUG if initENV.DEBUG else logging.INFO)
            handler = logging.FileHandler(log_path, encoding="utf-8")
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)s] [worker] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(handler)
        logger.propagate = False
        return logger
    except Exception:
        return None


_LOGGER = _configure_file_logger()

WORKER_ID = f"{os.uname().nodename}-pid{os.getpid()}"

def log(msg: str):
    if _LOGGER is None:
        return
    try:
        _LOGGER.info(f"[{WORKER_ID}] {msg}")
    except Exception:
        pass


def log_debug(msg: str):
    if initENV.DEBUG:
        log(msg)


def _log_poll_progress(task_state, msg: str):
    """Log async poll progress even when DEBUG is off (first + every 5th attempt)."""

    attempts = getattr(task_state, "attempts", 0)
    if initENV.DEBUG or attempts in {1, 5} or attempts % 5 == 0:
        log(msg)


def sanitize_images(images: List[Dict[str, object]]) -> List[Dict[str, object]]:
    sanitized = []
    for img in images or []:
        clean = {k: v for k, v in img.items() if k not in {"data"}}
        sanitized.append(clean)
    return sanitized


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
    slug: str


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
class ExtractionOutcome:
    extraction: DoclingExtraction
    source: str
    ingestor: "DoclingServeIngestor"
    debug_dir: Path | None

    def upload_images(self):
        self.ingestor.upload_images(self.extraction)

    def sanitized_images(self) -> List[Dict[str, object]]:
        return sanitize_images(self.extraction.images)

    def debug_dump(self, chunk_texts: Optional[List[str]] = None):
        if not initENV.DEBUG:
            return
        debug_dir = self.debug_dir or Path("logs") / "docling"
        try:
            self.ingestor._write_debug_dump(
                self.extraction.text,
                self.extraction.images,
                debug_dir,
                chunk_texts=chunk_texts,
            )
        except Exception as exc:
            log(f"[docling_debug_write_failed] err={exc}")


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
        async_url: str | None = None,
        async_enabled: bool = False,
        async_timeout: float = 900.0,
        poll_interval: float = 5.0,
        async_slot_provider=None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunks = max_chunks
        self.service_url = service_url.rstrip("/")
        self.async_url = (async_url or "").rstrip("/")
        self.async_enabled = async_enabled
        self.async_timeout = async_timeout
        self.poll_interval = poll_interval
        self.async_slot_provider = async_slot_provider
        self.image_dir = ensure_dir(Path(image_dir)) if nextcloud_client is None else Path(image_dir)
        self.nextcloud_client = nextcloud_client
        self.remote_image_dir = remote_image_dir or str(image_dir)

    def extract(
        self,
        file_path: Path,
        *,
        upload_images: bool = True,
        debug: bool = False,
        debug_dir: Path | None = None,
    ) -> DoclingExtraction:
        if not file_path.exists():
            raise FileNotFoundError(file_path)

        payload: Dict[str, object] | None = None
        async_error: Exception | None = None
        if self.async_enabled:
            try:
                payload = self._extract_async(file_path)
            except Exception as exc:
                async_error = exc
                log(f"[docling_async_error] path={file_path} err={exc}")

        if payload is None:
            if async_error:
                log(f"[docling_async_fallback_sync] path={file_path} err={async_error}")
            try:
                payload = self._extract_sync(file_path)
            except Exception as exc:
                if async_error:
                    raise RuntimeError(
                        f"sync extraction failed after async failure: {async_error}"
                    ) from exc
                raise

        text = self._extract_text(payload)
        base_slug = slugify(file_path.stem)
        effective_debug_dir = debug_dir or Path("logs") / "docling"
        request_info = {
            "service_url": self.service_url,
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size if file_path.exists() else None,
        }

        text, inline_images = self._extract_inline_images(
            text, base_slug, upload_images=upload_images, debug_dir=effective_debug_dir if debug else None
        )
        images_payload = payload.get("images") or payload.get("media", {}).get("images") or []
        stored_images = self._store_images(
            images_payload,
            base_slug,
            upload=upload_images,
            debug_dir=effective_debug_dir if debug else None,
        )
        if inline_images:
            stored_images.extend(inline_images)

        text_with_refs = self._inject_image_refs(text, stored_images)
        text_for_chunks = self._strip_extracted_images_section(text_with_refs)
        mime = payload.get("media_type") or payload.get("mime_type")
        chunks = self._chunk_text(text_for_chunks)
        if debug:
            self._write_debug_dump(
                text_for_chunks,
                stored_images,
                effective_debug_dir,
                payload=payload,
                request_info=request_info,
            )
        return DoclingExtraction(
            text=text_for_chunks,
            mime_type=mime,
            chunks=chunks,
            images=stored_images,
            slug=base_slug,
        )

    def _extract_sync(self, file_path: Path) -> Dict[str, object]:
        try:
            with file_path.open("rb") as fp:
                files = {"files": (file_path.name, fp, "application/octet-stream")}
                data = {}
                if ENABLE_OCR:
                    data["force_ocr"] = "true"
                response = requests.post(
                    self.service_url, files=files, data=data, timeout=DOCLING_SERVE_TIMEOUT
                )
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

        self._ensure_no_docling_error(payload)
        return payload

    def _extract_async(self, file_path: Path) -> Dict[str, object]:
        submit_url = self.async_url or self._derive_async_url(self.service_url)
        start = time.time()
        deadline = start + self.async_timeout
        with self._acquire_async_slot(file_path.name, deadline) as task_state:
            try:
                with file_path.open("rb") as fp:
                    files = {"files": (file_path.name, fp, "application/octet-stream")}
                    data = {}
                    if ENABLE_OCR:
                        data["force_ocr"] = "true"
                    response = requests.post(
                        submit_url,
                        files=files,
                        data=data,
                        timeout=min(DOCLING_SERVE_TIMEOUT, self.async_timeout),
                    )
                response.raise_for_status()
            except requests.HTTPError as exc:
                detail = self._response_detail(exc.response)
                suffix = f"; {detail}" if detail else ""
                raise DoclingUnavailableError(f"docling-serve async submit failed: {exc}{suffix}") from exc
            except Exception as exc:
                raise DoclingUnavailableError(f"docling-serve async submit failed: {exc}") from exc

            try:
                submission = response.json()
            except Exception as exc:
                raise RuntimeError("docling-serve async submit returned invalid JSON") from exc

            # Lightweight instrumentation: helps to verify whether polling is reached.
            if isinstance(submission, dict):
                _job_id = submission.get("job_id") or submission.get("task_id") or submission.get("id")
                _poll_hint = (
                    submission.get("status_url") or submission.get("poll_url") or submission.get("result_url")
                )
                log(f"[docling_async_submit_ok] url={submit_url} job_id={_job_id} keys={list(submission.keys())}")
                log(
                    "[docling_async_submit_payload_probe] "
                    f"has_document={'document' in submission} has_result={'result' in submission} poll_hint={_poll_hint}"
                )
            else:
                log(f"[docling_async_submit_ok] url={submit_url} type={type(submission).__name__}")

            self._ensure_no_docling_error(submission)
            payload = self._maybe_extract_payload(submission)
            if payload:
                return payload

            job_id = submission.get("job_id") or submission.get("task_id") or submission.get("id")

            api_root = self.service_url
            # alles hinter /v1/convert/... abschneiden
            m = re.split(r"/v1/convert/.*$", self.service_url, maxsplit=1)
            if m:
                api_root = m[0]

            poll_url = (
                submission.get("result_url")
                or submission.get("status_url")
                or submission.get("poll_url")
                or (f"{api_root}/v1/result/{job_id}" if job_id else None)
                or (f"{api_root}/v1/status/poll/{job_id}" if job_id else None)
            )


            if not poll_url:
                raise RuntimeError("docling-serve async response did not include a poll URL")

            max_interval = min(self.async_timeout * 0.3, 300)
            next_interval = self.poll_interval
            last_error: Exception | None = None

            log(
                f"[docling_async_poll_start] url={poll_url} initial_interval={self.poll_interval}s "
                f"timeout={self.async_timeout}s"
            )

            while True:
                now = time.time()
                # Ensure the very first poll respects the configured poll interval
                sleep_for = 0 if now > deadline else next_interval
                # Cap the backoff once 30% of the timeout is reached, but never exceed 300s
                if sleep_for > max_interval and max_interval > 0:
                    sleep_for = max_interval
                if sleep_for > 0:
                    time.sleep(sleep_for)

                task_state.attempts += 1
                _log_poll_progress(
                    task_state,
                    f"[docling_async_poll_attempt] url={poll_url} attempt={task_state.attempts} "
                    f"next_interval={round(next_interval, 2)}s",
                )
                try:
                    poll_response = requests.get(poll_url, timeout=DOCLING_SERVE_TIMEOUT)
                    if poll_response.status_code in (202, 404):
                        if time.time() > deadline:
                            break

                        next_interval = min(
                            next_interval * 2, max_interval if max_interval > 0 else next_interval
                        )
                        task_state.next_interval = next_interval

                        _log_poll_progress(
                            task_state,
                            f"[docling_async_poll_pending] url={poll_url} status={poll_response.status_code} "
                            f"attempt={task_state.attempts} next_interval={round(next_interval, 2)}s",
                        )

                        time.sleep(next_interval)   # <-- DAS fehlt/ist entscheidend
                        task_state.attempts += 1    # <-- falls nicht bereits am Loop-Anfang gemacht
                        continue

                    poll_response.raise_for_status()
                except Exception as exc:
                    last_error = exc
                    task_state.last_error = exc
                    raise DoclingUnavailableError(
                        f"docling-serve async poll failed after {int(time.time() - start)}s: {exc}"
                    ) from exc
                else:
                    try:
                        poll_data = poll_response.json()
                    except Exception as exc:
                        raise RuntimeError("docling-serve async poll returned invalid JSON") from exc

                    self._ensure_no_docling_error(poll_data)
                    payload = self._maybe_extract_payload(poll_data)
                    if payload:
                        return payload

                    status = str(poll_data.get("status") or poll_data.get("state") or "").lower()
                    if status in {"failed", "error"}:
                        detail = poll_data.get("detail") or poll_data.get("message")
                        suffix = f"; {detail}" if detail else ""
                        raise DoclingUnavailableError(f"docling-serve async failed{suffix}")

                if time.time() > deadline:
                    break

                next_interval = min(next_interval * 2, max_interval if max_interval > 0 else next_interval)
                task_state.next_interval = next_interval
                _log_poll_progress(
                    task_state,
                    f"[docling_async_poll_retry] url={poll_url} attempt={task_state.attempts} "
                    f"next_interval={round(next_interval, 2)}s",
                )

            suffix = f"; last_error={last_error}" if last_error else ""
            raise DoclingUnavailableError(
                f"docling-serve async timed out after {self.async_timeout}s while polling {poll_url}{suffix}"
            )

    @contextmanager
    def _acquire_async_slot(self, file_name: str, deadline: float):
        if self.async_slot_provider is None:
            default_state = DoclingAsyncTaskState(
                task_id=f"{file_name}-{time.time_ns()}",
                file_name=file_name,
                start_time=time.time(),
                deadline=deadline,
                next_interval=self.poll_interval,
            )
            with nullcontext(default_state) as state:
                yield state
            return

        with self.async_slot_provider(file_name, deadline, self.poll_interval) as state:
            yield state

    def _maybe_extract_payload(self, payload: Dict[str, object]) -> Optional[Dict[str, object]]:
        """Return a *final* extraction payload if present, otherwise None.

        Important: some async APIs return an early envelope that may already contain a
        "document" key but without any extracted content. Treating that as final would
        silently skip polling.
        """
        if not isinstance(payload, dict):
            return None
        if payload.get("error"):
            detail = payload.get("detail") or payload.get("message") or payload.get("error")
            raise DoclingUnavailableError(f"docling-serve reported error: {detail}")

        doc = payload.get("document")
        if isinstance(doc, dict):
            # Only treat as final when actual content is present
            for key in ("md_content", "text_content", "html_content", "json_content"):
                value = doc.get(key)
                if isinstance(value, str) and value.strip():
                    return payload
            media = payload.get("media") or {}
            if isinstance(media, dict) and media.get("images"):
                return payload

        result = payload.get("result")
        if isinstance(result, dict) and (result.get("document") or result.get("media")):
            self._ensure_no_docling_error(result)
            return result
        return None

    def _ensure_no_docling_error(self, payload: Dict[str, object]):
        if not isinstance(payload, dict):
            return
        status = str(payload.get("status") or payload.get("state") or "").lower()
        if status in {"failed", "error"}:
            detail = payload.get("detail") or payload.get("message") or payload.get("error")
            raise DoclingUnavailableError(
                f"docling-serve reported {status}{f'; {detail}' if detail else ''}"
            )
        error_msg = payload.get("error")
        if error_msg:
            detail = payload.get("detail") or payload.get("message")
            suffix = f"; {detail}" if detail else ""
            raise DoclingUnavailableError(f"docling-serve reported error: {error_msg}{suffix}")

    def _derive_async_url(self, service_url: str) -> str:
        """Best-effort async endpoint derivation.

        Prefer setting DOCLING_SERVE_ASYNC_URL explicitly. If we must derive, keep the
        original path and append `/async` (common for `/.../file` style endpoints).
        """
        return f"{service_url.rstrip('/')}/async"

    def _extract_text(self, payload: Dict[str, object]) -> str:
        doc = payload.get("document")
        if isinstance(doc, dict):
            for key in ("md_content", "text_content", "html_content", "json_content"):
                value = doc.get(key)
                if isinstance(value, str) and value.strip():
                    return value

        for key in ("text", "markdown", "md", "content"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value

        detail = self._payload_detail(payload)
        suffix = f"; {detail}" if detail else ""
        raise RuntimeError(f"docling-serve response did not contain textual content{suffix}")

    def _store_images(
        self,
        images_payload,
        base_slug: str,
        *,
        upload: bool = True,
        debug_dir: Path | None = None,
    ) -> List[Dict[str, object]]:
        stored: List[Dict[str, object]] = []
        if not images_payload:
            return stored

        items = images_payload
        if isinstance(images_payload, dict):
            items = [{"name": name, "data": data} for name, data in images_payload.items()]

        remote_base = posixpath.normpath(f"/{self.remote_image_dir}/{base_slug}")

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
            reference = posixpath.normpath(f"{remote_base}/{fname}")
            uploaded = False
            target = ensure_dir(self.image_dir / base_slug) / fname
            try:
                if self.nextcloud_client:
                    if upload:
                        log(f"[image_upload] pushing {fname} to Nextcloud {reference}")
                        self.nextcloud_client.upload_bytes(reference, raw)
                        uploaded = True
                else:
                    target = ensure_dir(self.image_dir / base_slug) / fname
                    if upload:
                        target.write_bytes(raw)
                        reference = str(target)
                        uploaded = True
            except Exception as exc:
                log(f"[image_store_failed] {target}: {exc}")
                continue

            stored.append(
                {
                    "label": fname,
                    "path": reference,
                    "reference": reference,
                    "mime": mime or mimetypes.guess_type(fname)[0] or "application/octet-stream",
                    "data": raw,
                    "uploaded": uploaded,
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

    def _extract_inline_images(self, text: str, base_slug: str, *, upload_images: bool, debug_dir: Path | None) -> Tuple[str, List[Dict[str, object]]]:
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

        stored_images = (
            self._store_images(items, base_slug, upload=upload_images, debug_dir=debug_dir)
            if items
            else []
        )
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
        cleaned_text = "".join(cleaned_parts)
        return cleaned_text, stored_images

    def _inject_image_refs(self, text: str, images: List[Dict[str, object]]) -> str:
        if not text or not images:
            return text
        lines = [text.rstrip(), "\n\n---\n\n", "## Extracted Images\n"]
        for img in images:
            label = img.get("label") or img.get("name") or img.get("id") or "image"
            ref = img.get("reference") or img.get("path") or label
            lines.append(f"- ![{label}]({ref})")
        return "\n".join(lines)

    def _strip_extracted_images_section(self, text: str) -> str:
        if not text:
            return ""
        marker = "## Extracted Images"
        idx = text.find(marker)
        return text[:idx].rstrip() if idx != -1 else text

    def _chunk_text(self, text: str) -> List[DoclingChunk]:
        if not text:
            return []
        parts: List[str] = []
        current = []
        for line in text.splitlines():
            if sum(len(l) for l in current) + len(line) + 1 > self.chunk_size + self.chunk_overlap:
                parts.append("\n".join(current).strip())
                current = []
            current.append(line)
        if current:
            parts.append("\n".join(current).strip())

        limited = parts[: self.max_chunks]
        chunks: List[DoclingChunk] = []
        for idx, part in enumerate(limited, 1):
            meta = {"chunk_index": idx, "chunks_total": len(limited)}
            chunks.append(DoclingChunk(text=part, meta=meta))
        return chunks

    def _write_debug_dump(
        self,
        text: str,
        images: List[Dict[str, object]],
        debug_dir: Path,
        *,
        payload: Dict[str, object] | None = None,
        request_info: Dict[str, object] | None = None,
        chunk_texts: List[str] | None = None,
    ):
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
            (debug_dir / "text.md").write_text(text or "", encoding="utf-8")
            (debug_dir / "images.json").write_text(json.dumps(sanitize_images(images), indent=2), encoding="utf-8")
            if payload:
                (debug_dir / "docling_response.json").write_text(
                    json.dumps(payload, indent=2), encoding="utf-8"
                )
            if request_info:
                (debug_dir / "docling_request.json").write_text(
                    json.dumps(request_info, indent=2), encoding="utf-8"
                )
            if chunk_texts:
                chunk_dump = [
                    {
                        "index": idx,
                        "length": len(text or ""),
                        "text": text,
                    }
                    for idx, text in enumerate(chunk_texts, start=1)
                ]
                (debug_dir / "chunks.json").write_text(
                    json.dumps(chunk_dump, indent=2), encoding="utf-8"
                )
            if images:
                image_dir = debug_dir / "images"
                image_dir.mkdir(parents=True, exist_ok=True)
                for img in images:
                    try:
                        label = img.get("label") or "image"
                        data = img.get("data")
                        if isinstance(data, bytes):
                            (image_dir / label).write_bytes(data)
                    except Exception as exc:
                        log(f"[docling_debug_image_write_failed] image={img.get('label')} err={exc}")
        except Exception as exc:
            log(f"[docling_debug_write_failed] err={exc}")

    def _response_detail(self, response) -> str:
        try:
            if response is not None and response.text:
                return response.text.strip()
        except Exception:
            return ""
        return ""

    def _payload_detail(self, payload) -> str:
        try:
            return json.dumps(payload)[:500]
        except Exception:
            return ""

    def upload_images(self, extraction: DoclingExtraction):
        if not extraction.images:
            return
        for img in extraction.images:
            if img.get("uploaded"):
                continue
            remote_base = posixpath.normpath(f"/{self.remote_image_dir}/{extraction.slug}")
            reference = posixpath.normpath(f"{remote_base}/{img['label']}")
            data = img.get("data")
            if not data:
                continue
            try:
                if self.nextcloud_client:
                    log(f"[image_upload] pushing {img['label']} to Nextcloud {reference}")
                    self.nextcloud_client.upload_bytes(reference, data)
                else:
                    target = ensure_dir(self.image_dir / extraction.slug) / img["label"]
                    target.write_bytes(data)
                    reference = str(target)
                img["uploaded"] = True
                img["reference"] = reference
            except Exception as exc:
                log(f"[image_upload_failed] {img['label']}: {exc}")


def get_docling_ingestor(async_slot_provider=None) -> DoclingServeIngestor:
    async_url = initENV.DOCLING_SERVE_ASYNC_URL or None
    if not async_url and initENV.DOCLING_SERVE_USE_ASYNC:
        async_url = None  # derive from service_url
    nextcloud_client = None
    try:
        if initENV.NEXTCLOUD_TOKEN:
            nextcloud_client = env_client()
    except Exception as exc:
        log(f"[nextcloud_client_unavailable] err={exc}")

    log(f"[docling_cfg] use_async={initENV.DOCLING_SERVE_USE_ASYNC} "
    f"service_url={initENV.DOCLING_SERVE_URL} "
    f"async_url={initENV.DOCLING_SERVE_ASYNC_URL} "
    f"poll_interval={initENV.DOCLING_SERVE_ASYNC_POLL_INTERVAL} "
    f"timeout={initENV.DOCLING_SERVE_ASYNC_TIMEOUT}")

    return DoclingServeIngestor(
        chunk_size=initENV.MAX_CHARS,
        chunk_overlap=initENV.OVERLAP,
        max_chunks=initENV.MAX_CHUNKS,
        service_url=initENV.DOCLING_SERVE_URL,
        image_dir=Path(initENV.NEXTCLOUD_IMAGE_DIR),
        nextcloud_client=nextcloud_client,
        remote_image_dir=initENV.NEXTCLOUD_IMAGE_DIR,
        async_url=async_url,
        async_enabled=initENV.DOCLING_SERVE_USE_ASYNC,
        async_timeout=initENV.DOCLING_SERVE_ASYNC_TIMEOUT,
        poll_interval=initENV.DOCLING_SERVE_ASYNC_POLL_INTERVAL,
        async_slot_provider=async_slot_provider,
    )


def extract_document(
    file_path: Path,
    *,
    job_id: str,
    file_id: str,
    original_path: str | None = None,
    async_slot_provider=None,
) -> ExtractionOutcome:
    p = Path(file_path)
    if not p.exists():
        raise ExtractionFailed(f"file not found: {p}")

    debug_dir: Optional[Path] = None
    if initENV.DEBUG:
        try:
            ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            original_name = Path(original_path or p).name or p.name
            safe_name = slugify(Path(original_name).stem)
            debug_dir = ensure_dir(Path("./debug") / f"{ts}_{safe_name}")
        except Exception:
            debug_dir = None

    docling_path = p
    try:
        docling_path, conversion_dir = convert_for_docling(p)
    except Exception as exc:
        docling_path = p
        log(f"[convert_for_docling_error] source={p} err={exc}")
        conversion_dir = None

    try:
        size = docling_path.stat().st_size
        log(f"[extract_start] job_id={job_id} file_id={file_id} path={docling_path} size={size} original={p}")
        ingestor = get_docling_ingestor(async_slot_provider=async_slot_provider)

        extraction: Optional[DoclingExtraction] = None
        docling_error: Optional[Exception] = None
        docling_source = "docling-serve"
        try:
            extraction = ingestor.extract(
                docling_path,
                upload_images=False,
                debug=initENV.DEBUG,
                debug_dir=debug_dir,
            )
        except Exception as exc:
            docling_error = exc
            log(f"[extract_error] job_id={job_id} file_id={file_id} path={docling_path} err={exc}")

        if extraction is None and docling_path.suffix.lower() == ".pdf":
            log(
                f"[extract_fallback] job_id={job_id} file_id={file_id} using legacy PDF fallback due to {docling_error}"
            )
            try:
                fallback_text = legacy_pdf_fallback_text(p)
            except Exception as exc:
                fallback_text = ""
                if docling_error is None:
                    docling_error = exc
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
                    slug=slugify(p.stem),
                )
                docling_source = "pdf_fallback"

        if extraction is None:
            raise ExtractionFailed(f"docling_failed: {docling_error}")

        clean = (extraction.text or "").strip()
        detected = extraction.mime_type or "docling-serve"
        log(
            f"[extract_ok] job_id={job_id} file_id={file_id} mime={detected} chars={len(clean)} chunks={len(extraction.chunks)} source={docling_source}"
        )

        return ExtractionOutcome(
            extraction=extraction,
            source=docling_source,
            ingestor=ingestor,
            debug_dir=debug_dir,
        )
    finally:
        try:
            if conversion_dir:
                shutil.rmtree(conversion_dir, ignore_errors=True)
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
DOCLING_SERVE_URL = initENV.DOCLING_SERVE_URL
DOCLING_SERVE_TIMEOUT = initENV.DOCLING_SERVE_TIMEOUT
NEXTCLOUD_DOC_DIR = initENV.NEXTCLOUD_DOC_DIR
NEXTCLOUD_IMAGE_DIR = initENV.NEXTCLOUD_IMAGE_DIR
NEXTCLOUD_BASE_URL = initENV.NEXTCLOUD_BASE_URL
NEXTCLOUD_USER = initENV.NEXTCLOUD_USER
NEXTCLOUD_TOKEN = initENV.NEXTCLOUD_TOKEN
DECISION_LOG_ENABLED = initENV.DECISION_LOG_ENABLED
DECISION_LOG_MAX_PER_JOB = initENV.DECISION_LOG_MAX_PER_JOB
WORKER_ID = f"{os.uname().nodename}-pid{os.getpid()}"
LOG_PATH = Path("logs") / "scan_scheduler.log"

