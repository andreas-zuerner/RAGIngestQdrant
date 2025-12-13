import json
import os
import posixpath
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests
from flask import Flask, flash, redirect, render_template, request, url_for

import initENV
from helpers import init_conn
from nextcloud_client import env_client
from scan_scheduler import mark_deleted
import prompt_store

app = Flask(__name__)
app.secret_key = initENV.WEB_GUI_SECRET

PROJECT_ROOT = initENV.PROJECT_ROOT
LOG_PATH = Path(initENV.SCHEDULER_LOG)
DB_PATH = Path(initENV.DB_PATH)
ENV_FILE = initENV.ENV_FILE
EXAMPLE_ENV = initENV.ENV_EXAMPLE

BRAIN_URL = initENV.BRAIN_URL
BRAIN_COLLECTION = initENV.BRAIN_COLLECTION
BRAIN_API_KEY = initENV.BRAIN_API_KEY

NEXTCLOUD_IMAGE_DIR = initENV.NEXTCLOUD_IMAGE_DIR


def log_debug(msg: str):
    if not initENV.DEBUG:
        return
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(f"[{ts}] [Web_GUI] {msg}\n")
    except Exception:
        pass


def _run_command(args: List[str]) -> Tuple[int, str]:
    cmd_display = " ".join(args)
    log_debug(f"[run_command] cmd={cmd_display}")
    try:
        proc = subprocess.run(
            args,
            cwd=PROJECT_ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_debug(f"[run_command_ok] cmd={cmd_display} exit={proc.returncode}")
        return proc.returncode, proc.stdout
    except FileNotFoundError as exc:
        log_debug(f"[run_command_error] cmd={cmd_display} err={exc}")
        return 1, f"Command not found: {cmd_display}"
    except Exception as exc:
        log_debug(f"[run_command_error] cmd={cmd_display} err={exc}")
        return 1, f"Command failed: {exc}"


def append_log(message: str):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        for line in message.splitlines() or [""]:
            fh.write(f"[{ts}] {line}\n")


def read_log() -> str:
    if not LOG_PATH.exists():
        return "No log file found yet."
    try:
        lines = LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
        recent_lines = list(reversed(lines[-50:]))
        return "\n".join(recent_lines)
    except Exception as exc:  # pragma: no cover - defensive
        return f"Failed to read log: {exc}"


def read_prompts() -> Dict[str, str]:
    return prompt_store.load_prompts()


def save_prompts(prompts: Dict[str, str]):
    prompt_store.save_prompts(prompts)


def load_env_file() -> Dict[str, str]:
    env: Dict[str, str] = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if not line or line.strip().startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    return env


def persist_env_file(env: Dict[str, str]):
    content = "\n".join(f"{k}={v}" for k, v in sorted(env.items())) + "\n"
    ENV_FILE.write_text(content, encoding="utf-8")


def load_env_descriptions() -> Dict[str, str]:
    descriptions: Dict[str, str] = {}
    if not EXAMPLE_ENV.exists():
        return descriptions

    pending_comments: List[str] = []
    for line in EXAMPLE_ENV.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            pending_comments = []
            continue
        if stripped.startswith("#"):
            comment = stripped.lstrip("#").strip()
            if comment:
                pending_comments.append(comment)
            continue
        if "=" not in line:
            pending_comments = []
            continue
        key = line.split("=", 1)[0].strip()
        descriptions[key] = " ".join(pending_comments).strip()
        pending_comments = []

    return descriptions


def current_env() -> Dict[str, str]:
    merged = dict(os.environ)
    merged.update(load_env_file())
    return merged


def _dict_from_row(row) -> Dict[str, Optional[str]]:
    if isinstance(row, dict):
        return dict(row)
    try:
        return {k: row[k] for k in row.keys()}  # type: ignore[attr-defined]
    except Exception:
        return {}


def load_pipeline_overview() -> Dict[str, List[Dict[str, Optional[str]]]]:
    if not DB_PATH.exists():
        return {"queued": [], "docling": [], "extracted": [], "pipeline": []}

    conn = init_conn(DB_PATH)
    try:
        queued = [
            _dict_from_row(r)
            for r in conn.execute(
                """
                SELECT jobs.job_id, jobs.file_id, files.path, jobs.enqueue_at
                  FROM jobs
                  JOIN files ON files.id = jobs.file_id
                 WHERE jobs.status='queued'
                 ORDER BY jobs.enqueue_at
                """
            ).fetchall()
        ]

        running = conn.execute(
            """
            SELECT jobs.job_id, jobs.file_id, jobs.worker_id, files.path
              FROM jobs
              JOIN files ON files.id = jobs.file_id
             WHERE jobs.status='running'
            """
        ).fetchall()

        def latest_step(job_id: str) -> str:
            row = conn.execute(
                "SELECT step FROM decision_log WHERE job_id=? ORDER BY id DESC LIMIT 1",
                (job_id,),
            ).fetchone()
            if isinstance(row, dict):
                return row.get("step") or ""
            if row:
                try:
                    return row[0]
                except Exception:
                    return ""
            return ""

        docling, extracted, pipeline = [], [], []
        for row in running:
            data = _dict_from_row(row)
            step = latest_step(data.get("job_id")) if data else ""
            entry = data or {}
            entry["stage"] = step or "running"
            if step.startswith("docling"):
                docling.append(entry)
            elif step == "extraction_complete":
                extracted.append(entry)
            else:
                stage_label = "relevance"
                if step == "stage_chunking":
                    stage_label = "chunking"
                elif step == "stage_context":
                    stage_label = "context"
                elif step == "stage_embedding":
                    stage_label = "embedding"
                elif step == "stage_relevance":
                    stage_label = "relevance"
                elif step == "pipeline_start":
                    stage_label = "pipeline"
                entry["stage_label"] = stage_label
                pipeline.append(entry)

        return {
            "queued": queued,
            "docling": docling,
            "extracted": extracted,
            "pipeline": pipeline,
        }
    finally:
        conn.close()


def _brain_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if BRAIN_API_KEY:
        headers["x-api-key"] = BRAIN_API_KEY
    return headers


def _brain_post(path: str, payload: dict | None = None) -> requests.Response:
    url = urljoin(BRAIN_URL.rstrip("/") + "/", path.lstrip("/"))
    payload_with_collection = dict(payload or {})
    payload_with_collection["collection"] = BRAIN_COLLECTION
    return requests.post(
        url, json=payload_with_collection, headers=_brain_headers(), timeout=30
    )


def qdrant_scroll_by_name(substring: str, limit: int = 200) -> List[dict]:
    payload = {"substring": substring, "limit": limit, "collection": BRAIN_COLLECTION}
    r = _brain_post("/admin/scroll-by-name", payload)
    if r.status_code != 200:
        raise RuntimeError(f"Brain scroll failed: {r.status_code} {r.text}")
    data = r.json()
    return data.get("results", [])


def qdrant_ids_for_file(file_id: str, limit: int = 500) -> List[str | int]:
    payload = {"file_id": file_id, "limit": limit, "collection": BRAIN_COLLECTION}
    r = _brain_post("/admin/ids-for-file", payload)
    if r.status_code != 200:
        raise RuntimeError(f"Brain lookup failed: {r.status_code} {r.text}")
    data = r.json()
    return data.get("ids", [])


def delete_qdrant_entries(file_id: str) -> str:
    try:
        ids = qdrant_ids_for_file(file_id)
    except Exception as exc:
        return f"Failed to locate items for deletion: {exc}"

    if not ids:
        return "No matching Qdrant entries found."

    r = _brain_post("/admin/delete", {"ids": ids, "collection": BRAIN_COLLECTION})
    if r.status_code not in (200, 202):
        return (
            "Brain deletion failed: "
            f"{r.status_code} {r.text or r.content.decode('utf-8', errors='ignore')}"
        )
    return f"Deleted {len(ids)} Qdrant point(s): {', '.join(str(i) for i in ids)}"


def image_references_for_file(file_id: str) -> List[str]:
    if not DB_PATH.exists():
        return []
    conn = init_conn(DB_PATH)
    try:
        rows = conn.execute("SELECT reference FROM images WHERE file_id=?", (file_id,)).fetchall()
        return [row["reference"] for row in rows if row and row["reference"]]
    except Exception:
        return []
    finally:
        conn.close()


def clear_image_records(file_id: str | None = None) -> str:
    if not DB_PATH.exists():
        return "No database found for image records."
    conn = init_conn(DB_PATH)
    try:
        if file_id:
            cur = conn.execute("DELETE FROM images WHERE file_id=?", (file_id,))
        else:
            cur = conn.execute("DELETE FROM images")
        conn.commit()
        removed = cur.rowcount or 0
        scope = f" for {file_id}" if file_id else " from state.db"
        return f"Removed {removed} image record(s){scope}."
    finally:
        conn.close()


def _image_parent_directories(references: List[str]) -> List[str]:
    base_dir = posixpath.normpath(
        NEXTCLOUD_IMAGE_DIR if NEXTCLOUD_IMAGE_DIR.startswith("/") else f"/{NEXTCLOUD_IMAGE_DIR}"
    )
    parents: List[str] = []
    for reference in references:
        if not reference:
            continue
        remote_path = reference
        if not remote_path.startswith("/"):
            remote_path = posixpath.normpath(f"/{NEXTCLOUD_IMAGE_DIR}/{remote_path}")
        parent = posixpath.dirname(remote_path.rstrip("/"))
        if parent and parent != "/" and parent.startswith(base_dir):
            parents.append(parent)
    return sorted(set(parents), key=len, reverse=True)


def delete_nextcloud_images(file_id: str, references: Optional[List[str]] = None) -> str:
    try:
        client = env_client()
    except Exception as exc:
        return f"Nextcloud connection failed: {exc}"
    targets = references if references is not None else image_references_for_file(file_id)
    if not targets:
        return "No recorded images for file."
    removed = 0
    for reference in targets:
        remote_path = reference
        if not remote_path.startswith("/"):
            remote_path = posixpath.normpath(f"/{NEXTCLOUD_IMAGE_DIR}/{remote_path}")
        try:
            client.delete(remote_path)
            removed += 1
        except Exception:
            continue
    removed_dirs = 0
    for parent in _image_parent_directories(targets):
        try:
            client.delete(parent)
            removed_dirs += 1
        except Exception:
            continue
    return (
        f"Removed {removed} image(s) from Nextcloud ({len(targets)} recorded)."
        f" Deleted {removed_dirs} folder(s)."
    )

def delete_images_for_file(file_id: str) -> Tuple[str, str]:
    """Remove a file's images from Nextcloud and the state database."""

    if not DB_PATH.exists():
        return "No recorded images for file.", "No database found for image records."

    conn = init_conn(DB_PATH)
    try:
        rows = conn.execute("SELECT reference FROM images WHERE file_id=?", (file_id,)).fetchall()
        references = [row["reference"] for row in rows if row and row["reference"]]
        nc_msg = delete_nextcloud_images(file_id, references)

        cur = conn.execute("DELETE FROM images WHERE file_id=?", (file_id,))
        conn.commit()
        db_removed = cur.rowcount or 0
        db_msg = f"Removed {db_removed} image record(s) for {file_id}."
        return nc_msg, db_msg
    finally:
        conn.close()

def reset_nextcloud_images() -> str:
    try:
        client = env_client()
    except Exception as exc:
        return f"Nextcloud connection failed: {exc}"
    removed = 0
    for entry in client.walk(NEXTCLOUD_IMAGE_DIR):
        path = entry.get("path")
        if not path:
            continue
        try:
            client.delete(path)
            removed += 1
        except Exception:
            continue
    db_msg = clear_image_records()
    return f"Deleted {removed} items from {NEXTCLOUD_IMAGE_DIR}. {db_msg}"


def delete_state_entry(file_id: str) -> str:
    if not DB_PATH.exists():
        return "Database not found."
    conn = init_conn(DB_PATH)
    try:
        if mark_deleted(conn, file_id):
            conn.commit()
            return "Marked as deleted in state.db."
        return "Entry not found in state.db."
    finally:
        conn.close()


def reset_state_db() -> str:
    if DB_PATH.exists():
        DB_PATH.unlink()
        return f"Removed {DB_PATH}"
    return "No database file to remove."


def reset_qdrant() -> str:
    r = _brain_post("/admin/purge", {"collection": BRAIN_COLLECTION})
    if r.status_code not in (200, 202):
        return f"Failed to purge collection via brain: {r.status_code} {r.text}"
    return "Qdrant collection purged via brain."


def render_gui(**extra):
    base = dict(
        log_preview=read_log(),
        env_values=load_env_file(),
        env_descriptions=load_env_descriptions(),
        current_env=current_env(),
        qdrant_collection=BRAIN_COLLECTION,
        prompts=read_prompts(),
        pipeline_overview=load_pipeline_overview(),
    )
    base.update(extra)
    return render_template("gui.html", **base)


@app.route("/")
def home():
    return render_gui()


@app.post("/start")
def start_scheduler():
    code, output = _run_command(["./brain_scan.sh", "start"])
    log_debug(f"[start_scheduler] exit_code={code}")
    append_log(output.strip() or "Started scheduler")
    append_log(f"Exit code: {code}")
    if code != 0:
        flash("Failed to start scheduler (see log for details).")
    return redirect(url_for("home"))


@app.post("/stop")
def stop_scheduler():
    code, output = _run_command(["./brain_scan.sh", "stop"])
    log_debug(f"[stop_scheduler] exit_code={code}")
    append_log(output.strip() or "Stopped scheduler")
    append_log(f"Exit code: {code}")
    if code != 0:
        flash("Failed to stop scheduler (see log for details).")
    return redirect(url_for("home"))


@app.post("/search")
def search():
    term = request.form.get("search_term", "")
    found = []
    error = None
    if term:
        try:
            found = qdrant_scroll_by_name(term, limit=200)
            log_debug(f"[search_success] term='{term}' results={len(found)}")
        except Exception as exc:
            error = str(exc)
            log_debug(f"[search_error] term='{term}' err={exc}")
    else:
        log_debug("[search_skipped] empty term")
    return render_gui(search_term=term, search_results=found, search_error=error)


@app.post("/delete")
def delete_file():
    file_id = request.form.get("file_id", "").strip()
    messages = []
    if not file_id:
        log_debug("[delete_file_skipped] missing file_id")
        flash("Missing file_id")
        return redirect(url_for("home"))
    image_refs = image_references_for_file(file_id)
    messages.append(delete_qdrant_entries(file_id))
    messages.append(delete_state_entry(file_id))
    messages.append(delete_nextcloud_images(file_id, image_refs))
    messages.append(clear_image_records(file_id))
    for msg in messages:
        log_debug(f"[delete_file] file_id={file_id} msg={msg}")
        append_log(f"[delete_file] {file_id}: {msg}")
    return redirect(url_for("home"))


@app.post("/env/update")
def update_env():
    env = load_env_file()
    updates = {k: v for k, v in request.form.items() if k.startswith("env_")}
    for key, value in updates.items():
        clean_key = key[len("env_"):]
        env[clean_key] = value
    persist_env_file(env)
    if updates:
        log_debug(f"[env_update] keys={[k[len('env_'):] for k in updates.keys()]}")
    flash("Environment file updated.")
    return redirect(url_for("home"))


@app.post("/env/add")
def add_env_var():
    key = request.form.get("new_key", "").strip()
    value = request.form.get("new_value", "")
    if not key:
        log_debug("[env_add_skipped] empty key")
        flash("Key cannot be empty")
        return redirect(url_for("home"))
    env = load_env_file()
    env[key] = value
    persist_env_file(env)
    log_debug(f"[env_add] key={key}")
    flash(f"Added {key} to env file.")
    return redirect(url_for("home"))


@app.post("/reset")
def reset_all():
    restore_defaults = request.form.get("restore_defaults") == "on"
    messages = [reset_qdrant(), reset_state_db(), reset_nextcloud_images()]
    if restore_defaults and EXAMPLE_ENV.exists():
        ENV_FILE.write_text(EXAMPLE_ENV.read_text(encoding="utf-8"), encoding="utf-8")
        messages.append("Restored .env.local from example file.")
    for msg in messages:
        log_debug(f"[reset_all] {msg}")
        append_log(f"[reset_all] {msg}")
    return redirect(url_for("home"))


@app.post("/reload-log")
def reload_log():
    log_debug("[reload_log]")
    return redirect(url_for("home"))


@app.get("/log-data")
def log_data():
    return {"log": read_log()}


@app.post("/prompts/update")
def update_prompts():
    prompts = {
        "relevance": request.form.get("prompt_relevance", ""),
        "chunking": request.form.get("prompt_chunking", ""),
        "context": request.form.get("prompt_context", ""),
    }
    save_prompts(prompts)
    flash("Prompts updated.")
    return redirect(url_for("home"))


def format_payload(payload: Optional[dict]) -> str:
    if not payload:
        return ""
    try:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception:
        return str(payload)


app.jinja_env.filters["format_payload"] = format_payload


if __name__ == "__main__":
    port = initENV.WEB_GUI_PORT
    host = initENV.WEB_GUI_HOST
    app.run(host=host, port=port, debug=initENV.DEBUG)
