import json
import os
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests
from flask import Flask, flash, redirect, render_template, request, url_for

from helpers import init_conn
from nextcloud_client import NextcloudError, env_client
from scan_scheduler import mark_deleted

def load_env_file():
    env_file = Path(__file__).parent / ".env.local"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()

load_env_file()

app = Flask(__name__)
app.secret_key = os.environ.get("WEB_GUI_SECRET", "dev-secret")

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_PATH = Path(os.environ.get("SCHEDULER_LOG", PROJECT_ROOT / "log/scan_scheduler.log"))
DB_PATH = Path(os.environ.get("DB_PATH", PROJECT_ROOT / "DocumentDatabase/state.db"))
ENV_FILE = Path(os.environ.get("ENV_FILE", PROJECT_ROOT / ".env.local"))
EXAMPLE_ENV = Path(os.environ.get("ENV_EXAMPLE", PROJECT_ROOT / ".env.local.example"))

WEB_GUI_DEBUG = os.environ.get("WEB_GUI_DEBUG", "false").lower() in {"1", "true", "yes"}

BRAIN_URL = os.environ.get("BRAIN_URL", "http://192.168.177.151:8080").rstrip("/")
BRAIN_COLLECTION = os.environ.get("BRAIN_COLLECTION", "documents")
BRAIN_API_KEY = os.environ.get("BRAIN_API_KEY", "change-me")

NEXTCLOUD_IMAGE_DIR = os.environ.get("NEXTCLOUD_IMAGE_DIR", "/RAGimages")


def log_debug(msg: str):
    if not WEB_GUI_DEBUG:
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


def read_log() -> str:
    if not LOG_PATH.exists():
        return "No log file found yet."
    try:
        lines = LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
        recent_lines = list(reversed(lines[-50:]))
        return "\n".join(recent_lines)
    except Exception as exc:  # pragma: no cover - defensive
        return f"Failed to read log: {exc}"


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


def current_env() -> Dict[str, str]:
    merged = dict(os.environ)
    merged.update(load_env_file())
    return merged


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


def delete_nextcloud_images(file_id: str) -> str:
    try:
        client = env_client()
    except Exception as exc:
        return f"Nextcloud connection failed: {exc}"
    removed = 0
    for entry in client.walk(NEXTCLOUD_IMAGE_DIR):
        name = entry.get("path") or ""
        if file_id in name:
            try:
                client.delete(name)
                removed += 1
            except Exception:
                continue
    return f"Removed {removed} image(s) from Nextcloud."


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
    return f"Deleted {removed} items from {NEXTCLOUD_IMAGE_DIR}."


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


@app.route("/")
def home():
    env = load_env_file()
    return render_template(
        "gui.html",
        log_preview=read_log(),
        env_values=env,
        current_env=current_env(),
        qdrant_collection=BRAIN_COLLECTION,
    )


@app.post("/start")
def start_scheduler():
    code, output = _run_command(["./brain_scan.sh", "start"])
    log_debug(f"[start_scheduler] exit_code={code}")
    flash(output)
    flash(f"Exit code: {code}")
    return redirect(url_for("home"))


@app.post("/stop")
def stop_scheduler():
    code, output = _run_command(["./brain_scan.sh", "stop"])
    log_debug(f"[stop_scheduler] exit_code={code}")
    flash(output)
    flash(f"Exit code: {code}")
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
    return render_template(
        "gui.html",
        log_preview=read_log(),
        search_term=term,
        search_results=found,
        search_error=error,
        env_values=load_env_file(),
        current_env=current_env(),
        qdrant_collection=BRAIN_COLLECTION,
    )


@app.post("/delete")
def delete_file():
    file_id = request.form.get("file_id", "").strip()
    messages = []
    if not file_id:
        log_debug("[delete_file_skipped] missing file_id")
        flash("Missing file_id")
        return redirect(url_for("home"))
    messages.append(delete_qdrant_entries(file_id))
    messages.append(delete_state_entry(file_id))
    messages.append(delete_nextcloud_images(file_id))
    for msg in messages:
        log_debug(f"[delete_file] file_id={file_id} msg={msg}")
    for msg in messages:
        flash(msg)
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
    for msg in messages:
        flash(msg)
    return redirect(url_for("home"))


@app.post("/reload-log")
def reload_log():
    log_debug("[reload_log]")
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
    port = int(os.environ.get("WEB_GUI_PORT", "8088"))
    host = os.environ.get("WEB_GUI_HOST", "0.0.0.0")
    debug = os.environ.get("WEB_GUI_DEBUG", "false").lower() in {"1", "true", "yes"}
    app.run(host=host, port=port, debug=debug)
