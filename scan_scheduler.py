#!/usr/bin/env python3
import hashlib
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Set

from nextcloud_client import env_client, NextcloudError

import initENV
from helpers import init_conn, compute_file_id, is_due

DB_PATH = initENV.DB_PATH
NEXTCLOUD_DOC_DIR = initENV.NEXTCLOUD_DOC_DIR
EXCLUDE_GLOBS = initENV.EXCLUDE_GLOBS
MAX_JOBS_PER_PASS = initENV.MAX_JOBS_PER_PASS
SLEEP_SECS = initENV.SLEEP_SECS
WORKER_STOP_TIMEOUT = initENV.WORKER_STOP_TIMEOUT
NEXTCLOUD_BASE_URL = initENV.NEXTCLOUD_BASE_URL
NEXTCLOUD_USER = initENV.NEXTCLOUD_USER
DOCLING_MAX_WORKERS = initENV.DOCLING_MAX_WORKERS
PIPELINE_WORKERS = initENV.PIPELINE_WORKERS
SCAN_ROOTS = (NEXTCLOUD_DOC_DIR,)

HIGH_PRIORITY_NEW = 100
RETRY_PRIORITY_ERROR = 90
RETRY_PRIORITY_REVIEW_SMALL = 70
RETRY_PRIORITY_REVIEW_LARGE = 50
RETRY_PRIORITY_DEFAULT = 30

PROJECT_ROOT = Path(__file__).resolve().parent
WORKER_SCRIPT = PROJECT_ROOT / "ingest_worker.py"
PID_FILE = initENV.SCAN_SCHEDULER_PID_FILE


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_scan(message: str):
    ts = _timestamp()
    print(f"[{ts}] [scan] {message}", flush=True)


def row_get(row, key, default=None):
    if row is None:
        return default
    if isinstance(row, dict):
        return row.get(key, default)
    try:
        return row[key]
    except Exception:
        try:
            return row[int(key)]
        except Exception:
            return default


def compute_content_hash(path: Path, chunk_size: int = 1_048_576) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def find_by_content_hash(conn, content_hash: str, new_path: Path):
    if not content_hash:
        return None
    rows = conn.execute(
        "SELECT id, path, deleted_at, content_hash FROM files WHERE content_hash=?",
        (content_hash,),
    ).fetchall()
    # Prefer entries that are already marked deleted or whose path vanished.
    for prefer_deleted in (True, False):
        for row in rows:
            deleted_at = row_get(row, "deleted_at")
            if bool(deleted_at) != prefer_deleted:
                continue
            candidate_path = row_get(row, "path")
            if not candidate_path:
                return row
            try:
                cpath = Path(candidate_path)
                if cpath.resolve() == new_path.resolve():
                    return row
                if not cpath.exists():
                    return row
            except Exception:
                # Broken path -> treat as rename candidate
                return row
    return None
    
def mark_deleted(conn, fid: str) -> bool:
    cur = conn.execute(
        """
        UPDATE files
           SET status='deleted',
               last_checked_at=datetime('now'),
               next_review_at=NULL,
               review_reason=NULL,
               should_reingest=0,
               deleted_at=datetime('now'),
               updated_at=datetime('now')
         WHERE id=? AND deleted_at IS NULL
        """,
        (fid,),
    )
    updated = cur.rowcount if hasattr(cur, "rowcount") else 0
    if updated:
        conn.execute(
            "UPDATE jobs SET status='canceled', locked_at=NULL WHERE file_id=? AND status IN ('queued','running')",
            (fid,),
        )
    return bool(updated)


def purge_missing(conn, seen_ids: Set[str]) -> int:
    removed = 0
    rows = conn.execute(
        "SELECT id, path FROM files WHERE deleted_at IS NULL"
    ).fetchall()
    for row in rows:
        fid = row_get(row, "id")
        if not fid or fid in seen_ids:
            continue
        path = row_get(row, "path")
        if path:
            try:
                p = Path(path)
                if p.exists():
                    # File still present -> keep
                    continue
            except Exception:
                pass
        if mark_deleted(conn, fid):
            removed += 1
    return removed


def write_pid_file():
    if not PID_FILE:
        return
    try:
        pid_path = Path(PID_FILE)
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(os.getpid()), encoding="utf-8")
    except Exception as exc:
        log_scan(f"failed to write pid file: {exc}")

def excluded(path: Path) -> bool:
    s = str(path)
    for g in EXCLUDE_GLOBS:
        if g and Path(s).match(g):
            return True
    return False

def upsert_file(conn, fid, path, mtime, size, inode, device, content_hash):
    conn.execute(
        """
        INSERT INTO files (id, path, mtime, size, inode, device, content_hash, updated_at, first_seen_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
        ON CONFLICT(id) DO UPDATE SET
            path=excluded.path,
            mtime=excluded.mtime,
            size=excluded.size,
            inode=COALESCE(excluded.inode, files.inode),
            device=COALESCE(excluded.device, files.device),
            content_hash=COALESCE(excluded.content_hash, files.content_hash),
            deleted_at=NULL,
            updated_at=datetime('now');
    """,
        (
            fid,
            path,
            int(mtime),
            int(size),
            None if inode is None else int(inode),
            None if device is None else int(device),
            content_hash,
        ),
    )

def has_active_job(conn, fid):
    return conn.execute(
        "SELECT 1 FROM jobs WHERE file_id=? AND status IN ('queued','running') LIMIT 1;",
        (fid,)
    ).fetchone() is not None

def should_enqueue(conn, fid):
    row = conn.execute(
        """
        SELECT status, last_checked_at, next_review_at, should_reingest, deleted_at, size, priority
          FROM files
         WHERE id=?
        """,
        (fid,),
    ).fetchone()
    if not row:
        return True, "new_file"
    status = row_get(row, "status", row_get(row, 0))
    last_checked_at = row_get(row, "last_checked_at", row_get(row, 1))
    next_review_at = row_get(row, "next_review_at", row_get(row, 2))
    should_reingest = row_get(row, "should_reingest", row_get(row, 3))
    deleted_at = row_get(row, "deleted_at", row_get(row, 4))

    if deleted_at:
        return False, "deleted", row

    normalized_status = (str(status).strip().lower() if status is not None else "")

    # Status fehlt bzw. ist "NULL" -> wie Neuprüfung behandeln
    if not normalized_status or normalized_status == "null":
        return True, "status_null", row

    # 1️⃣  expliziter Reingest
    if should_reingest:
        return True, "should_reingest", row

    # 2️⃣  Erstprüfung
    if last_checked_at is None:
        return True, "first_check", row

    # 3️⃣  Wiedervorlage fällig?
    if next_review_at and is_due(next_review_at):
        return True, "review_due", row

    # 4️⃣  Fehlversuch erneut versuchen (Legacy: keine review_at gesetzt)
    if status and status.startswith("error") and not next_review_at:
        return True, f"retry_{status}", row

    return False, "no_need", row


def compute_priority(row) -> int:
    status = row_get(row, "status")
    size = row_get(row, "size", 0) or 0
    last_checked_at = row_get(row, "last_checked_at")

    if last_checked_at is None:
        return HIGH_PRIORITY_NEW

    normalized = (status or "").lower()
    if normalized.startswith("error"):
        return RETRY_PRIORITY_ERROR

    if normalized.startswith("ai_not_relevant") or normalized.startswith("skipped"):
        if int(size) < 1_000_000:
            return RETRY_PRIORITY_REVIEW_SMALL
        return RETRY_PRIORITY_REVIEW_LARGE

    return RETRY_PRIORITY_DEFAULT

def enqueue_job(conn, fid, reason):
    if has_active_job(conn, fid):
        # conn.execute("INSERT INTO decision_log(step, file_id, detail) VALUES(?,?,?)",
        #             ("scan_skip", fid, "already_queued"))
        return False
    jid = str(uuid.uuid4())
    conn.execute("""
        INSERT INTO jobs (job_id, file_id, enqueue_at, run_not_before, status, attempts)
        VALUES (?, ?, datetime('now'), datetime('now'), 'queued', 0)
    """, (jid, fid))
    conn.execute("INSERT INTO decision_log(step, job_id, file_id, detail) VALUES(?,?,?,?)",
                 ("scan_enqueue", jid, fid, reason))
    return True

def sync_and_enqueue(conn):
    added = enq = 0
    seen: Set[str] = set()
    try:
        client = env_client()
    except Exception as exc:
        log_scan(f"failed to init Nextcloud client: {exc}")
        return added, enq, 0

    for root in SCAN_ROOTS:
        log_scan(
            f"walking Nextcloud root={root} base_url={NEXTCLOUD_BASE_URL} user={NEXTCLOUD_USER}"
        )
        try:
            for entry in client.walk(root):
                if entry.get("is_dir"):
                    continue
                path_str = entry.get("path") or ""
                if excluded(Path(path_str)):
                    continue
                mtime = int(entry.get("mtime") or 0)
                size = int(entry.get("size") or 0)
                content_hash = entry.get("etag")

                fid = fid_guess = compute_file_id(path_str)
                row = conn.execute(
                    "SELECT id, content_hash FROM files WHERE id=?",
                    (fid_guess,),
                ).fetchone()
                if row:
                    fid = row_get(row, "id", fid_guess)
                    if not content_hash:
                        content_hash = row_get(row, "content_hash", row_get(row, 1))

                upsert_file(conn, fid, path_str, mtime, size, None, None, content_hash)
                added += 1
                seen.add(fid)

                need, reason, meta = should_enqueue(conn, fid)
                if not need:
                    continue

                desired_priority = compute_priority(meta or {})
                current_priority = row_get(meta, "priority") if meta else None
                if desired_priority is not None and desired_priority != current_priority:
                    conn.execute(
                        "UPDATE files SET priority=?, updated_at=datetime('now') WHERE id=?",
                        (int(desired_priority), fid),
                    )
                    meta = None

                if enq >= MAX_JOBS_PER_PASS:
                    continue

                if enqueue_job(conn, fid, reason):
                    enq += 1
        except NextcloudError as exc:
            log_scan(f"Nextcloud error for root {root}: {exc}")
            continue
        log_scan(f"finished root={root} added={added} enqueued={enq}")

    removed = purge_missing(conn, seen)
    conn.commit()
    return added, enq, removed

def count_queued_jobs(conn) -> int:
    row = conn.execute("SELECT COUNT(*) FROM jobs WHERE status='queued';").fetchone()
    if not row:
        return 0
    try:
        return int(row[0])
    except (KeyError, TypeError, ValueError):
        try:
            return int(row["COUNT(*)"])
        except Exception:
            return 0


def count_running_jobs(conn) -> int:
    row = conn.execute("SELECT COUNT(*) FROM jobs WHERE status='running';").fetchone()
    if not row:
        return 0
    try:
        return int(row[0])
    except (KeyError, TypeError, ValueError):
        try:
            return int(row["COUNT(*)"])
        except Exception:
            return 0


def fail_running_jobs(conn, reason: str = "scheduler_stopped") -> int:
    rows = conn.execute(
        "SELECT job_id, file_id FROM jobs WHERE status='running';"
    ).fetchall()
    failed = 0
    for row in rows:
        job_id = row_get(row, "job_id")
        file_id = row_get(row, "file_id")
        conn.execute(
            """
            UPDATE jobs
               SET status='failed',
                   last_error=?,
                   locked_at=NULL,
                   worker_id=NULL
             WHERE job_id=?
            """,
            (reason, job_id),
        )
        conn.execute(
            "INSERT INTO decision_log(step, job_id, file_id, detail) VALUES(?,?,?,?)",
            ("error", job_id, file_id, reason),
        )
        failed += 1

    if failed:
        conn.commit()

    return failed

def main():
    conn = init_conn(DB_PATH)
    worker_procs: list[subprocess.Popen] = []
    stop_requested = False

    def handle_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    def worker_running(proc: subprocess.Popen | None) -> bool:
        return proc is not None and proc.poll() is None

    def prune_workers():
        nonlocal worker_procs
        alive: list[subprocess.Popen] = []
        for proc in worker_procs:
            if worker_running(proc):
                alive.append(proc)
            else:
                rc = proc.returncode if proc else None
                log_scan(f"worker exited with code {rc}")
        worker_procs = alive

    def start_worker():
        cmd = [sys.executable, str(WORKER_SCRIPT)]
        log_scan(f"starting worker: {' '.join(cmd)}")
        try:
            proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
            worker_procs.append(proc)
        except Exception as exc:
            log_scan(f"failed to start worker: {exc}")
            return False
        return True

    def stop_workers(limit: int | None = None):
        nonlocal worker_procs
        targets = worker_procs if limit is None else worker_procs[:limit]
        remaining = worker_procs if limit is None else worker_procs[limit:]
        for proc in targets:
            if not worker_running(proc):
                continue
            log_scan("stopping worker...")
            proc.terminate()
            try:
                proc.wait(timeout=WORKER_STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                log_scan("worker did not exit in time - killing")
                proc.kill()
                proc.wait()
        worker_procs = [p for p in remaining if worker_running(p)]

    log_scan(
        f"DB={DB_PATH} roots={SCAN_ROOTS} sleep={SLEEP_SECS}s max_jobs={MAX_JOBS_PER_PASS}"
    )
    write_pid_file()
    try:
        while not stop_requested:
            prune_workers()

            n, e, d = sync_and_enqueue(conn)
            queued = count_queued_jobs(conn)
            running_jobs = count_running_jobs(conn)
            log_scan(
                f"synced={n} enqueued={e} pruned={d} queued={queued} running={running_jobs}"
            )

            # Keep starting new workers whenever the amount of active work dips below
            # the configured cap, as long as there is something queued to process.
            desired_workers = min(PIPELINE_WORKERS, running_jobs + queued)
            running_workers = sum(1 for p in worker_procs if worker_running(p))

            if desired_workers > running_workers:
                to_start = desired_workers - running_workers
                for _ in range(to_start):
                    if not start_worker():
                        break
            elif desired_workers < running_workers:
                stop_workers(running_workers - desired_workers)

            if stop_requested:
                break

            sleep_remaining = 300 if queued == 0 else SLEEP_SECS
            if queued == 0 and sleep_remaining != SLEEP_SECS:
                log_scan(f"idle queue detected, sleeping {sleep_remaining}s")
            while sleep_remaining > 0 and not stop_requested:
                time.sleep(min(1, sleep_remaining))
                sleep_remaining -= 1
    except KeyboardInterrupt:
        log_scan("interrupted, stopping...")
    finally:
        stop_workers()
        failed_jobs = fail_running_jobs(conn, "failed_due_to_scheduler_stop")
        if failed_jobs:
            log_scan(f"marked {failed_jobs} running jobs as failed after stop")
        try:
            conn.close()
        except Exception:
            pass
        log_scan("scheduler stopped")

if __name__ == "__main__":
    main()
