#!/usr/bin/env python3
import hashlib
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Set

from helpers import init_conn, compute_file_id, is_due

DB_PATH = os.environ.get("DB_PATH", "DocumentDatabase/state.db")
NEXTCLOUD_DOC_DIR = os.environ.get("NEXTCLOUD_DOC_DIR", "/nextcloud/documents")
NEXTCLOUD_IMAGE_DIR = os.environ.get("NEXTCLOUD_IMAGE_DIR", "/nextcloud/docling-images")
ROOT_DIRS = [p for p in os.environ.get("ROOT_DIRS", f"{NEXTCLOUD_DOC_DIR},{NEXTCLOUD_IMAGE_DIR}").split(",") if p]
EXCLUDE_GLOBS = [g for g in os.environ.get("EXCLUDE_GLOBS", "").split(",") if g]
FOLLOW_SYMLINKS = os.environ.get("FOLLOW_SYMLINKS", "0") == "1"
MAX_JOBS_PER_PASS = int(os.environ.get("MAX_JOBS_PER_PASS", "5"))
SLEEP_SECS = int(os.environ.get("SLEEP_SECS", "10"))
WORKER_STOP_TIMEOUT = int(os.environ.get("WORKER_STOP_TIMEOUT", "20"))

HIGH_PRIORITY_NEW = 100
RETRY_PRIORITY_ERROR = 90
RETRY_PRIORITY_REVIEW_SMALL = 70
RETRY_PRIORITY_REVIEW_LARGE = 50
RETRY_PRIORITY_DEFAULT = 30

PROJECT_ROOT = Path(__file__).resolve().parent
WORKER_SCRIPT = PROJECT_ROOT / "ingest_worker.py"
PID_FILE = os.environ.get("SCAN_SCHEDULER_PID_FILE")


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
        print(f"[scan] failed to write pid file: {exc}", flush=True)

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
    for root in ROOT_DIRS:
        rootp = Path(root)
        if not rootp.exists():
            continue
        for dirpath, _, filenames in os.walk(root, followlinks=FOLLOW_SYMLINKS):
            if any(part.startswith(".") for part in Path(dirpath).parts):
                continue
            for fn in filenames:
                p = Path(dirpath) / fn
                if excluded(p):
                    continue
                try:
                    st = p.stat()
                except FileNotFoundError:
                    continue

                fid_guess = compute_file_id(str(p))
                inode = getattr(st, "st_ino", None)
                device = getattr(st, "st_dev", None)
                content_hash = None
                fid = fid_guess

                row = conn.execute(
                    "SELECT id, mtime, size, content_hash FROM files WHERE id=?",
                    (fid_guess,),
                ).fetchone()
                if row:
                    fid = row_get(row, "id", fid_guess)
                    stored_hash = row_get(row, "content_hash", row_get(row, 3))
                    prev_mtime = row_get(row, "mtime", row_get(row, 1))
                    prev_size = row_get(row, "size", row_get(row, 2))
                    if (
                        stored_hash
                        and prev_mtime is not None
                        and prev_size is not None
                        and int(prev_mtime) == int(st.st_mtime)
                        and int(prev_size) == int(st.st_size)
                    ):
                        content_hash = stored_hash
                    else:
                        try:
                            content_hash = compute_content_hash(p)
                        except Exception:
                            content_hash = stored_hash
                else:
                    existing_row = None
                    if inode is not None and device is not None:
                        existing_row = conn.execute(
                            "SELECT id, content_hash, path FROM files WHERE device=? AND inode=? ORDER BY deleted_at IS NULL DESC LIMIT 1",
                            (int(device), int(inode)),
                        ).fetchone()
                    if existing_row:
                        fid = row_get(existing_row, "id", fid_guess)
                        stored_hash = row_get(existing_row, "content_hash", row_get(existing_row, 1))
                        if stored_hash:
                            content_hash = stored_hash
                        else:
                            try:
                                content_hash = compute_content_hash(p)
                            except Exception:
                                content_hash = None
                    else:
                        try:
                            content_hash = compute_content_hash(p)
                        except Exception:
                            content_hash = None
                        match = find_by_content_hash(conn, content_hash, p) if content_hash else None
                        if match:
                            fid = row_get(match, "id", fid_guess)

                upsert_file(conn, fid, str(p), st.st_mtime, st.st_size, inode, device, content_hash)
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

def main():
    conn = init_conn(DB_PATH)
    worker_proc: subprocess.Popen | None = None
    stop_requested = False

    def handle_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    def worker_running() -> bool:
        return worker_proc is not None and worker_proc.poll() is None

    def start_worker():
        nonlocal worker_proc
        if worker_running():
            return
        cmd = [sys.executable, str(WORKER_SCRIPT)]
        print(f"[scan] starting worker: {' '.join(cmd)}", flush=True)
        try:
            worker_proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
        except Exception as exc:
            print(f"[scan] failed to start worker: {exc}", flush=True)
            worker_proc = None

    def stop_worker():
        nonlocal worker_proc
        if not worker_proc:
            return
        if worker_proc.poll() is not None:
            worker_proc = None
            return
        print("[scan] stopping worker...", flush=True)
        worker_proc.terminate()
        try:
            worker_proc.wait(timeout=WORKER_STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            print("[scan] worker did not exit in time - killing", flush=True)
            worker_proc.kill()
            worker_proc.wait()
        finally:
            worker_proc = None

    print(f"[scan] DB={DB_PATH} roots={ROOT_DIRS} sleep={SLEEP_SECS}s max_jobs={MAX_JOBS_PER_PASS}")
    write_pid_file()
    try:
        while not stop_requested:
            if worker_proc and worker_proc.poll() is not None:
                rc = worker_proc.returncode
                print(f"[scan] worker exited with code {rc}", flush=True)
                worker_proc = None

            n, e, d = sync_and_enqueue(conn)
            queued = count_queued_jobs(conn)
            print(f"[scan] synced={n} enqueued={e} pruned={d} queued={queued}", flush=True)

            if queued and not worker_running():
                start_worker()

            if stop_requested:
                break

            sleep_remaining = SLEEP_SECS
            while sleep_remaining > 0 and not stop_requested:
                time.sleep(min(1, sleep_remaining))
                sleep_remaining -= 1
    except KeyboardInterrupt:
        print("[scan] interrupted, stopping...", flush=True)
    finally:
        stop_worker()
        try:
            conn.close()
        except Exception:
            pass
        print("[scan] scheduler stopped", flush=True)

if __name__ == "__main__":
    main()
