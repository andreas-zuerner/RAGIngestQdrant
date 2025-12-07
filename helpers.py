import os, json, time, uuid, sqlite3, hashlib, random
from datetime import datetime, timezone, timedelta
from pathlib import Path

DEFAULT_DB_PATH = os.environ.get("DB_PATH", "DocumentDatabase/state.db")

def utcnow_iso():
    # Keep for logging if needed (not used for DB timestamps anymore)
    return datetime.now(timezone.utc).isoformat()


def ensure_db(conn: sqlite3.Connection):
    """Create core tables and indexes if missing (idempotent)."""

    def column_exists(table: str, column: str) -> bool:
        cur = conn.execute(f"PRAGMA table_info({table})")
        for row in cur.fetchall():
            if len(row) > 1 and row[1] == column:
                return True
        return False

    conn.executescript(
        """
        PRAGMA foreign_keys=ON;

        CREATE TABLE IF NOT EXISTS files (
          id TEXT PRIMARY KEY,
          path TEXT NOT NULL,
          mtime INTEGER NOT NULL,
          size INTEGER NOT NULL,
          status TEXT,
          last_result TEXT,
          last_error TEXT,
          updated_at TEXT NOT NULL,
          first_seen_at TEXT,
          last_checked_at TEXT,
          should_reingest INTEGER,
          review_reason TEXT,
          next_review_at TEXT,
          priority INTEGER DEFAULT 100,
          inode INTEGER,
          device INTEGER,
          content_hash TEXT,
          deleted_at TEXT
        );

        CREATE TABLE IF NOT EXISTS jobs (
          job_id TEXT PRIMARY KEY,
          file_id TEXT NOT NULL,
          enqueue_at TEXT NOT NULL,
          run_not_before TEXT NOT NULL,
          status TEXT NOT NULL,
          attempts INTEGER NOT NULL DEFAULT 0,
          locked_at TEXT,
          worker_id TEXT,
          last_error TEXT,
          FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS decision_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT DEFAULT (datetime('now')),
            job_id TEXT,
            file_id TEXT,
            step TEXT,
            detail TEXT
        );
        """
    )
    conn.commit()

    for column, ddl in (
        ("inode", "ALTER TABLE files ADD COLUMN inode INTEGER"),
        ("device", "ALTER TABLE files ADD COLUMN device INTEGER"),
        ("content_hash", "ALTER TABLE files ADD COLUMN content_hash TEXT"),
        ("deleted_at", "ALTER TABLE files ADD COLUMN deleted_at TEXT"),
    ):
        if not column_exists("files", column):
            conn.execute(ddl)

    conn.commit()

    conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_files_next_review ON files(next_review_at);
        CREATE INDEX IF NOT EXISTS idx_files_status ON files(status);
        CREATE INDEX IF NOT EXISTS idx_files_inode_device ON files(device, inode);
        CREATE INDEX IF NOT EXISTS idx_files_content_hash ON files(content_hash);

        CREATE INDEX IF NOT EXISTS idx_jobs_q ON jobs(status, run_not_before);
        CREATE INDEX IF NOT EXISTS idx_jobs_enqueue ON jobs(enqueue_at, job_id);
        CREATE INDEX IF NOT EXISTS idx_decision_log_step ON decision_log(step, ts);
        """
    )
    conn.commit()

def compute_file_id(path: str):
    # stable ID: sha1 of absolute path (lowercased)
    return hashlib.sha1(os.path.abspath(path).lower().encode("utf-8")).hexdigest()

def schedule_next(status: str):
    """Return a tuple of (reason, days) for the next review window."""

    normalized = (status or "").lower()

    if normalized == "vectorized":
        days = random.randint(335, 395)  # 11–13 months
        return ("vectorized→recheck_11-13m", days)

    if normalized.startswith("error") or normalized in {
        "ai_not_relevant",
        "skipped_too_short",
    }:
        days = random.randint(25, 30)
        return (f"{normalized or 'unknown'}→retry_25-30d", days)

    # Fallback: review in roughly one month
    days = random.randint(25, 30)
    return (f"{normalized or 'unspecified'}→retry_25-30d", days)

def compute_next_review_at(status: str):
    """Return (timestamp, reason) using SQLite-compatible 'YYYY-MM-DD HH:MM:SS'."""
    reason, days = schedule_next(status)
    ts = datetime.now(timezone.utc) + timedelta(days=days)
    return ts.strftime("%Y-%m-%d %H:%M:%S"), reason

def is_due(next_review_at: str | None):
    if not next_review_at:
        return True
    try:
        # Accept both 'YYYY-MM-DD HH:MM:SS' and ISO with 'T'
        s = next_review_at.replace("T", " ").replace("Z", "")
        due = datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
    except Exception:
        return True
    now = datetime.utcnow()
    return now >= due

def init_conn(db_path: str|None=None):
    db = db_path or DEFAULT_DB_PATH
    db_path_obj = Path(db)
    try:
        db_path_obj.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    conn = sqlite3.connect(str(db_path_obj), timeout=30)
    ensure_db(conn)
    conn.row_factory = sqlite3.Row
    return conn

def upsert_file(conn, fid, path, st_mtime, st_size, inode=None, device=None, content_hash=None):
    # DB timestamps set via SQLite to avoid format mismatch
    conn.execute("""
    INSERT INTO files (id, path, mtime, size, inode, device, content_hash, updated_at, first_seen_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
    ON CONFLICT(id) DO UPDATE SET
      mtime=excluded.mtime,
      size=excluded.size,
      path=excluded.path,
      inode=COALESCE(excluded.inode, files.inode),
      device=COALESCE(excluded.device, files.device),
      content_hash=COALESCE(excluded.content_hash, files.content_hash),
      deleted_at=NULL,
      updated_at=datetime('now'),
      first_seen_at=COALESCE(files.first_seen_at, excluded.first_seen_at)
    """, (
        fid,
        path,
        int(st_mtime),
        int(st_size),
        None if inode is None else int(inode),
        None if device is None else int(device),
        content_hash,
    ))
    conn.commit()

def enqueue_job_if_absent(conn, file_id):
    # skip if job queued/running
    cur = conn.execute("SELECT 1 FROM jobs WHERE file_id=? AND status IN ('queued','running') LIMIT 1", (file_id,))
    if cur.fetchone():
        return False
    jid = str(uuid.uuid4())
    conn.execute("""
    INSERT INTO jobs (job_id, file_id, enqueue_at, run_not_before, status, attempts)
    VALUES (?, ?, datetime('now'), datetime('now'), 'queued', 0)
    """, (jid, file_id))
    conn.commit()
    return True

