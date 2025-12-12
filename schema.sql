-- Fresh schema for state.db (drop your old DB file to start clean)
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS files (
  id TEXT PRIMARY KEY,
  path TEXT NOT NULL,
  mtime INTEGER NOT NULL,
  size INTEGER NOT NULL,
  status TEXT,                -- vectorized | ai_not_relevant | skipped_too_short | error_*
  last_result TEXT,           -- JSON
  last_error TEXT,
  updated_at TEXT NOT NULL,
  first_seen_at TEXT,
  last_checked_at TEXT,
  should_reingest INTEGER,    -- 1 if accepted else 0
  review_reason TEXT,
  next_review_at TEXT,        -- ISO-UTC
  priority INTEGER DEFAULT 100,
  error_count INTEGER DEFAULT 0,
  inode INTEGER,
  device INTEGER,
  content_hash TEXT,
  deleted_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_files_next_review ON files(next_review_at);
CREATE INDEX IF NOT EXISTS idx_files_status ON files(status);
CREATE INDEX IF NOT EXISTS idx_files_inode_device ON files(device, inode);
CREATE INDEX IF NOT EXISTS idx_files_content_hash ON files(content_hash);

CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY,
  file_id TEXT NOT NULL,
  enqueue_at TEXT NOT NULL,
  run_not_before TEXT NOT NULL,
  status TEXT NOT NULL,       -- queued | running | done | failed | canceled
  attempts INTEGER NOT NULL DEFAULT 0,
  locked_at TEXT,
  worker_id TEXT,
  last_error TEXT,
  FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_jobs_q ON jobs(status, run_not_before);

-- Add decision_log table and indexes (idempotent)
CREATE TABLE IF NOT EXISTS decision_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT DEFAULT (datetime('now')),
    job_id TEXT,
    file_id TEXT,
    step TEXT,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_decision_log_step ON decision_log(step, ts);

CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id TEXT NOT NULL,
    label TEXT,
    reference TEXT NOT NULL,
    mime TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_images_file_id ON images(file_id);
