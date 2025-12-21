-- Normalize any existing ISO timestamps with 'T'/'Z' to SQLite's 'YYYY-MM-DD HH:MM:SS'
UPDATE jobs  SET run_not_before = replace(replace(run_not_before,'T',' '),'Z','');
UPDATE jobs  SET enqueue_at     = replace(replace(enqueue_at,'T',' '),'Z','');
UPDATE jobs  SET locked_at      = replace(replace(locked_at,'T',' '),'Z','');
UPDATE files SET first_seen_at  = replace(replace(first_seen_at,'T',' '),'Z','');
UPDATE files SET last_checked_at= replace(replace(last_checked_at,'T',' '),'Z','');
UPDATE files SET updated_at     = replace(replace(updated_at,'T',' '),'Z','');
UPDATE files SET next_review_at = replace(replace(next_review_at,'T',' '),'Z','');
