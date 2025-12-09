#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$ROOT_DIR/.env.local"
PID_DIR="$ROOT_DIR/.run"
PID_FILE="$PID_DIR/scan_scheduler.pid"
LOG_DIR="$ROOT_DIR/log"
LOG_FILE="$LOG_DIR/scan_scheduler.log"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

usage() {
    cat <<USAGE
Usage: $(basename "$0") <start|stop|restart|status>

start    Start the scan scheduler (and implicitly the worker)
stop     Stop the scan scheduler and its managed worker
restart  Restart the scheduler process
status   Show whether the scheduler is running
USAGE
}

require_env() {
    if [[ ! -f "$ENV_FILE" ]]; then
        echo "Missing $ENV_FILE. Copy .env.local.example and adjust the values." >&2
        exit 1
    fi
}

load_env() {
    require_env
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
}

ensure_python() {
    if [[ ! -x "$PYTHON_BIN" ]]; then
        echo "Python virtualenv not found at $PYTHON_BIN" >&2
        echo "Create it first, e.g. python3 -m venv .venv && .venv/bin/pip install -r DEPENDENCIES.md" >&2
        exit 1
    fi
}

is_running() {
    if [[ ! -f "$PID_FILE" ]]; then
        return 1
    fi
    local pid
    pid=$(<"$PID_FILE")
    if kill -0 "$pid" 2>/dev/null; then
        return 0
    fi
    rm -f "$PID_FILE"
    return 1
}

start_scheduler() {
    if is_running; then
        local pid
        pid=$(<"$PID_FILE")
        echo "Scheduler already running (PID $pid)"
        return 0
    fi
    load_env
    ensure_python
    mkdir -p "$PID_DIR" "$LOG_DIR"
    rm -f "$PID_FILE"
    echo "Starting scan scheduler..."
    (cd "$ROOT_DIR" && SCAN_SCHEDULER_PID_FILE="$PID_FILE" "$PYTHON_BIN" "$ROOT_DIR/scan_scheduler.py" >>"$LOG_FILE" 2>&1 &)
    local waited=0
    local max_wait=50
    while [[ ! -f "$PID_FILE" && $waited -lt $max_wait ]]; do
        sleep 0.1
        waited=$((waited + 1))
    done
    if is_running; then
        local pid
        pid=$(<"$PID_FILE")
        echo "Scheduler started (PID $pid)"
        echo "Logs: $LOG_FILE"
    else
        echo "Failed to start scheduler. See $LOG_FILE for details." >&2
        exit 1
    fi
}

stop_scheduler() {
    if ! is_running; then
        echo "Scheduler is not running"
        return 0
    fi
    local pid
    pid=$(<"$PID_FILE")
    echo "Stopping scheduler (PID $pid)..."
    if ! kill "$pid" 2>/dev/null; then
        echo "Unable to signal scheduler (already dead?)." >&2
    else
        local waited=0
        local timeout=${WORKER_STOP_TIMEOUT:-20}
        while kill -0 "$pid" 2>/dev/null; do
            sleep 1
            waited=$((waited + 1))
            if (( waited >= timeout )); then
                echo "Force killing scheduler (PID $pid)" >&2
                kill -9 "$pid" 2>/dev/null || true
                break
            fi
        done
    fi
    rm -f "$PID_FILE"
    echo "Scheduler stopped"
}

status_scheduler() {
    if is_running; then
        local pid
        pid=$(<"$PID_FILE")
        echo "Scheduler running (PID $pid)"
    else
        echo "Scheduler not running"
    fi
}

case "${1:-}" in
    start)
        start_scheduler
        ;;
    stop)
        stop_scheduler
        ;;
    restart)
        stop_scheduler
        start_scheduler
        ;;
    status)
        status_scheduler
        ;;
    *)
        usage
        exit 1
        ;;
esac
