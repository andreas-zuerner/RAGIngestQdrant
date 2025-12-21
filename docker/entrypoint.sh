#!/usr/bin/env bash
set -euo pipefail

# Persistent runtime directory (db + logs + env)
DATA_DIR="${DATA_DIR:-/data}"
DB_DIR="${DB_DIR:-${DATA_DIR}/DocumentDatabase}"
LOG_DIR="${LOG_DIR:-${DATA_DIR}/logs}"
ENV_FILE="${ENV_FILE:-${DATA_DIR}/.env.local}"

mkdir -p "${DB_DIR}" "${LOG_DIR}"

# Create a minimal default env file on first run (persisted in the volume).
# This keeps config editable without rebuilding images, and prevents surprises.
if [[ ! -f "${ENV_FILE}" ]]; then
  cat > "${ENV_FILE}" <<'EOF'
# --- RAGIngestQdrant runtime config (Docker default) ---
# Adjust as needed. This file is persisted in the rag_data volume.

# SQLite DB (override is supported; scan_scheduler reads initENV.DB_PATH)
DB_PATH=/data/DocumentDatabase/state.db
SCHEDULER_LOG=/data/logs/scan_scheduler.log

# IMPORTANT: These endpoints point to EXISTING containers/services.
# They must not be bound/started by rag-app.
# Adjust to your host/LXC IPs.
OLLAMA_HOST=http://127.0.0.1:11434
DOCLING_SERVE_URL=http://127.0.0.1:5001/v1/convert/file

# Brain service (external)
BRAIN_URL=http://127.0.0.1:8080
BRAIN_API_KEY=change-me
BRAIN_COLLECTION=documents
BRAIN_CHUNK_TOKENS=400
BRAIN_REQUEST_TIMEOUT=120

# Nextcloud (external)
NEXTCLOUD_BASE_URL=http://127.0.0.1:8080
NEXTCLOUD_USER=andreas
NEXTCLOUD_TOKEN=
NEXTCLOUD_DOC_DIR=/RAGdocuments
NEXTCLOUD_IMAGE_DIR=/RAG-images

# Web GUI
WEB_GUI_SECRET=dev-secret
WEB_GUI_HOST=0.0.0.0
WEB_GUI_PORT=8088

# Docling async (if you use it)
DOCLING_SERVE_USE_ASYNC=True
DOCLING_SERVE_TIMEOUT=300
DOCLING_SERVE_ASYNC_TIMEOUT=900
DOCLING_SERVE_ASYNC_POLL_INTERVAL=5
EOF
  echo "[entrypoint] created ${ENV_FILE}"
fi

# Export ENV_FILE for initENV.py; also source it so supervisor children inherit values.
export ENV_FILE
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

exec "$@"
