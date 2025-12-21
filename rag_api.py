"""
Minimal FastAPI service for RAGIngestQdrant.

Purpose:
- Provide a stable HTTP surface for reading runtime status + SQLite content
  (e.g., tables stored in DocumentDatabase/state.db).
- Keep it minimal and non-invasive: no new ingestion logic, no heavy refactors.

Notes:
- Uses DB_PATH from variables/initENV.py (which respects ENV_FILE and environment variables).
- If your schema evolves, adjust the SQL queries accordingly.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from variables import initENV

app = FastAPI(title="RAGIngestQdrant API", version="0.1.0")


def _connect() -> sqlite3.Connection:
    db_path = initENV.DB_PATH
    # Ensure directory exists; DB file may be created lazily by the pipeline
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "db_path": initENV.DB_PATH,
        "brain_url": initENV.BRAIN_URL,
        "docling_url": initENV.DOCLING_SERVE_URL,
        "ollama_host": os.environ.get("OLLAMA_HOST", ""),
    }


@app.get("/db/tables")
def list_db_tables() -> Dict[str, Any]:
    try:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        return {"tables": [r["name"] for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tables/registry")
def list_table_registry(limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    """
    Lists ingested tables (if the optional table pipeline is enabled).

    Expected schema (based on your current approach):
    - table_registry(table_id TEXT PRIMARY KEY, ... metadata columns ...)
    - table_data(table_id TEXT, sheet_name TEXT, row_index INT, data_json TEXT, ...)

    If these tables do not exist yet, the endpoint returns 404 with a helpful message.
    """
    try:
        with _connect() as conn:
            # check existence
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='table_registry'"
            ).fetchone()
            if not exists:
                raise HTTPException(
                    status_code=404,
                    detail="table_registry not found in SQLite DB. Ingest at least one spreadsheet with table mode enabled.",
                )
            rows = conn.execute(
                "SELECT * FROM table_registry ORDER BY rowid DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return {"items": [dict(r) for r in rows], "limit": limit, "offset": offset}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tables/{table_id}/rows")
def get_table_rows(
    table_id: str,
    sheet_name: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    Returns stored rows for a given table_id (optionally filtered by sheet_name).

    Assumes the row payload is stored in table_data.data_json or a similar column.
    If your column naming differs, adjust the SQL here.
    """
    try:
        with _connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='table_data'"
            ).fetchone()
            if not exists:
                raise HTTPException(
                    status_code=404,
                    detail="table_data not found in SQLite DB. Enable table ingestion and ingest at least one spreadsheet.",
                )

            if sheet_name:
                rows = conn.execute(
                    "SELECT * FROM table_data WHERE table_id=? AND sheet_name=? ORDER BY row_index LIMIT ? OFFSET ?",
                    (table_id, sheet_name, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM table_data WHERE table_id=? ORDER BY sheet_name, row_index LIMIT ? OFFSET ?",
                    (table_id, limit, offset),
                ).fetchall()

        return {
            "table_id": table_id,
            "sheet_name": sheet_name,
            "limit": limit,
            "offset": offset,
            "rows": [dict(r) for r in rows],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
