import json
import sqlite3
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from variables import initENV


class Settings(BaseSettings):
    DB_PATH: str = initENV.DB_PATH
    TABLE_API_KEY: str | None = None  # set via env, recommended

settings = Settings()
app = FastAPI(title="Table API", version="1.0.0")


def check_api_key(x_api_key: str | None = Header(default=None)) -> bool:
    if not settings.TABLE_API_KEY:
        return True
    if not x_api_key or x_api_key != settings.TABLE_API_KEY:
        raise HTTPException(401, "Invalid API key")
    return True


def db_connect() -> sqlite3.Connection:
    try:
        conn = sqlite3.connect(settings.DB_PATH, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"DB connect failed: {exc}")


class TablePreviewOut(BaseModel):
    table_id: str
    file_id: str
    source_path: str | None = None
    label: str | None = None
    offset: int
    limit: int
    row_count: int
    rows: list[dict[str, Any]]


class TableStatsOut(BaseModel):
    table_id: str
    file_id: str
    source_path: str | None = None
    label: str | None = None
    row_count: int
    columns: list[str]


@app.get("/health")
def health(_: bool = Depends(check_api_key)):
    return {"ok": True}


@app.get("/tables/{table_id}/preview", response_model=TablePreviewOut)
def table_preview(
    table_id: str,
    limit: int = 25,
    offset: int = 0,
    _: bool = Depends(check_api_key),
):
    if limit < 1:
        limit = 1
    if limit > 200:
        limit = 200
    if offset < 0:
        offset = 0

    conn = db_connect()
    try:
        reg = conn.execute(
            "SELECT table_id, file_id, source_path, label FROM table_registry WHERE table_id=?",
            (table_id,),
        ).fetchone()
        if not reg:
            raise HTTPException(404, f"Unknown table_id: {table_id}")

        row_count = conn.execute(
            "SELECT COUNT(*) AS c FROM table_data WHERE table_id=?",
            (table_id,),
        ).fetchone()["c"]

        rows_raw = conn.execute(
            """
            SELECT row_idx, row_json
            FROM table_data
            WHERE table_id=?
            ORDER BY row_idx
            LIMIT ? OFFSET ?
            """,
            (table_id, limit, offset),
        ).fetchall()

        rows: list[dict[str, Any]] = []
        for r in rows_raw:
            try:
                obj = json.loads(r["row_json"])
                if isinstance(obj, dict):
                    obj["_row_idx"] = r["row_idx"]
                    rows.append(obj)
                else:
                    rows.append({"_row_idx": r["row_idx"], "value": obj})
            except Exception:
                rows.append({"_row_idx": r["row_idx"], "raw": r["row_json"]})

        return {
            "table_id": reg["table_id"],
            "file_id": reg["file_id"],
            "source_path": reg["source_path"],
            "label": reg["label"],
            "offset": offset,
            "limit": limit,
            "row_count": int(row_count or 0),
            "rows": rows,
        }
    finally:
        conn.close()


@app.get("/tables/{table_id}/stats", response_model=TableStatsOut)
def table_stats(
    table_id: str,
    sample_rows: int = 50,
    _: bool = Depends(check_api_key),
):
    if sample_rows < 1:
        sample_rows = 1
    if sample_rows > 500:
        sample_rows = 500

    conn = db_connect()
    try:
        reg = conn.execute(
            "SELECT table_id, file_id, source_path, label FROM table_registry WHERE table_id=?",
            (table_id,),
        ).fetchone()
        if not reg:
            raise HTTPException(404, f"Unknown table_id: {table_id}")

        row_count = conn.execute(
            "SELECT COUNT(*) AS c FROM table_data WHERE table_id=?",
            (table_id,),
        ).fetchone()["c"]

        rows_raw = conn.execute(
            """
            SELECT row_json
            FROM table_data
            WHERE table_id=?
            ORDER BY row_idx
            LIMIT ?
            """,
            (table_id, sample_rows),
        ).fetchall()

        cols: set[str] = set()
        for r in rows_raw:
            try:
                obj = json.loads(r["row_json"])
                if isinstance(obj, dict):
                    cols.update(obj.keys())
            except Exception:
                continue

        return {
            "table_id": reg["table_id"],
            "file_id": reg["file_id"],
            "source_path": reg["source_path"],
            "label": reg["label"],
            "row_count": int(row_count or 0),
            "columns": sorted(cols),
        }
    finally:
        conn.close()
