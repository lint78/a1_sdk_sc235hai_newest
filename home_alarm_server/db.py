"""SQLite 持久化模块：建表、写入、查询。"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("db")

DB_PATH = Path(__file__).resolve().parent / "alarm_service.db"

# MQTT 回调线程与 FastAPI 异步线程都会访问数据库，用锁串行化写操作。
_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def get_conn() -> sqlite3.Connection:
    """返回模块级共享连接（跨线程安全，配合 _lock 使用）。"""
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
    return _conn


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {key: row[key] for key in row.keys()}


def init_db() -> None:
    """启动时确保数据库文件与表存在。"""
    conn = get_conn()
    with _lock:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alarm_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                alarm_type TEXT NOT NULL,
                alarm_state TEXT NOT NULL,
                frame_index INTEGER,
                source TEXT,
                ts_millis INTEGER,
                server_time TEXT NOT NULL,
                raw_json TEXT
            )
            """
        )
        conn.commit()
    logger.info("数据库已就绪: %s", DB_PATH)


def insert_event(data: dict) -> dict | None:
    """写入一条告警记录，返回含 id / server_time 的完整记录。"""
    conn = get_conn()
    server_time = datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        raw_json = json.dumps(data, ensure_ascii=False)
    except (TypeError, ValueError):
        raw_json = None

    with _lock:
        cur = conn.execute(
            """
            INSERT INTO alarm_events
                (device_id, alarm_type, alarm_state, frame_index, source, ts_millis, server_time, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data.get("device_id"),
                data.get("type"),
                data.get("state"),
                data.get("frame"),
                data.get("source"),
                data.get("ts_millis"),
                server_time,
                raw_json,
            ),
        )
        conn.commit()
        new_id = cur.lastrowid

    return get_event(new_id)


def get_event(event_id: int) -> dict | None:
    conn = get_conn()
    with _lock:
        cur = conn.execute("SELECT * FROM alarm_events WHERE id = ?", (event_id,))
        row = cur.fetchone()
    return _row_to_dict(row) if row else None


def get_recent_alarms(limit: int = 100) -> list[dict]:
    """按 id 倒序返回最近 limit 条告警。"""
    conn = get_conn()
    with _lock:
        cur = conn.execute(
            "SELECT * FROM alarm_events ORDER BY id DESC LIMIT ?", (limit,)
        )
        rows = cur.fetchall()
    return [_row_to_dict(row) for row in rows]
