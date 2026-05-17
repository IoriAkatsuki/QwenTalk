"""L1 短期(跨 reconnect) + L2 人格 facts 长期记忆 SQLite 存储。

设计:
- L1: turns 表存历史对话；reload 浏览器后 recall_recent_turns 注入 history。
- L2: persona_facts 表存显式 key/value；get_persona_facts 注入 system prompt。
- 单连接 + Lock：SQLite single-thread 默认；webui 并发量小，全局 Lock 足够。
- 全部 SQL 参数化，避免注入。
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

DEFAULT_USER_KEY = "default"
RECALL_DEFAULT_TURNS = 6
FACTS_DEFAULT_TOP_N = 5

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_key TEXT NOT NULL DEFAULT 'default',
        started_at REAL NOT NULL,
        ended_at REAL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS turns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        ts REAL NOT NULL,
        tool_calls_json TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS persona_facts (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        importance REAL NOT NULL DEFAULT 0.5,
        updated_at REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_turns_ts ON turns(ts)",
)


class MemoryStore:
    """SQLite 持久化记忆存储。线程安全（全局 Lock 包裹写入）。"""

    def __init__(self, db_path: str) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: 允许 executor 线程访问；写入用 Lock 串行化。
        self._conn = sqlite3.connect(
            self.db_path, check_same_thread=False, isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            for stmt in _SCHEMA:
                self._conn.execute(stmt)

    # ---------- sessions ----------
    def start_session(self, user_key: str = DEFAULT_USER_KEY) -> int:
        ts = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO sessions(user_key, started_at) VALUES (?, ?)",
                (user_key, ts),
            )
            return int(cur.lastrowid)

    def end_session(self, session_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET ended_at = ? WHERE id = ?",
                (time.time(), int(session_id)),
            )

    # ---------- turns (L1) ----------
    def append_turn(
        self, session_id: int | None, role: str, content: str,
        tool_calls: object | None = None,
    ) -> None:
        import json as _json
        tc_json = _json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None
        with self._lock:
            self._conn.execute(
                "INSERT INTO turns(session_id, role, content, ts, tool_calls_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, time.time(), tc_json),
            )

    def recall_recent_turns(
        self, user_key: str = DEFAULT_USER_KEY,
        max_turns: int = RECALL_DEFAULT_TURNS,
    ) -> list[dict]:
        """跨 session 取最近 N turn (role limited to user/assistant)，按时序正序返回。"""
        with self._lock:
            cur = self._conn.execute(
                "SELECT t.role, t.content, t.ts FROM turns t "
                "LEFT JOIN sessions s ON t.session_id = s.id "
                "WHERE (s.user_key = ? OR t.session_id IS NULL) "
                "  AND t.role IN ('user', 'assistant') "
                "ORDER BY t.ts DESC LIMIT ?",
                (user_key, int(max_turns)),
            )
            rows = cur.fetchall()
        # DESC 取出后反转 → 正序（最老在前）
        return [{"role": r["role"], "content": r["content"], "ts": r["ts"]}
                for r in reversed(rows)]

    # ---------- persona_facts (L2) ----------
    def upsert_persona_fact(
        self, key: str, value: str, importance: float = 0.5,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO persona_facts(key, value, importance, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET "
                "  value = excluded.value, "
                "  importance = excluded.importance, "
                "  updated_at = excluded.updated_at",
                (key, value, float(importance), time.time()),
            )

    def get_persona_facts(self, top_n: int = FACTS_DEFAULT_TOP_N) -> list[dict]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT key, value, importance, updated_at FROM persona_facts "
                "ORDER BY importance DESC, updated_at DESC LIMIT ?",
                (int(top_n),),
            )
            rows = cur.fetchall()
        return [{"key": r["key"], "value": r["value"],
                 "importance": r["importance"], "updated_at": r["updated_at"]}
                for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


__all__ = ["MemoryStore"]
