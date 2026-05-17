#!/usr/bin/env python3
"""SQLite memory store for the DK-2500 local edge agent.

The design is intentionally dependency-free: it keeps the board usable even
when network access or Python package installs are unavailable.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

DB_PATH = Path.home() / ".hermes" / "qwentalk_memory.sqlite3"
SHORT_HALF_LIFE_HOURS = 24.0
LONG_HALF_LIFE_HOURS = 24.0 * 30.0
SHORT_EXPIRY_MULTIPLIER = 8.0
MAX_STABILITY_HOURS = 24.0 * 365.0
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", re.UNICODE)


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    db_path = db_path or DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    _init_schema(con)
    return con


def _init_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_type TEXT NOT NULL CHECK (memory_type IN ('short_term', 'long_term')),
            content TEXT NOT NULL,
            summary TEXT NOT NULL DEFAULT '',
            tags_json TEXT NOT NULL DEFAULT '[]',
            source TEXT NOT NULL DEFAULT 'agent',
            importance REAL NOT NULL DEFAULT 0.5,
            stability_hours REAL NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            last_accessed_at REAL,
            access_count INTEGER NOT NULL DEFAULT 0,
            expires_at REAL,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
        CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at);
        CREATE INDEX IF NOT EXISTS idx_memories_expires ON memories(expires_at);

        CREATE TABLE IF NOT EXISTS memory_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER,
            event_type TEXT NOT NULL,
            event_at REAL NOT NULL,
            details_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE SET NULL
        );
        """
    )
    con.commit()


def store_memory(
    content: str,
    memory_type: str = "short_term",
    importance: float = 0.5,
    tags: list[str] | None = None,
    source: str = "agent",
    half_life_hours: float | None = None,
    summary: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    content = content.strip()
    if not content:
        return {"error": "content must not be empty"}
    if memory_type not in {"short_term", "long_term"}:
        return {"error": "memory_type must be short_term or long_term"}
    importance = _clamp(float(importance), 0.0, 1.0)
    stability = float(half_life_hours or _default_half_life(memory_type))
    if stability <= 0:
        return {"error": "half_life_hours must be positive"}

    now = time.time()
    expires_at = None
    if memory_type == "short_term":
        expires_at = now + stability * SHORT_EXPIRY_MULTIPLIER * 3600.0

    with _connect() as con:
        cur = con.execute(
            """
            INSERT INTO memories (
                memory_type, content, summary, tags_json, source, importance,
                stability_hours, created_at, updated_at, expires_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_type,
                content,
                summary.strip(),
                json.dumps(tags or [], ensure_ascii=False),
                source.strip() or "agent",
                importance,
                stability,
                now,
                now,
                expires_at,
                json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
            ),
        )
        memory_id = int(cur.lastrowid)
        _event(con, memory_id, "store", {"memory_type": memory_type, "importance": importance})

    return {
        "id": memory_id,
        "memory_type": memory_type,
        "importance": importance,
        "half_life_hours": round(stability, 3),
        "expires_at": _iso(expires_at) if expires_at else None,
        "db_path": str(DB_PATH),
    }


def recall_memory(
    query: str,
    memory_type: str = "any",
    top_k: int = 5,
    reinforce: bool = True,
    include_decayed: bool = False,
) -> dict[str, Any]:
    if memory_type not in {"any", "short_term", "long_term"}:
        return {"error": "memory_type must be any, short_term, or long_term"}
    top_k = max(1, min(int(top_k), 20))
    now = time.time()
    query_tokens = _tokens(query)

    where = []
    params: list[Any] = []
    if memory_type != "any":
        where.append("memory_type = ?")
        params.append(memory_type)
    where_sql = "WHERE " + " AND ".join(where) if where else ""

    with _connect() as con:
        rows = con.execute(f"SELECT * FROM memories {where_sql}", params).fetchall()
        scored = []
        for row in rows:
            if row["expires_at"] is not None and row["expires_at"] < now and not include_decayed:
                continue
            retention = _retention(row, now)
            if retention < 0.02 and not include_decayed:
                continue
            lexical = _lexical_score(query_tokens, row["content"], row["summary"], row["tags_json"])
            if query_tokens and lexical <= 0:
                continue
            score = _score(row, retention, lexical)
            scored.append((score, retention, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = scored[:top_k]

        if reinforce and selected:
            for _, retention, row in selected:
                _reinforce(con, row, retention)
            con.commit()
            refreshed = []
            for score, _, row in selected:
                updated = con.execute("SELECT * FROM memories WHERE id = ?", (row["id"],)).fetchone()
                if updated is not None:
                    refreshed.append((score, _retention(updated, time.time()), updated))
            selected = refreshed

    return {
        "query": query,
        "memory_type": memory_type,
        "results": [_row_payload(row, score, retention) for score, retention, row in selected],
        "count": len(selected),
        "scoring": "lexical_overlap * importance * exp_decay * access_reinforcement",
    }


def reinforce_memory(memory_id: int, amount: float = 1.0) -> dict[str, Any]:
    amount = _clamp(float(amount), 0.0, 3.0)
    now = time.time()
    with _connect() as con:
        row = con.execute("SELECT * FROM memories WHERE id = ?", (int(memory_id),)).fetchone()
        if row is None:
            return {"error": f"memory id not found: {memory_id}"}
        retention = _retention(row, now)
        for _ in range(max(1, int(math.ceil(amount)))):
            _reinforce(con, row, retention)
            row = con.execute("SELECT * FROM memories WHERE id = ?", (int(memory_id),)).fetchone()
        con.commit()
        updated = con.execute("SELECT * FROM memories WHERE id = ?", (int(memory_id),)).fetchone()
    return _row_payload(updated, score=None, retention=_retention(updated, now))


def decay_memory(min_retention: float = 0.02, dry_run: bool = True) -> dict[str, Any]:
    now = time.time()
    min_retention = _clamp(float(min_retention), 0.0, 1.0)
    with _connect() as con:
        rows = con.execute("SELECT * FROM memories").fetchall()
        delete_ids = []
        for row in rows:
            expired = row["expires_at"] is not None and row["expires_at"] < now
            faded = _retention(row, now) < min_retention
            if expired or faded:
                delete_ids.append(int(row["id"]))
        if not dry_run:
            for memory_id in delete_ids:
                _event(con, memory_id, "decay_delete", {"min_retention": min_retention})
                con.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            con.commit()
    return {
        "dry_run": bool(dry_run),
        "candidate_delete_ids": delete_ids,
        "deleted": 0 if dry_run else len(delete_ids),
        "min_retention": min_retention,
    }


def memory_stats() -> dict[str, Any]:
    now = time.time()
    with _connect() as con:
        rows = con.execute("SELECT * FROM memories").fetchall()
        events = con.execute("SELECT COUNT(*) AS n FROM memory_events").fetchone()["n"]
    by_type = {"short_term": 0, "long_term": 0}
    active = 0
    for row in rows:
        by_type[row["memory_type"]] += 1
        if row["expires_at"] is None or row["expires_at"] >= now:
            active += 1
    return {
        "db_path": str(DB_PATH),
        "total": len(rows),
        "active": active,
        "by_type": by_type,
        "events": int(events),
        "short_half_life_hours_default": SHORT_HALF_LIFE_HOURS,
        "long_half_life_hours_default": LONG_HALF_LIFE_HOURS,
    }


def _default_half_life(memory_type: str) -> float:
    return LONG_HALF_LIFE_HOURS if memory_type == "long_term" else SHORT_HALF_LIFE_HOURS


def _event(con: sqlite3.Connection, memory_id: int | None, event_type: str, details: dict[str, Any]) -> None:
    con.execute(
        "INSERT INTO memory_events(memory_id, event_type, event_at, details_json) VALUES (?, ?, ?, ?)",
        (memory_id, event_type, time.time(), json.dumps(details, ensure_ascii=False, sort_keys=True)),
    )


def _reinforce(con: sqlite3.Connection, row: sqlite3.Row, retention: float) -> None:
    now = time.time()
    access_count = int(row["access_count"]) + 1
    stability = float(row["stability_hours"])
    importance = float(row["importance"])
    strengthened = min(MAX_STABILITY_HOURS, stability * (1.12 + 0.10 * importance) + 2.0 * retention)
    memory_type = row["memory_type"]
    expires_at = row["expires_at"]
    if memory_type == "short_term" and (importance >= 0.85 or access_count >= 3):
        memory_type = "long_term"
        expires_at = None
    elif memory_type == "short_term":
        expires_at = now + strengthened * SHORT_EXPIRY_MULTIPLIER * 3600.0
    con.execute(
        """
        UPDATE memories
           SET memory_type = ?, stability_hours = ?, last_accessed_at = ?,
               access_count = ?, updated_at = ?, expires_at = ?
         WHERE id = ?
        """,
        (memory_type, strengthened, now, access_count, now, expires_at, row["id"]),
    )
    _event(
        con,
        int(row["id"]),
        "reinforce",
        {"retention": retention, "stability_hours": strengthened, "access_count": access_count},
    )


def _retention(row: sqlite3.Row, now: float) -> float:
    anchor = row["last_accessed_at"] or row["updated_at"] or row["created_at"]
    elapsed_hours = max(0.0, (now - float(anchor)) / 3600.0)
    half_life = max(0.001, float(row["stability_hours"]))
    return math.exp(-math.log(2.0) * elapsed_hours / half_life)


def _score(row: sqlite3.Row, retention: float, lexical: float) -> float:
    importance = float(row["importance"])
    access_boost = 1.0 + math.log1p(int(row["access_count"])) * 0.12
    type_boost = 1.10 if row["memory_type"] == "long_term" else 1.0
    return lexical * (0.35 + importance) * retention * access_boost * type_boost


def _lexical_score(query_tokens: set[str], content: str, summary: str, tags_json: str) -> float:
    if not query_tokens:
        return 1.0
    memory_tokens = _tokens(content)
    memory_tokens.update(_tokens(summary))
    try:
        for tag in json.loads(tags_json):
            memory_tokens.update(_tokens(str(tag)))
    except Exception:
        pass
    if not memory_tokens:
        return 0.0
    overlap = len(query_tokens & memory_tokens)
    if overlap == 0:
        return 0.0
    return overlap / math.sqrt(len(query_tokens) * len(memory_tokens))


def _tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in _TOKEN_RE.finditer(text or ""):
        token = match.group(0).lower().strip()
        if not token:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            tokens.update(token)
            tokens.update(token[i:i + 2] for i in range(max(0, len(token) - 1)))
            tokens.update(token[i:i + 3] for i in range(max(0, len(token) - 2)))
        else:
            tokens.add(token)
    return tokens


def _row_payload(row: sqlite3.Row, score: float | None, retention: float) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "memory_type": row["memory_type"],
        "content": row["content"],
        "summary": row["summary"],
        "tags": _json_list(row["tags_json"]),
        "source": row["source"],
        "importance": round(float(row["importance"]), 3),
        "retention": round(float(retention), 4),
        "score": None if score is None else round(float(score), 4),
        "half_life_hours": round(float(row["stability_hours"]), 3),
        "access_count": int(row["access_count"]),
        "created_at": _iso(row["created_at"]),
        "last_accessed_at": _iso(row["last_accessed_at"]),
        "expires_at": _iso(row["expires_at"]),
    }


def _json_list(raw: str) -> list[str]:
    try:
        data = json.loads(raw or "[]")
    except Exception:
        return []
    return [str(x) for x in data] if isinstance(data, list) else []


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(float(ts)))


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
