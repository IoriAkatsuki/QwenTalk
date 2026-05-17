#!/usr/bin/env python3
"""L4 KV Cache Persistence Layer — 设计 + 原型

把 llama-server slot KV cache 持久化到 PC801（板卡 USB 3.2 Gen 2x1 NVMe），
支持跨 session 续 context。灵感来源 antirez/ds4 的 first-class disk KV 设计。

详见 docs/L4_kv_persistence.md。

模式：
  - online：通过 HTTP API 调 llama-server /slots/{id}?action=save|restore
  - offline / dummy：跳过 HTTP，直接写 dummy bytes（本地基准测试）
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import urllib.request as _ureq
    import urllib.error as _uerr
except Exception:  # pragma: no cover
    _ureq = None  # type: ignore
    _uerr = None  # type: ignore

DEFAULT_BASE_DIR = Path("/home/intel/kv_sessions")
DEFAULT_LLM_URL = "http://127.0.0.1:8080"
META_SUFFIX = ".meta.json"
KV_SUFFIX = ".kvbin"
HEADER_BYTES = 1 << 20  # 1 MB head hash window


class KVPersistenceError(Exception):
    """L4 层基类异常。"""


class KVPersistenceUnavailable(KVPersistenceError):
    """llama-server 未启用 --slot-save-path 或不可达。"""


@dataclass
class SessionMeta:
    session_id: str
    model_hash: str
    ctx_used: int
    n_ctx: int
    created_at: float
    size_bytes: int = 0
    tags: list[str] = field(default_factory=list)
    user_summary: str = ""

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False, sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, payload: str) -> "SessionMeta":
        return cls(**json.loads(payload))


def _sha1_head(path: Path, n: int = HEADER_BYTES) -> str:
    """模型文件头 N 字节 sha1 作为指纹（避免读 GB 级 GGUF）。"""
    if not path.exists():
        return "missing"
    sha = hashlib.sha1()
    with path.open("rb") as fh:
        sha.update(fh.read(n))
    return sha.hexdigest()


def _http_post(url: str, body: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    if _ureq is None:
        raise KVPersistenceUnavailable("urllib not available")
    data = json.dumps(body).encode("utf-8")
    req = _ureq.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with _ureq.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8") or "{}")
    except _uerr.HTTPError as exc:  # type: ignore[union-attr]
        raise KVPersistenceUnavailable(f"HTTP {exc.code}: {exc.read().decode()}") from exc
    except Exception as exc:
        raise KVPersistenceUnavailable(f"POST {url} failed: {exc}") from exc


class KVStore:
    """Session 级 KV cache 持久化。线程不安全，单消费者使用。"""

    def __init__(
        self,
        base_dir: str | Path = DEFAULT_BASE_DIR,
        llm_url: str | None = DEFAULT_LLM_URL,
        model_path: str | Path | None = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.llm_url = llm_url
        self.model_path = Path(model_path) if model_path else None
        self._model_hash_cache: str | None = None

    # ---------- public API ----------

    def save_session(self, session_id: str, slot_id: int = 0, **meta_kwargs: Any) -> SessionMeta:
        """落盘 slot KV。online 模式调 llama-server；offline 模式写 dummy bytes。"""
        kv_path = self._kv_path(session_id)
        if self.llm_url:
            self._llm_save(slot_id, kv_path)
        else:
            self._dummy_write(kv_path, meta_kwargs.pop("_dummy_bytes", 1 << 20))
        size = kv_path.stat().st_size if kv_path.exists() else 0
        meta = SessionMeta(
            session_id=session_id,
            model_hash=self._model_hash(),
            ctx_used=int(meta_kwargs.pop("ctx_used", 0)),
            n_ctx=int(meta_kwargs.pop("n_ctx", 0)),
            created_at=time.time(),
            size_bytes=size,
            tags=list(meta_kwargs.pop("tags", []) or []),
            user_summary=str(meta_kwargs.pop("user_summary", "")),
        )
        self._meta_path(session_id).write_text(meta.to_json(), encoding="utf-8")
        return meta

    def load_session(self, session_id: str, slot_id: int = 0) -> bool:
        """从盘恢复 slot KV。返回 True 表示成功；False 表示需要 fallback。"""
        kv_path = self._kv_path(session_id)
        meta_path = self._meta_path(session_id)
        if not kv_path.exists() or not meta_path.exists():
            return False
        meta = SessionMeta.from_json(meta_path.read_text(encoding="utf-8"))
        if not self._model_matches(meta):
            return False
        if not self.llm_url:
            return True  # offline / dummy 模式视为成功
        try:
            self._llm_restore(slot_id, kv_path)
        except KVPersistenceUnavailable:
            return False
        return True

    def list_sessions(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for meta_file in sorted(self.base_dir.glob(f"*{META_SUFFIX}")):
            try:
                meta = SessionMeta.from_json(meta_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            out.append(meta.__dict__)
        return out

    def gc_old_sessions(self, max_age_days: float = 30.0, max_total_gb: float = 200.0) -> dict[str, Any]:
        """LRU + 容量上限 GC。先删过期，再按 created_at 删到 < max_total_gb。"""
        sessions = sorted(self.list_sessions(), key=lambda s: s.get("created_at", 0))
        deleted, kept = self._gc_expired(sessions, max_age_days)
        deleted += self._gc_capacity(kept, max_total_gb)
        return {"deleted": deleted, "remaining": len(self.list_sessions())}

    # ---------- llama-server bridge ----------

    def _llm_save(self, slot_id: int, kv_path: Path) -> None:
        url = f"{self.llm_url}/slots/{int(slot_id)}?action=save"
        _http_post(url, {"filename": kv_path.name})
        # llama-server 写到 --slot-save-path/{filename}，确保我们能看到
        if not kv_path.exists():
            # 部署侧 slot_save_path 可能不在 self.base_dir；这里做 best-effort 探测
            raise KVPersistenceUnavailable(
                f"slot saved but file not found at {kv_path}; check --slot-save-path"
            )

    def _llm_restore(self, slot_id: int, kv_path: Path) -> None:
        url = f"{self.llm_url}/slots/{int(slot_id)}?action=restore"
        _http_post(url, {"filename": kv_path.name})

    # ---------- helpers ----------

    def _kv_path(self, session_id: str) -> Path:
        return self.base_dir / f"{session_id}{KV_SUFFIX}"

    def _meta_path(self, session_id: str) -> Path:
        return self.base_dir / f"{session_id}{META_SUFFIX}"

    def _model_hash(self) -> str:
        if self._model_hash_cache is not None:
            return self._model_hash_cache
        if self.model_path is None:
            self._model_hash_cache = "unknown"
        else:
            self._model_hash_cache = _sha1_head(self.model_path)
        return self._model_hash_cache

    def _model_matches(self, meta: SessionMeta) -> bool:
        if meta.model_hash in {"unknown", "missing"}:
            return True  # offline / dummy 测试
        return meta.model_hash == self._model_hash()

    @staticmethod
    def _dummy_write(kv_path: Path, n_bytes: int) -> None:
        """写 n_bytes 伪随机数据，用于离线带宽基准。"""
        chunk = os.urandom(min(n_bytes, 4 << 20))  # 最大单次 4 MB 随机
        with kv_path.open("wb") as fh:
            written = 0
            while written < n_bytes:
                take = min(len(chunk), n_bytes - written)
                fh.write(chunk[:take])
                written += take
            fh.flush()
            os.fsync(fh.fileno())

    def _gc_expired(self, sessions: list[dict[str, Any]], max_age_days: float) -> tuple[int, list[dict]]:
        cutoff = time.time() - max_age_days * 86400.0
        deleted = 0
        kept: list[dict[str, Any]] = []
        for s in sessions:
            if float(s.get("created_at", 0)) < cutoff:
                self._delete_session(s["session_id"])
                deleted += 1
            else:
                kept.append(s)
        return deleted, kept

    def _gc_capacity(self, kept: list[dict[str, Any]], max_total_gb: float) -> int:
        budget = max_total_gb * (1 << 30)
        total = sum(int(s.get("size_bytes", 0)) for s in kept)
        deleted = 0
        for s in kept:
            if total <= budget:
                break
            self._delete_session(s["session_id"])
            total -= int(s.get("size_bytes", 0))
            deleted += 1
        return deleted

    def _delete_session(self, session_id: str) -> None:
        for path in (self._kv_path(session_id), self._meta_path(session_id)):
            if path.exists():
                path.unlink()

    def storage_status(self) -> dict[str, Any]:
        usage = shutil.disk_usage(self.base_dir)
        used = sum(p.stat().st_size for p in self.base_dir.glob(f"*{KV_SUFFIX}"))
        return {
            "base_dir": str(self.base_dir),
            "disk_total_gb": round(usage.total / (1 << 30), 2),
            "disk_free_gb": round(usage.free / (1 << 30), 2),
            "kv_used_gb": round(used / (1 << 30), 3),
            "session_count": len(list(self.base_dir.glob(f"*{META_SUFFIX}"))),
        }


__all__ = [
    "KVStore",
    "SessionMeta",
    "KVPersistenceError",
    "KVPersistenceUnavailable",
]
