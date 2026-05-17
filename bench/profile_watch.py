#!/usr/bin/env python3
"""Low-impact heartbeat watcher for long Qwen3.6 profiling runs."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def run(cmd: list[str], timeout: int = 10) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
        return {"rc": proc.returncode, "stdout": proc.stdout[-12000:], "stderr": proc.stderr[-4000:]}
    except Exception as exc:
        return {"rc": -1, "stdout": "", "stderr": repr(exc)}


def health() -> str:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return f"ERR {type(exc).__name__}: {exc}"


def tail(path: Path, n: int = 8) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-n:]


def latest_files(out_dir: Path, limit: int = 40) -> dict[str, Any]:
    try:
        rows: list[tuple[float, str]] = []
        for path in out_dir.rglob("*"):
            if not path.is_file():
                continue
            stat = path.stat()
            ts = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S.%f")
            rows.append((stat.st_mtime, f"{ts} {stat.st_size} {path}"))
        rows.sort(key=lambda item: item[0])
        return {"rc": 0, "stdout": "\n".join(row for _, row in rows[-limit:]), "stderr": ""}
    except Exception as exc:
        return {"rc": -1, "stdout": "", "stderr": repr(exc)}


def snapshot(out_dir: Path, pid: int | None) -> dict[str, Any]:
    summary = out_dir / "summary.jsonl"
    usage = shutil.disk_usage("/home")
    free_gib = usage.free / (1024 ** 3)
    alerts: list[str] = []
    if free_gib < 2.0:
        alerts.append(f"home_free_low_gib={free_gib:.2f}")
    row = {
        "time": now_iso(),
        "watcher": "qwen36_profile_watch",
        "out_dir": str(out_dir),
        "main_pid": pid,
        "main_ps": run(["ps", "-p", str(pid), "-o", "pid,ppid,stat,etime,pcpu,pmem,args", "--no-headers"]) if pid else None,
        "qwen_processes": run(["pgrep", "-a", "-f", "qwen36_overnight_profile|llama-bench|llama-server.*Qwen3.6|pidstat|turbostat"]),
        "free": run(["free", "-h"]),
        "df_home": run(["df", "-h", "/home"]),
        "health_8080": health(),
        "home_free_gib": round(free_gib, 3),
        "alerts": alerts,
        "summary_size": summary.stat().st_size if summary.exists() else 0,
        "summary_tail": tail(summary),
        "latest_files": latest_files(out_dir),
    }
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Append heartbeat snapshots for a Qwen3.6 profiling run")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pid", type=int, default=None)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--hours", type=float, default=8.5)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    heartbeat = args.out / "heartbeat.jsonl"
    deadline = time.time() + args.hours * 3600
    while time.time() < deadline:
        row = snapshot(args.out, args.pid)
        with heartbeat.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        main_alive = args.pid is not None and subprocess.run(["kill", "-0", str(args.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        if args.pid is not None and not main_alive:
            time.sleep(60)
            row = snapshot(args.out, args.pid)
            row["note"] = "main process exited; final post-exit snapshot"
            with heartbeat.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            break
        time.sleep(max(10, args.interval))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
