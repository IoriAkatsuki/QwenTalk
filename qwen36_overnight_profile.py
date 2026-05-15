#!/usr/bin/env python3
"""Overnight profiling matrix for Qwen3.6 on DK-2500.

The script stops the resident Qwen3.6 server, runs llama-bench and selected
server-context probes with system monitors, then restores the 64K resident
Hermes profile.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
MODEL = Path("/home/intel/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")
LLAMA_DIR = Path("/home/intel/llama.cpp/build_sycl/bin")
LLAMA_BENCH = LLAMA_DIR / "llama-bench"
LLAMA_SERVER = LLAMA_DIR / "llama-server"
SETVARS = "/opt/intel/oneapi/setvars.sh"
HEALTH_URL = "http://127.0.0.1:8080/health"
LOCK_PATH = Path("/tmp/qwen36_overnight_profile.lock")
RESTORE_MARGIN_S = 600


@dataclass
class BenchCase:
    name: str
    args: list[str]
    timeout_s: int = 2400
    min_left_s: int = 300
    repetitions: int = 2


@dataclass
class ServerCase:
    name: str
    ctx: int
    prefill_chars: int = 9000
    cache_k: str = "q4_0"
    cache_v: str = "q4_0"
    flash_attn: int = 1
    timeout_s: int = 1800
    min_left_s: int = 420
    deadline: float = 0.0


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def run_quiet(cmd: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=timeout)


def which(name: str) -> str | None:
    return shutil.which(name)


def sudo_available() -> bool:
    return subprocess.run(["sudo", "-n", "true"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def source_cmd(argv: list[str]) -> list[str]:
    quoted = " ".join(shlex.quote(a) for a in argv)
    return ["bash", "-lc", f"source {SETVARS} >/dev/null 2>&1; exec {quoted}"]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stop_process(proc: subprocess.Popen[Any], grace_s: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        proc.terminate()
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        proc.wait(timeout=grace_s)


def stop_monitors(monitors: list[subprocess.Popen[Any]]) -> None:
    for proc in monitors:
        stop_process(proc, grace_s=2.0)


def start_monitors(case_dir: Path, target_pid: int | None) -> list[subprocess.Popen[Any]]:
    monitors: list[tuple[str, list[str]]] = []
    if which("vmstat"):
        monitors.append(("vmstat.txt", ["vmstat", "-SM", "1"]))
    if which("iostat"):
        monitors.append(("iostat.txt", ["iostat", "-xm", "1"]))
    if target_pid is not None and which("pidstat"):
        monitors.insert(0, ("pidstat.txt", ["pidstat", "-p", str(target_pid), "-durh", "1"]))

    turbostat_cmd = ["turbostat", "--Summary", "--quiet", "--interval", "1"]
    if which("turbostat"):
        monitors.append(("turbostat.txt", (["sudo", "-n"] if sudo_available() else []) + turbostat_cmd))

    if which("intel_gpu_top"):
        gpu_cmd = ["intel_gpu_top", "-J", "-s", "1000"]
        monitors.append(("intel_gpu_top.jsonl", (["sudo", "-n"] if sudo_available() else []) + gpu_cmd))

    procs: list[subprocess.Popen[Any]] = []
    for filename, cmd in monitors:
        out = (case_dir / filename).open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=ROOT, start_new_session=True)
        out.close()
        procs.append(proc)
    return procs


def acquire_lock() -> Any:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_file = LOCK_PATH.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_file.close()
        raise RuntimeError(f"another qwen36 profiling run holds {LOCK_PATH}") from exc
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(f"{os.getpid()} {now_iso()}\n")
    lock_file.flush()
    return lock_file


def find_qwen_server_pids(port: int = 8080) -> list[int]:
    proc = run_quiet(["ps", "-eo", "pid=,comm=,args="], timeout=10)
    pids: list[int] = []
    for line in proc.stdout.splitlines():
        parts = line.strip().split(maxsplit=2)
        if len(parts) < 3:
            continue
        pid_s, comm, args = parts
        if (
            comm == "llama-server"
            and "Qwen3.6-35B-A3B-UD-IQ2_M.gguf" in args
            and (f"--port {port}" in args or f"--port={port}" in args)
        ):
            try:
                pids.append(int(pid_s))
            except ValueError:
                pass
    return pids


def stop_qwen_servers() -> None:
    pids = find_qwen_server_pids()
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + 20
    while time.time() < deadline and find_qwen_server_pids():
        time.sleep(0.5)
    for pid in find_qwen_server_pids():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for stale in ["/tmp/qwen36_64k_q4_start.pid"]:
        try:
            Path(stale).unlink()
        except FileNotFoundError:
            pass


def wait_health(timeout_s: int = 240) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def qwen64_server_verified() -> dict[str, Any]:
    health = ""
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=3) as resp:
            health = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        health = f"ERR {type(exc).__name__}: {exc}"
    ps = run_quiet(["ps", "-eo", "pid=,comm=,args="], timeout=10)
    cmdlines: list[str] = []
    matching_pids: list[int] = []
    for line in ps.stdout.splitlines():
        parts = line.strip().split(maxsplit=2)
        if len(parts) < 3:
            continue
        pid_s, comm, args = parts
        if comm != "llama-server" or "Qwen3.6-35B-A3B-UD-IQ2_M.gguf" not in args:
            continue
        cmdlines.append(line.strip())
        if (
            "-c 65536" in args
            and "--cache-type-k q4_0" in args
            and "--cache-type-v q4_0" in args
            and ("--port 8080" in args or "--port=8080" in args)
            and "--host 127.0.0.1" in args
            and "--reasoning off" in args
        ):
            try:
                matching_pids.append(int(pid_s))
            except ValueError:
                pass
    owner_checks = {str(pid): pid_owns_listen_port(pid) for pid in matching_pids}
    checked = [value for value in owner_checks.values() if value is not None]
    owner_ok = any(checked) if checked else True
    return {
        "health": health,
        "cmdlines": cmdlines,
        "matching_pids": matching_pids,
        "port_owner_checks": owner_checks,
        "verified": health.startswith("{") and bool(matching_pids) and owner_ok,
    }


def pid_owns_listen_port(pid: int, port: int = 8080) -> bool | None:
    if not which("ss"):
        return None
    ss = run_quiet(["ss", "-ltnp"], timeout=10)
    if ss.returncode != 0:
        return None
    for line in ss.stdout.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        local = fields[3]
        if (
            local in {f"127.0.0.1:{port}", f"[::1]:{port}"}
            and f"pid={pid}," in line
        ):
            return True
    return False


def restore_resident(out_dir: Path) -> dict[str, Any]:
    restore_log = out_dir / "restore_64k_resident.log"
    started = now_iso()
    rc = -1
    error = ""
    with restore_log.open("a", encoding="utf-8") as log:
        log.write(f"[{started}] restoring 64K resident server\n")
        try:
            proc = subprocess.run(
                ["bash", "-lc", "cd /home/intel/QwenTalk && bash ./qwen36_64k_q4_resident.sh"],
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=420,
            )
            rc = proc.returncode
        except Exception as exc:
            error = repr(exc)
        log.write(f"[{now_iso()}] restore rc={rc} error={error}\n")
    row = {"type": "restore", "started_at": started, "finished_at": now_iso(), "rc": rc, "error": error, **qwen64_server_verified()}
    (out_dir / "restore_status.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def restore_failed(row: dict[str, Any]) -> bool:
    return row.get("rc") != 0 or not row.get("verified")


def clamp_timeout(case: BenchCase | ServerCase, deadline: float) -> BenchCase | ServerCase | None:
    remaining = int(deadline - time.time() - RESTORE_MARGIN_S)
    if remaining < case.min_left_s:
        return None
    planned_timeout = max(60, min(case.timeout_s, remaining))
    if isinstance(case, ServerCase):
        return replace(case, timeout_s=planned_timeout, deadline=time.time() + planned_timeout)
    return replace(case, timeout_s=planned_timeout)


def remaining_timeout(deadline: float, cap_s: int, reserve_s: int = 30) -> int:
    if deadline <= 0:
        return cap_s
    remaining = int(deadline - time.time() - reserve_s)
    if remaining <= 0:
        raise TimeoutError("case deadline reached")
    return max(1, min(cap_s, remaining))


def parse_llama_server_timings(log_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for line in log_text.splitlines():
        if "new prompt" in line:
            m = re.search(r"task\.n_tokens = (\d+)", line)
            if m:
                current = {"prompt_tokens_seen": int(m.group(1))}
        elif "prompt eval time" in line:
            m = re.search(r"=\s*([0-9.]+) ms /\s*(\d+) tokens .*?,\s*([0-9.]+) tokens per second", line)
            if m:
                current.update({
                    "prompt_eval_ms": float(m.group(1)),
                    "prompt_eval_tokens": int(m.group(2)),
                    "prompt_eval_tps": float(m.group(3)),
                })
        elif re.search(r"^\s*eval time", line):
            m = re.search(r"=\s*([0-9.]+) ms /\s*(\d+) tokens .*?,\s*([0-9.]+) tokens per second", line)
            if m:
                current.update({
                    "eval_ms": float(m.group(1)),
                    "eval_tokens": int(m.group(2)),
                    "eval_tps": float(m.group(3)),
                })
        elif "total time" in line:
            m = re.search(r"=\s*([0-9.]+) ms /\s*(\d+) tokens", line)
            if m:
                current.update({"total_ms": float(m.group(1)), "total_tokens": int(m.group(2))})
                rows.append(current)
                current = {}
    return rows


def chat(payload: dict[str, Any], timeout_s: int = 240) -> dict[str, Any]:
    req = urllib.request.Request(
        "http://127.0.0.1:8080/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    body["client_elapsed_s"] = round(time.time() - t0, 3)
    return body


def run_tokenizer_probe(out_dir: Path, summary: Path, parent_case: str, ctx: int) -> None:
    case_dir = out_dir / "tokenizer_probe"
    case_dir.mkdir(parents=True, exist_ok=True)
    text = (ROOT / "test_corpus" / "worm_ouroboros_gutenberg.txt").read_text(encoding="utf-8", errors="ignore")[:9000]
    req = urllib.request.Request(
        "http://127.0.0.1:8080/tokenize",
        data=json.dumps({"content": text}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - t0
    result = {
        "type": "tokenizer",
        "started_at": now_iso(),
        "parent_case": parent_case,
        "ctx": ctx,
        "chars": len(text),
        "tokens": len(body.get("tokens", [])),
        "elapsed_s": round(elapsed, 4),
        "tokens_per_s": round(len(body.get("tokens", [])) / max(elapsed, 1e-9), 1),
    }
    (case_dir / f"{parent_case}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    append_jsonl(summary, result)


def run_bench_case(case: BenchCase, out_dir: Path, summary: Path) -> int:
    case_dir = out_dir / "bench" / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(LLAMA_BENCH),
        "-m", str(MODEL),
        "-o", "jsonl",
        "-r", str(case.repetitions),
        "--delay", "1",
        *case.args,
    ]
    metadata = {"type": "bench", "name": case.name, "args": cmd, "started_at": now_iso()}
    (case_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    with (case_dir / "stdout.jsonl").open("w", encoding="utf-8") as stdout, (case_dir / "stderr.log").open("w", encoding="utf-8") as stderr:
        proc = subprocess.Popen(source_cmd(cmd), cwd=ROOT, stdout=stdout, stderr=stderr, text=True, start_new_session=True)
        time.sleep(1)
        monitors = start_monitors(case_dir, proc.pid)
        t0 = time.time()
        timed_out = False
        try:
            rc = proc.wait(timeout=case.timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            stop_process(proc, grace_s=10)
            rc = proc.returncode if proc.returncode is not None else -9
        elapsed = round(time.time() - t0, 3)
        stop_monitors(monitors)
    row = {**metadata, "finished_at": now_iso(), "rc": rc, "elapsed_s": elapsed, "timed_out": timed_out}
    (case_dir / "result.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    append_jsonl(summary, row)
    return rc


def start_server(case: ServerCase, case_dir: Path) -> subprocess.Popen[Any]:
    cmd = [
        str(LLAMA_SERVER),
        "-m", str(MODEL),
        "-ngl", "99",
        "-fa", str(case.flash_attn),
        "--cache-type-k", case.cache_k,
        "--cache-type-v", case.cache_v,
        "-c", str(case.ctx),
        "-np", "1",
        "--cache-ram", "0",
        "--no-cache-prompt",
        "--ctx-checkpoints", "0",
        "--port", "8080",
        "--host", "127.0.0.1",
        "--jinja",
        "--reasoning", "off",
    ]
    log = (case_dir / "server.log").open("w", encoding="utf-8")
    return subprocess.Popen(source_cmd(cmd), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True)


def run_server_case(case: ServerCase, out_dir: Path, summary: Path) -> int:
    case_dir = out_dir / "server" / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "type": "server",
        "name": case.name,
        "ctx": case.ctx,
        "prefill_chars": case.prefill_chars,
        "cache_k": case.cache_k,
        "cache_v": case.cache_v,
        "flash_attn": case.flash_attn,
        "started_at": now_iso(),
    }
    (case_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    stop_qwen_servers()
    proc = start_server(case, case_dir)
    t0 = time.time()
    case_deadline = case.deadline if case.deadline > 0 else (t0 + case.timeout_s)
    monitors: list[subprocess.Popen[Any]] = start_monitors(case_dir, proc.pid)
    healthy = False
    owner_check: bool | None = None
    probes: dict[str, Any] = {}
    rc = 0
    try:
        healthy = wait_health(timeout_s=remaining_timeout(case_deadline, 300, reserve_s=20))
        owner_check = pid_owns_listen_port(proc.pid)
        owns_port = owner_check is not False and proc.poll() is None
        if not healthy:
            rc = -1
        elif not owns_port:
            rc = -3
            probes["owner_error"] = {
                "server_pid": proc.pid,
                "proc_returncode": proc.poll(),
                "port_owner_check": owner_check,
                "message": "8080 health did not belong to the started llama-server process",
            }
        else:
            decode_payload = {
                "messages": [
                    {"role": "system", "content": "Benchmark responder. Do not stop early."},
                    {"role": "user", "content": "Write exactly 160 short English words about local agents and memory compression. No bullets."},
                ],
                "max_tokens": 128,
                "temperature": 0,
            }
            probes["decode_128"] = chat(decode_payload, timeout_s=remaining_timeout(case_deadline, 360, reserve_s=20))
            text = (ROOT / "test_corpus" / "worm_ouroboros_gutenberg.txt").read_text(encoding="utf-8", errors="ignore")[:case.prefill_chars]
            prefill_payload = {
                "messages": [
                    {"role": "system", "content": "Benchmark prompt ingestion. Answer with one word only."},
                    {"role": "user", "content": "Read the following source and answer OK only.\n\n" + text},
                ],
                "max_tokens": 1,
                "temperature": 0,
            }
            probes["prefill_2500"] = chat(prefill_payload, timeout_s=remaining_timeout(case_deadline, case.timeout_s, reserve_s=20))
            try:
                run_tokenizer_probe(out_dir, summary, case.name, case.ctx)
            except Exception as exc:
                probes["tokenizer_error"] = repr(exc)
    except Exception as exc:
        rc = -2
        probes["error"] = repr(exc)
    finally:
        stop_monitors(monitors)
        stop_process(proc, grace_s=20)
    log_text = (case_dir / "server.log").read_text(encoding="utf-8", errors="ignore") if (case_dir / "server.log").exists() else ""
    row = {
        **metadata,
        "finished_at": now_iso(),
        "healthy": healthy,
        "server_pid": proc.pid,
        "port_owner_check": owner_check,
        "rc": rc,
        "elapsed_s": round(time.time() - t0, 3),
        "probes": probes,
        "server_timings": parse_llama_server_timings(log_text),
    }
    (case_dir / "result.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    append_jsonl(summary, row)
    return rc


def build_bench_cases(mode: str) -> list[BenchCase]:
    base = ["-ngl", "99", "-fa", "1", "-ctk", "q4_0", "-ctv", "q4_0", "-b", "2048", "-ub", "512", "-t", "2"]
    if mode == "smoke":
        return [BenchCase("smoke_q4_pp128_tg8", ["-p", "128", "-n", "8", *base], timeout_s=600, repetitions=1)]
    if mode == "validation":
        return [
            BenchCase("q4_pp2048_tg64_t2", ["-p", "2048", "-n", "64", *base]),
            BenchCase("q4_no_kv_offload_pp2048_tg64", ["-p", "2048", "-n", "64", "-ngl", "99", "-fa", "1", "-ctk", "q4_0", "-ctv", "q4_0", "-nkvo", "1", "-b", "2048", "-ub", "512", "-t", "2"]),
            BenchCase("gpu_short_pp128_tg16_t8", ["-p", "128", "-n", "16", "-ngl", "99", "-fa", "1", "-ctk", "q4_0", "-ctv", "q4_0", "-b", "512", "-ub", "256", "-t", "8"], repetitions=1),
            BenchCase("cpu_only_short_pp128_tg16_t8", ["-p", "128", "-n", "16", "-ngl", "0", "-fa", "1", "-ctk", "q4_0", "-ctv", "q4_0", "-b", "512", "-ub", "256", "-t", "8"], timeout_s=1800, repetitions=1),
        ]
    return [
        BenchCase("q4_pp512_tg128_t2", ["-p", "512", "-n", "128", *base]),
        BenchCase("q4_pp2048_tg128_t2", ["-p", "2048", "-n", "128", *base]),
        BenchCase("q4_pp8192_tg32_t2", ["-p", "8192", "-n", "32", *base]),
        BenchCase("q4_pp2048_tg128_t4", ["-p", "2048", "-n", "128", *base[:-2], "-t", "4"]),
        BenchCase("q4_pp2048_tg128_t8", ["-p", "2048", "-n", "128", *base[:-2], "-t", "8"]),
        BenchCase("q4_pp4096_tg64_b1024", ["-p", "4096", "-n", "64", "-ngl", "99", "-fa", "1", "-ctk", "q4_0", "-ctv", "q4_0", "-b", "1024", "-ub", "512", "-t", "2"]),
        BenchCase("q4_pp4096_tg64_b4096", ["-p", "4096", "-n", "64", "-ngl", "99", "-fa", "1", "-ctk", "q4_0", "-ctv", "q4_0", "-b", "4096", "-ub", "512", "-t", "2"]),
        BenchCase("q4_pp2048_tg64_t2", ["-p", "2048", "-n", "64", *base]),
        BenchCase("q8kv_pp2048_tg128", ["-p", "2048", "-n", "128", "-ngl", "99", "-fa", "1", "-ctk", "q8_0", "-ctv", "q8_0", "-b", "2048", "-ub", "512", "-t", "2"]),
        BenchCase("f16kv_fa0_pp2048_tg128", ["-p", "2048", "-n", "128", "-ngl", "99", "-fa", "0", "-ctk", "f16", "-ctv", "f16", "-b", "2048", "-ub", "512", "-t", "2"]),
        BenchCase("f16kv_fa1_pp2048_tg128", ["-p", "2048", "-n", "128", "-ngl", "99", "-fa", "1", "-ctk", "f16", "-ctv", "f16", "-b", "2048", "-ub", "512", "-t", "2"]),
        BenchCase("q4_no_kv_offload_pp2048_tg64", ["-p", "2048", "-n", "64", "-ngl", "99", "-fa", "1", "-ctk", "q4_0", "-ctv", "q4_0", "-nkvo", "1", "-b", "2048", "-ub", "512", "-t", "2"]),
        BenchCase("q4_mmap0_pp2048_tg128", ["-p", "2048", "-n", "128", "-ngl", "99", "-fa", "1", "-ctk", "q4_0", "-ctv", "q4_0", "-b", "2048", "-ub", "512", "-t", "2", "-mmp", "0"], timeout_s=3600),
        BenchCase("gpu_short_pp128_tg16_t8", ["-p", "128", "-n", "16", "-ngl", "99", "-fa", "1", "-ctk", "q4_0", "-ctv", "q4_0", "-b", "512", "-ub", "256", "-t", "8"], repetitions=1),
        BenchCase("cpu_only_short_pp128_tg16_t8", ["-p", "128", "-n", "16", "-ngl", "0", "-fa", "1", "-ctk", "q4_0", "-ctv", "q4_0", "-b", "512", "-ub", "256", "-t", "8"], timeout_s=1800, repetitions=1),
    ]


def build_server_cases(mode: str) -> list[ServerCase]:
    if mode == "smoke":
        return [ServerCase("smoke_ctx8192_q4", ctx=8192, timeout_s=900)]
    if mode == "validation":
        return [
            ServerCase("ctx65536_q4_nearfull", ctx=65536, prefill_chars=210000, timeout_s=7200, min_left_s=5400),
            ServerCase("ctx128000_q4_long", ctx=128000, prefill_chars=330000, timeout_s=10800, min_left_s=7200),
        ]
    return [
        ServerCase("ctx8192_q4", ctx=8192, prefill_chars=24000),
        ServerCase("ctx32768_q4", ctx=32768, prefill_chars=90000, timeout_s=3600, min_left_s=1800),
        ServerCase("ctx65536_q4", ctx=65536, prefill_chars=180000, timeout_s=7200, min_left_s=3600),
        ServerCase("ctx128000_q4", ctx=128000, prefill_chars=270000, timeout_s=9000, min_left_s=5400),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run overnight Qwen3.6 profiling matrix")
    parser.add_argument("--hours", type=float, default=7.5)
    parser.add_argument("--mode", choices=["full", "smoke", "validation"], default="full")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--skip-bench", action="store_true")
    parser.add_argument("--skip-server", action="store_true")
    parser.add_argument("--no-restore", action="store_true")
    args = parser.parse_args()

    if not math.isfinite(args.hours) or args.hours <= 0:
        print(f"refusing invalid --hours value: {args.hours}", file=sys.stderr)
        return 3

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or (ROOT / "profile_runs" / f"qwen36_{args.mode}_{stamp}")
    deadline = time.time() + args.hours * 3600
    if args.out and out_dir.exists():
        if not out_dir.is_dir():
            print(f"refusing to use non-directory --out path: {out_dir}", file=sys.stderr)
            return 3
        if any(out_dir.iterdir()):
            print(f"refusing to reuse non-empty --out directory: {out_dir}", file=sys.stderr)
            return 3

    try:
        lock_file = acquire_lock()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 4

    summary: Path | None = None
    exit_code = 0
    server_control_started = False
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = out_dir / "summary.jsonl"
        manifest = {
            "started_at": now_iso(),
            "mode": args.mode,
            "hours": args.hours,
            "out_dir": str(out_dir),
            "model": str(MODEL),
            "tools": {
                "sudo_nopass": sudo_available(),
                "pidstat": bool(which("pidstat")),
                "vmstat": bool(which("vmstat")),
                "iostat": bool(which("iostat")),
                "turbostat": bool(which("turbostat")),
                "intel_gpu_top": bool(which("intel_gpu_top")),
            },
            "lock_path": str(LOCK_PATH),
            "restore_margin_s": RESTORE_MARGIN_S,
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        append_jsonl(summary, {"type": "manifest", **manifest})
        server_control_started = True
        stop_qwen_servers()
        if not args.skip_bench:
            for case in build_bench_cases(args.mode):
                planned = clamp_timeout(case, deadline)
                if planned is None:
                    append_jsonl(summary, {"type": "skip", "name": case.name, "reason": "deadline", "at": now_iso()})
                    break
                if run_bench_case(planned, out_dir, summary) != 0:
                    exit_code = max(exit_code, 1)

        if not args.skip_server:
            for case in build_server_cases(args.mode):
                planned = clamp_timeout(case, deadline)
                if planned is None:
                    append_jsonl(summary, {"type": "skip", "name": case.name, "reason": "deadline", "at": now_iso()})
                    break
                if run_server_case(planned, out_dir, summary) != 0:
                    exit_code = max(exit_code, 1)
    except Exception as exc:
        exit_code = 1
        if summary is not None:
            append_jsonl(summary, {"type": "fatal", "at": now_iso(), "error": repr(exc)})
        else:
            print(repr(exc), file=sys.stderr)
    finally:
        if server_control_started:
            stop_qwen_servers()
            if not args.no_restore:
                restore_row = restore_resident(out_dir)
                if summary is not None:
                    append_jsonl(summary, restore_row)
                if restore_failed(restore_row):
                    exit_code = max(exit_code, 2)
        if summary is not None:
            append_jsonl(summary, {"type": "done", "finished_at": now_iso(), "exit_code": exit_code})
        if lock_file is not None:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
            except Exception:
                pass

    print(str(out_dir))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
