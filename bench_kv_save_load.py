#!/usr/bin/env python3
"""KV cache 持久化层带宽基准 —— dummy bytes only，不调 llama-server。

跑法：
    python3 bench_kv_save_load.py                  # 默认 10/100/1000/5000 MB
    python3 bench_kv_save_load.py --sizes 100,500  # 自定义
    python3 bench_kv_save_load.py --base /home/intel/kv_sessions  # 板卡 PC801 路径

输出：markdown 表格 + 可选 iostat（如果系统有 sysstat）。
"""
from __future__ import annotations

import argparse
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from kv_persistence import KVStore  # noqa: E402

DEFAULT_SIZES_MB = [10, 100, 1000, 5000]
DEFAULT_REPEATS = 3


def _drop_caches() -> bool:
    """尽力清 page cache；非 root 直接 skip。"""
    try:
        with open("/proc/sys/vm/drop_caches", "w") as fh:
            fh.write("3")
        return True
    except PermissionError:
        return False


def _bench_one(store: KVStore, mb: int, repeats: int) -> dict:
    """单 size 做 repeats 次 write+read，取中位数。"""
    n_bytes = mb * (1 << 20)
    writes_mbs: list[float] = []
    reads_mbs: list[float] = []
    for i in range(repeats):
        sid = f"bench-{mb}MB-r{i}"
        t0 = time.perf_counter()
        store.save_session(sid, _dummy_bytes=n_bytes)
        dt_w = max(time.perf_counter() - t0, 1e-6)
        writes_mbs.append(mb / dt_w)

        _drop_caches()
        kv_path = store._kv_path(sid)  # noqa: SLF001 — bench friend access
        t0 = time.perf_counter()
        with kv_path.open("rb") as fh:
            while fh.read(4 << 20):
                pass
        dt_r = max(time.perf_counter() - t0, 1e-6)
        reads_mbs.append(mb / dt_r)

        kv_path.unlink(missing_ok=True)
        store._meta_path(sid).unlink(missing_ok=True)  # noqa: SLF001

    return {
        "size_mb": mb,
        "write_mbs_median": statistics.median(writes_mbs),
        "read_mbs_median": statistics.median(reads_mbs),
        "write_mbs_min": min(writes_mbs),
        "read_mbs_min": min(reads_mbs),
    }


def _iostat_snapshot(device_hint: str = "") -> str:
    """跑一次 iostat 1 2，截最后一段。无 iostat 返回空串。"""
    if shutil.which("iostat") is None:
        return ""
    try:
        out = subprocess.run(
            ["iostat", "-x", "-d", "1", "2"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        return ""
    if device_hint:
        return "\n".join(line for line in out.splitlines() if device_hint in line)
    return out


def _print_markdown(results: list[dict], iostat: str, base_dir: Path) -> None:
    print("\n# KV Persistence Bench Results\n")
    print(f"- base_dir: `{base_dir}`")
    print(f"- platform: `{os.uname().sysname} {os.uname().release}`")
    print(f"- drop_caches: {'yes' if _can_drop() else 'no (non-root)'}")
    print()
    print("| size (MB) | write (MB/s) | read (MB/s) | write_min | read_min |")
    print("|---:|---:|---:|---:|---:|")
    for r in results:
        print(
            f"| {r['size_mb']} "
            f"| {r['write_mbs_median']:.1f} "
            f"| {r['read_mbs_median']:.1f} "
            f"| {r['write_mbs_min']:.1f} "
            f"| {r['read_mbs_min']:.1f} |"
        )
    if iostat:
        print("\n## iostat -x -d 1 2 snapshot\n```")
        print(iostat.strip())
        print("```")


def _can_drop() -> bool:
    try:
        with open("/proc/sys/vm/drop_caches", "w") as fh:
            fh.write("0")
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES_MB),
                    help="comma-separated MB sizes")
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--base", default=None, help="base dir; default = mkdtemp")
    ap.add_argument("--iostat-device", default="",
                    help="filter iostat lines by device name, e.g. sda or nvme1n1")
    args = ap.parse_args()

    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    base = Path(args.base) if args.base else Path(tempfile.mkdtemp(prefix="kvbench_"))
    base.mkdir(parents=True, exist_ok=True)

    store = KVStore(base_dir=base, llm_url=None)
    results = [_bench_one(store, mb, args.repeats) for mb in sizes]
    iostat = _iostat_snapshot(args.iostat_device)
    _print_markdown(results, iostat, base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
