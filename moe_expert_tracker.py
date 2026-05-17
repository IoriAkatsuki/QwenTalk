#!/usr/bin/env python3
"""
moe_expert_tracker.py — Qwen3.6 MoE expert 频次统计 + 卸载配置生成

两种模式：
  1) static: 仅基于 gguf_dump 输出生成 `--override-tensor` 配置（无需 patch llama.cpp）
  2) parse:  解析 llama.cpp verbose log (需 build 时 patch build_moe_ffn 打印 expert ids)

输出:
  - per-layer / per-expert frequency
  - 建议的 -ot 正则（基于 layer-level cold ranking, 不到 single-expert 粒度）

设计：纯旁路工具，不连 llmsrv，仅读 log/gguf。

注意：GGUF 把 256 experts 打包成单 tensor，--override-tensor 只能整层粒度。
要做到单 expert offload 需要改 llama.cpp build_moe_ffn (本工具不做)。

用法:
  python3 moe_expert_tracker.py static <gguf_path>
  python3 moe_expert_tracker.py parse <verbose_log>
  python3 moe_expert_tracker.py gen-ot --cold-layers 30-40
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


GGUF_DUMP_SCRIPT = "/home/intel/llama_mtp/gguf-py/gguf/scripts/gguf_dump.py"


# 解析 gguf_dump 行：" 9: 268435456 |  2048,  512,  256, 1 | Q4_K | blk.0.ffn_gate_exps.weight"
TENSOR_LINE_RE = re.compile(
    r"^\s*\d+:\s+(\d+)\s+\|.*\|\s+(\S+)\s+\|\s+(blk\.\d+\.ffn_.*_exps\.weight)\s*$"
)

# log 中（patch 后）期望格式：例 "MOE: layer=12 experts=[5,17,42,...]"
LOG_MOE_RE = re.compile(r"MOE:\s*layer=(\d+)\s+experts=\[([0-9,\s]+)\]")


def cmd_static(gguf_path: str) -> int:
    """从 gguf_dump 输出列举所有 expert tensor 和大小估算"""
    if not Path(gguf_path).exists():
        print(f"[ERR] 文件不存在: {gguf_path}", file=sys.stderr)
        return 2
    if not Path(GGUF_DUMP_SCRIPT).exists():
        print(f"[ERR] gguf_dump 脚本缺失: {GGUF_DUMP_SCRIPT}", file=sys.stderr)
        print("      尝试: python3 -m gguf.scripts.gguf_dump", file=sys.stderr)
        return 2

    proc = subprocess.run(
        ["python3", GGUF_DUMP_SCRIPT, gguf_path],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        print("[ERR] gguf_dump 失败:", proc.stderr[:500], file=sys.stderr)
        return proc.returncode

    by_layer: dict[int, list[tuple[str, int, str]]] = {}
    total_bytes = 0
    for line in proc.stdout.splitlines():
        m = TENSOR_LINE_RE.match(line)
        if not m:
            continue
        elems, dtype, name = m.groups()
        elems = int(elems)
        # Q4_K_S 实际字节估算：每元素 ~0.55 字节（K-quant 平均）
        approx_bytes = int(elems * 0.55)
        total_bytes += approx_bytes
        layer_m = re.match(r"blk\.(\d+)\.", name)
        if layer_m:
            layer = int(layer_m.group(1))
            by_layer.setdefault(layer, []).append((name, approx_bytes, dtype))

    print(f"=== Expert tensors 概览 ({gguf_path}) ===")
    print(f"总 expert 字节估算: {total_bytes / 1024**3:.2f} GB")
    print(f"涉及 layers: {len(by_layer)}")
    print()
    print(f"{'layer':>5} | {'tensors':>7} | {'size_MB':>9}")
    print("-" * 30)
    for layer in sorted(by_layer):
        tensors = by_layer[layer]
        size_mb = sum(b for _, b, _ in tensors) / 1024**2
        print(f"{layer:>5} | {len(tensors):>7} | {size_mb:>9.1f}")
    return 0


def cmd_parse(log_path: str) -> int:
    """解析 verbose log 中 patch 打印的 expert ids"""
    p = Path(log_path)
    if not p.exists():
        print(f"[ERR] log 不存在: {log_path}", file=sys.stderr)
        return 2

    per_layer_expert: dict[int, Counter] = {}
    total_lines = 0
    matched = 0
    for line in p.read_text(errors="ignore").splitlines():
        total_lines += 1
        m = LOG_MOE_RE.search(line)
        if not m:
            continue
        matched += 1
        layer = int(m.group(1))
        ids = [int(x.strip()) for x in m.group(2).split(",") if x.strip()]
        c = per_layer_expert.setdefault(layer, Counter())
        for eid in ids:
            c[eid] += 1

    if matched == 0:
        print(f"[WARN] log 中没有 'MOE: layer=N experts=[...]' 行 (扫描 {total_lines} 行)")
        print("       需 patch llama.cpp build_moe_ffn 添加打印；当前不可用")
        return 1

    print(f"匹配 MOE 行: {matched} / {total_lines}")
    for layer in sorted(per_layer_expert):
        c = per_layer_expert[layer]
        top5 = c.most_common(5)
        cold = sum(1 for v in c.values() if v <= 1)
        print(f"layer {layer:2d}: top5={top5}  cold(<=1)={cold}/{len(c)}")
    return 0


def cmd_gen_ot(cold_layers: str) -> int:
    """生成 --override-tensor 正则；cold_layers 形如 '30-40' 或 '5,7,30-40'"""
    parts = set()
    for chunk in cold_layers.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            for i in range(int(a), int(b) + 1):
                parts.add(i)
        else:
            parts.add(int(chunk))
    if not parts:
        print("[ERR] 没有有效 layer 编号", file=sys.stderr)
        return 2

    nums = sorted(parts)
    # 拼成正则，简化版逐一列举：blk\.(N1|N2|...)\.ffn_.*_exps\.weight=CPU
    inner = "|".join(str(n) for n in nums)
    regex = rf"blk\.({inner})\.ffn_.*_exps\.weight=CPU"
    print("# 卸载到 CPU 的层数:", nums)
    print(f"# 单层 expert 占用 ~201 MB → 总卸载 ~{len(nums) * 201 / 1024:.2f} GB")
    print(f"--override-tensor '{regex}'")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen3.6 MoE expert tracker / -ot 生成")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sp_static = sub.add_parser("static", help="从 GGUF 列出 expert tensors")
    sp_static.add_argument("gguf", help="GGUF 路径")
    sp_parse = sub.add_parser("parse", help="解析 patched llama log")
    sp_parse.add_argument("log", help="log 路径")
    sp_gen = sub.add_parser("gen-ot", help="生成 -ot 正则")
    sp_gen.add_argument("--cold-layers", required=True, help="如 '30-40' 或 '5,7,30-40'")

    args = parser.parse_args()
    if args.cmd == "static":
        return cmd_static(args.gguf)
    if args.cmd == "parse":
        return cmd_parse(args.log)
    if args.cmd == "gen-ot":
        return cmd_gen_ot(args.cold_layers)
    return 1


if __name__ == "__main__":
    sys.exit(main())
