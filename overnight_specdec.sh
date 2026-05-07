#!/usr/bin/env bash
# overnight_specdec.sh — Round 3: 推测解码 (Speculative Decoding) 自动探索
#
# 目标: 在不训练任何 head 的前提下，验证经典 spec decoding 在 SYCL Xe-LPG
# 后端的可行性，覆盖 Qwen3.6 自推测、Qwen3.6×Qwen3-4B 跨版本、Gemma 4 同族对。
#
# 设计: target 全跑 GPU (-ngl 99)，draft 留 CPU (-ngld 0)，避免 22GB UMA OOM。
# 每个测试 timeout 180s，vocab 不兼容会快速失败而非挂死。
set -E -o pipefail

# 加载 Intel oneAPI 运行时 (libsvml.so 等)
# systemd-run --user 不继承交互 shell 环境，必须显式 source
# shellcheck disable=SC1091
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1 || {
    echo "[FATAL] oneAPI setvars.sh 加载失败" >&2; exit 1;
}

ROOT=~/llama.cpp/build_sycl/bin
M=~/models
TS=$(date +%Y%m%d_%H%M)
OUT=~/overnight_results/specdec_$TS
mkdir -p "$OUT"
LOG="$OUT/main.log"

# Tee 整个脚本输出到 main.log
exec > >(tee -a "$LOG") 2>&1

echo "=== Spec Decoding 探索 开始 $(date -Is) ==="
echo "结果目录: $OUT"
echo "板卡内存: $(free -h | awk '/^Mem:/{print $2 " 总, " $7 " 可用"}')"
echo

# 短 prompt + 中等生成长度，每测试 ~60s 内完成
PROMPT="请用 100 个字解释什么是大语言模型推理加速技术，包括其核心原理和典型方法。"
NGEN=128
COMMON_TARGET="-ngl 99 -c 2048 -ctk q8_0 -ctv f16 --temp 0 --no-warmup"

run_one() {
    local name="$1"; shift
    echo
    echo "=========================================================="
    echo "[$(date +%T)] >>> $name"
    echo "=========================================================="
    set +e
    timeout 240 "$@" > "$OUT/${name}.log" 2>&1
    local ec=$?
    set -e
    # 只打印关键尾部供 main.log 概览
    tail -30 "$OUT/${name}.log"
    echo "[$(date +%T)] <<< $name (exit=$ec)"
}

# ----------------------------------------------------------------------
# Phase 0: Baselines (无 spec decoding，作为参照)
# ----------------------------------------------------------------------
run_one "base_qwen36_iq4xs" "$ROOT/llama-cli" \
    -m "$M/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf" \
    $COMMON_TARGET -p "$PROMPT" -n $NGEN -no-cnv

run_one "base_qwen36_iq2m" "$ROOT/llama-cli" \
    -m "$M/Qwen3.6-35B-A3B-UD-IQ2_M.gguf" \
    $COMMON_TARGET -p "$PROMPT" -n $NGEN -no-cnv

run_one "base_qwen3_4b" "$ROOT/llama-cli" \
    -m "$M/Qwen3-4B-Instruct-2507-Q4_K_M.gguf" \
    $COMMON_TARGET -p "$PROMPT" -n $NGEN -no-cnv

run_one "base_gemma4_26b_iq2m" "$ROOT/llama-cli" \
    -m "$M/gemma-4-26B-A4B-it-UD-IQ2_M.gguf" \
    $COMMON_TARGET -p "$PROMPT" -n $NGEN -no-cnv

run_one "base_gemma4_e4b" "$ROOT/llama-cli" \
    -m "$M/gemma-4-E4B-it-Q4_K_M.gguf" \
    $COMMON_TARGET -p "$PROMPT" -n $NGEN -no-cnv

run_one "base_gemma4_e2b" "$ROOT/llama-cli" \
    -m "$M/gemma-4-E2B-it-Q4_K_M.gguf" \
    $COMMON_TARGET -p "$PROMPT" -n $NGEN -no-cnv

# ----------------------------------------------------------------------
# Phase 1: Qwen3.6 自推测 (target=IQ4_XS, draft=IQ2_M, 同 vocab 保证)
# ----------------------------------------------------------------------
for DM in 4 8 16; do
    run_one "spec_qwen36_self_dmax${DM}" "$ROOT/llama-speculative" \
        -m  "$M/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf" \
        -md "$M/Qwen3.6-35B-A3B-UD-IQ2_M.gguf" \
        -ngl 99 -ngld 0 -c 2048 -ctk q8_0 -ctv f16 \
        --draft-max $DM --draft-min 1 --temp 0 --no-warmup \
        -p "$PROMPT" -n $NGEN
done

# ----------------------------------------------------------------------
# Phase 2: Qwen3.6 × Qwen3-4B 跨版本 (vocab 可能不兼容，快速失败 OK)
# ----------------------------------------------------------------------
run_one "spec_qwen36_x_qwen3_4b" "$ROOT/llama-speculative" \
    -m  "$M/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf" \
    -md "$M/Qwen3-4B-Instruct-2507-Q4_K_M.gguf" \
    -ngl 99 -ngld 0 -c 2048 -ctk q8_0 -ctv f16 \
    --draft-max 8 --draft-min 1 --temp 0 --no-warmup \
    -p "$PROMPT" -n $NGEN

# ----------------------------------------------------------------------
# Phase 3: Gemma 4 同族 (26B target, E2B/E4B draft)
# ----------------------------------------------------------------------
run_one "spec_gemma4_26biq2m_x_e2b" "$ROOT/llama-speculative" \
    -m  "$M/gemma-4-26B-A4B-it-UD-IQ2_M.gguf" \
    -md "$M/gemma-4-E2B-it-Q4_K_M.gguf" \
    -ngl 99 -ngld 0 -c 2048 -ctk q8_0 -ctv f16 \
    --draft-max 8 --draft-min 1 --temp 0 --no-warmup \
    -p "$PROMPT" -n $NGEN

run_one "spec_gemma4_26biq2m_x_e4b" "$ROOT/llama-speculative" \
    -m  "$M/gemma-4-26B-A4B-it-UD-IQ2_M.gguf" \
    -md "$M/gemma-4-E4B-it-Q4_K_M.gguf" \
    -ngl 99 -ngld 0 -c 2048 -ctk q8_0 -ctv f16 \
    --draft-max 8 --draft-min 1 --temp 0 --no-warmup \
    -p "$PROMPT" -n $NGEN

# Q4_K_M 高质量 target (16G，单 GPU 装下) + E2B draft
run_one "spec_gemma4_26bq4km_x_e2b" "$ROOT/llama-speculative" \
    -m  "$M/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf" \
    -md "$M/gemma-4-E2B-it-Q4_K_M.gguf" \
    -ngl 99 -ngld 0 -c 2048 -ctk q8_0 -ctv f16 \
    --draft-max 8 --draft-min 1 --temp 0 --no-warmup \
    -p "$PROMPT" -n $NGEN

# Phase 3b: dmax 扫描 (基于 Phase 3 最佳对，先选 26b_iq2m × e2b 作为代表)
for DM in 4 16; do
    run_one "spec_gemma4_26biq2m_x_e2b_dmax${DM}" "$ROOT/llama-speculative" \
        -m  "$M/gemma-4-26B-A4B-it-UD-IQ2_M.gguf" \
        -md "$M/gemma-4-E2B-it-Q4_K_M.gguf" \
        -ngl 99 -ngld 0 -c 2048 -ctk q8_0 -ctv f16 \
        --draft-max $DM --draft-min 1 --temp 0 --no-warmup \
        -p "$PROMPT" -n $NGEN
done

# ----------------------------------------------------------------------
# Phase 4: 汇总 t/s 与 accept rate
# ----------------------------------------------------------------------
echo
echo "=== 汇总 ==="
{
    echo "# Speculative Decoding 探索结果 ($TS)"
    echo
    echo "板卡: DK-2500 / Arrow Lake-U / Xe-LPG / 24GB DDR5-5600"
    echo "llama.cpp: SYCL backend"
    echo "Prompt: $PROMPT"
    echo "Generation length: $NGEN tokens"
    echo
    echo "## 全部测试结果"
    echo
    for f in "$OUT"/*.log; do
        name=$(basename "$f" .log)
        echo "### $name"
        echo '```'
        # 抓性能与接受率关键行
        grep -E "tokens per second|accept|n_drafted|n_accept|eval time|prompt eval time|^encode|^decode|llama_perf_(context|sampler)_print" "$f" 2>/dev/null | head -20
        # 错误行（vocab mismatch 等）
        grep -iE "error|failed|mismatch|abort" "$f" 2>/dev/null | head -5
        echo '```'
        echo
    done

    echo "## 简表 (从日志提取)"
    echo
    echo "| 测试 | tg t/s | accept% | 备注 |"
    echo "|------|--------|---------|------|"
    for f in "$OUT"/*.log; do
        name=$(basename "$f" .log)
        # 抓最后一段的 eval tokens per second
        tg=$(grep -oE "[0-9]+\.[0-9]+ tokens per second" "$f" 2>/dev/null | tail -1 | grep -oE "^[0-9.]+")
        # 抓 accept rate (llama-speculative 输出 "accept = N/M (P%)")
        accept=$(grep -oE "accept[t ]*= *[0-9]+/[0-9]+ *\([0-9.]+%\)" "$f" 2>/dev/null | tail -1 | grep -oE "[0-9.]+%")
        err=$(grep -ic "error\|failed\|mismatch" "$f" 2>/dev/null | head -1)
        note=""
        [ "${err:-0}" -gt 0 ] && note="有 error"
        echo "| $name | ${tg:-?} | ${accept:-N/A} | $note |"
    done
} > "$OUT/SUMMARY.md"

echo
echo "=== 全部完成 $(date -Is) ==="
echo "结果: $OUT/SUMMARY.md"
echo "完整日志: $OUT/*.log"
