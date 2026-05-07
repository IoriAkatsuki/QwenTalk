#!/usr/bin/env bash
# overnight_specdec_v2.sh — Round 3.5: 只重跑 spec decoding 阶段
# v1 修复: 移除 --no-warmup (spec 二进制不识别)，去掉 -no-cnv (已弃用)
set -E -o pipefail

# shellcheck disable=SC1091
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1 || {
    echo "[FATAL] oneAPI setvars.sh 失败" >&2; exit 1;
}

ROOT=~/llama.cpp/build_sycl/bin
M=~/models
TS=$(date +%Y%m%d_%H%M)
OUT=~/overnight_results/specdec_v2_$TS
mkdir -p "$OUT"
LOG="$OUT/main.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Spec Decoding v2 (仅 spec 阶段) 开始 $(date -Is) ==="
echo "结果: $OUT"
echo

PROMPT="请用 100 个字解释什么是大语言模型推理加速技术，包括其核心原理和典型方法。"
NGEN=128

run_one() {
    local name="$1"; shift
    echo
    echo "=========================================================="
    echo "[$(date +%T)] >>> $name"
    echo "=========================================================="
    set +e
    # < /dev/null 防止任何二进制误进交互模式
    timeout 240 "$@" < /dev/null > "$OUT/${name}.log" 2>&1
    local ec=$?
    set -e
    # 抓关键性能行打印到 main
    grep -E "tokens per second|t/s|accept|n_drafted|n_accept|encoded|decoded|draft.*accepted" "$OUT/${name}.log" 2>/dev/null | tail -10 || true
    echo "[$(date +%T)] <<< $name (exit=$ec, log size=$(stat -c%s "$OUT/${name}.log") bytes)"
}

# Phase 1: Qwen3.6 自推测 (target=IQ4_XS, draft=IQ2_M)
for DM in 4 8 16; do
    run_one "spec_qwen36_self_dmax${DM}" "$ROOT/llama-speculative" \
        -m  "$M/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf" \
        -md "$M/Qwen3.6-35B-A3B-UD-IQ2_M.gguf" \
        -ngl 99 -ngld 0 -c 2048 -ctk q8_0 -ctv f16 \
        --draft-max "$DM" --draft-min 1 --temp 0 \
        -p "$PROMPT" -n $NGEN
done

# Phase 2: Qwen3.6 × Qwen3-4B 跨版本 (vocab 兼容性测试)
run_one "spec_qwen36_x_qwen3_4b" "$ROOT/llama-speculative" \
    -m  "$M/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf" \
    -md "$M/Qwen3-4B-Instruct-2507-Q4_K_M.gguf" \
    -ngl 99 -ngld 0 -c 2048 -ctk q8_0 -ctv f16 \
    --draft-max 8 --draft-min 1 --temp 0 \
    -p "$PROMPT" -n $NGEN

# Phase 3: Gemma 4 同族
run_one "spec_gemma4_26biq2m_x_e2b" "$ROOT/llama-speculative" \
    -m  "$M/gemma-4-26B-A4B-it-UD-IQ2_M.gguf" \
    -md "$M/gemma-4-E2B-it-Q4_K_M.gguf" \
    -ngl 99 -ngld 0 -c 2048 -ctk q8_0 -ctv f16 \
    --draft-max 8 --draft-min 1 --temp 0 \
    -p "$PROMPT" -n $NGEN

run_one "spec_gemma4_26biq2m_x_e4b" "$ROOT/llama-speculative" \
    -m  "$M/gemma-4-26B-A4B-it-UD-IQ2_M.gguf" \
    -md "$M/gemma-4-E4B-it-Q4_K_M.gguf" \
    -ngl 99 -ngld 0 -c 2048 -ctk q8_0 -ctv f16 \
    --draft-max 8 --draft-min 1 --temp 0 \
    -p "$PROMPT" -n $NGEN

run_one "spec_gemma4_26bq4km_x_e2b" "$ROOT/llama-speculative" \
    -m  "$M/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf" \
    -md "$M/gemma-4-E2B-it-Q4_K_M.gguf" \
    -ngl 99 -ngld 0 -c 2048 -ctk q8_0 -ctv f16 \
    --draft-max 8 --draft-min 1 --temp 0 \
    -p "$PROMPT" -n $NGEN

# Phase 3b: dmax 扫描
for DM in 4 16; do
    run_one "spec_gemma4_26biq2m_x_e2b_dmax${DM}" "$ROOT/llama-speculative" \
        -m  "$M/gemma-4-26B-A4B-it-UD-IQ2_M.gguf" \
        -md "$M/gemma-4-E2B-it-Q4_K_M.gguf" \
        -ngl 99 -ngld 0 -c 2048 -ctk q8_0 -ctv f16 \
        --draft-max "$DM" --draft-min 1 --temp 0 \
        -p "$PROMPT" -n $NGEN
done

# 汇总
echo
echo "=== 汇总 $(date -Is) ==="
{
    echo "# Spec Decoding v2 结果 ($TS)"
    echo
    echo "板卡: DK-2500 / Arrow Lake-U / Xe-LPG / 24GB DDR5-5600"
    echo "Prompt: $PROMPT"
    echo "Gen: $NGEN tokens / temp=0 / draft on CPU (-ngld 0)"
    echo
    echo "## Baseline (Round 3 v1 已测)"
    echo
    echo "| 模型 | Prompt t/s | Gen t/s |"
    echo "|------|-----------|---------|"
    echo "| Qwen3.6 35B IQ4_XS | 9.8 | **5.7** |"
    echo "| Qwen3.6 35B IQ2_M | 11.7 | 4.3 |"
    echo "| Qwen3-4B Q4_K_M | 23.4 | 7.6 |"
    echo "| Gemma 4 26B IQ2_M | 11.4 | 3.1 |"
    echo "| Gemma 4 26B Q4_K_M | (未测，需补) | - |"
    echo "| Gemma 4 E4B Q4 | 21.0 | 7.2 |"
    echo "| Gemma 4 E2B Q4 | 35.8 | 12.6 |"
    echo
    echo "## Spec Decoding 结果"
    echo
    echo "| 测试 | 退出 | 关键指标 |"
    echo "|------|------|----------|"
    for f in "$OUT"/spec_*.log; do
        name=$(basename "$f" .log)
        size=$(stat -c%s "$f")
        # 抓 t/s
        ts=$(grep -oE "[0-9]+\.[0-9]+ tokens per second|[0-9]+\.[0-9]+ t/s" "$f" 2>/dev/null | tail -3 | tr '\n' ',' | sed 's/,$//')
        # 抓 accept
        acc=$(grep -iE "accept|n_drafted|n_accept" "$f" 2>/dev/null | tail -3 | tr '\n' '|' | sed 's/|$//')
        # 错误
        err=$(grep -iE "error|invalid argument|abort|failed|mismatch" "$f" 2>/dev/null | head -2 | tr '\n' '|')
        if [ -n "$err" ]; then
            echo "| $name | ❌ | err: $err |"
        else
            echo "| $name | ✅ | $ts ; $acc |"
        fi
    done
    echo
    echo "## 详细日志关键段"
    echo
    for f in "$OUT"/spec_*.log; do
        name=$(basename "$f" .log)
        echo "### $name (size=$(stat -c%s "$f"))"
        echo '```'
        # 提取性能与接受率段，避免完整日志
        grep -E "tokens per second|t/s|accept|n_drafted|n_accept|encoded|decoded|draft|llama_perf|model.*size|^load_tensors" "$f" 2>/dev/null | head -25
        grep -iE "error|invalid|abort|failed|mismatch" "$f" 2>/dev/null | head -3
        echo '```'
        echo
    done
} > "$OUT/SUMMARY.md"

echo
echo "=== 全部完成 $(date -Is) ==="
echo "结果: $OUT/SUMMARY.md"
