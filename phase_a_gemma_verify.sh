#!/usr/bin/env bash
# Phase A 后半段: Gemma 4 E4B 架构兼容性验证 + llama.cpp Vulkan 基准
# 前提: /home/intel/models/gemma-4-E4B-it-Q4_K_M.gguf 已存在
set -u

MODEL=${1:-/home/intel/models/gemma-4-E4B-it-Q4_K_M.gguf}
LLAMA_BIN=/home/intel/llama.cpp/build/bin

if [ ! -f "$MODEL" ]; then
    echo "[FAIL] 模型文件不存在: $MODEL"
    exit 1
fi

echo "========== Phase A: Gemma 4 E4B 兼容性验证 =========="
echo ""

echo "[1/4] llama.cpp 版本："
$LLAMA_BIN/llama-cli --version 2>&1 | head -2

echo ""
echo "[2/4] GGUF 文件元信息（架构识别）："
$LLAMA_BIN/llama-cli -m "$MODEL" --no-warmup -p "test" -n 1 -no-cnv 2>&1 | \
    grep -E "arch|tokenizer|embedding|n_layer|ffn|expert|n_expert" | head -20

echo ""
echo "[3/4] Vulkan 设备确认 + 加载测试："
echo "(-ngl 99 = 所有层上 GPU)"
timeout 60 $LLAMA_BIN/llama-cli -m "$MODEL" -ngl 99 \
    -p "你好，用中文自我介绍。" -n 30 --no-warmup \
    --temp 0.7 -no-cnv 2>&1 | tail -25

echo ""
echo "[4/4] 基准测试（prompt=128 / generate=64）："
$LLAMA_BIN/llama-bench -m "$MODEL" -ngl 99 -p 128 -n 64 -r 2 2>&1 | tail -10

echo ""
echo "========== 判定 =========="
echo "若上面所有步骤 tok/s 有数字且输出中文连贯 → 架构兼容，可下 26B-A4B"
echo "若报 'unknown architecture' / 'unsupported tokenizer' → 需要升级 llama.cpp 或改选模型"
