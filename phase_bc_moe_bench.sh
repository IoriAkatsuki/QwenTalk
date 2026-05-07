#!/usr/bin/env bash
# Phase B+C: Gemma 4 26B-A4B MoE 基准 + flash-moe 风格 expert offload
# 先 gguf_dump 确认 tensor 命名，再用实测正则做 --override-tensor
set -u

MODEL=${1:-/home/intel/models/gemma-4-26B-A4B-it-Q4_K_M.gguf}
LLAMA_BIN=/home/intel/llama.cpp/build/bin
LLAMA_SRC=/home/intel/llama.cpp
LOGDIR=/tmp/phase_bc
mkdir -p "$LOGDIR"

if [ ! -f "$MODEL" ]; then
    echo "[FAIL] 模型文件不存在: $MODEL"
    exit 1
fi

echo "========== Phase B+C: Gemma 4 26B-A4B MoE 基准 =========="
echo ""

echo "[1/6] 模型大小："
ls -lh "$MODEL"

echo ""
echo "[2/6] 分析 MoE tensor 命名（确定 --override-tensor 正则）："
/home/intel/miniforge3/envs/openvino/bin/python \
    $LLAMA_SRC/gguf-py/gguf/scripts/gguf_dump.py "$MODEL" --no-tensors 2>/dev/null | head -30

echo ""
echo "[3/6] 列出 expert 类 tensor 名："
/home/intel/miniforge3/envs/openvino/bin/python \
    $LLAMA_SRC/gguf-py/gguf/scripts/gguf_dump.py "$MODEL" 2>/dev/null | \
    grep -iE "exps|expert|ffn" | head -10

echo ""
echo "[4/6] 裸跑（纯 CPU + mmap，flash-moe trust-OS 基线）："
echo "  先刷 page cache，测冷启动"
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

iostat -xm 1 sda > $LOGDIR/iostat_cpu.log 2>&1 &
I_PID=$!
time $LLAMA_BIN/llama-cli -m "$MODEL" -ngl 0 \
    --mmap -p "用中文介绍英特尔 Core Ultra 处理器的三引擎架构。" \
    -n 80 --temp 0.7 -no-cnv 2>&1 | tee $LOGDIR/cpu_run.log | tail -15
kill $I_PID 2>/dev/null

echo ""
echo "[5/6] GPU + expert 卸载（flash-moe 精髓：非专家 GPU，expert FFN CPU/SSD）："
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

iostat -xm 1 sda > $LOGDIR/iostat_moe.log 2>&1 &
I_PID=$!
time $LLAMA_BIN/llama-cli -m "$MODEL" -ngl 99 \
    --override-tensor '\.ffn_.*_exps\.=CPU' \
    --mmap -p "用中文介绍英特尔 Core Ultra 处理器的三引擎架构。" \
    -n 80 --temp 0.7 -no-cnv 2>&1 | tee $LOGDIR/moe_run.log | tail -15
kill $I_PID 2>/dev/null

echo ""
echo "[6/6] 最终基准（50 warmup + 200 timed, llama-bench）："
$LLAMA_BIN/llama-bench -m "$MODEL" -ngl 99 \
    -ot '\.ffn_.*_exps\.=CPU' \
    -p 128 -n 200 -r 2 2>&1 | tail -10

echo ""
echo "========== iostat 分析 =========="
for f in iostat_cpu iostat_moe; do
    echo "--- $f ---"
    awk '/^sda / {s+=$(NF); if($(NF)>mx)mx=$(NF); if($3>rmx)rmx=$3; rs+=$3; n++}
         END {if(n>0) printf "  %%util 均=%.1f 峰=%.1f | rkB/s 均=%.0f 峰=%.0f\n",
              s/n, mx, rs/n, rmx}' $LOGDIR/$f.log
done
