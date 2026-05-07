#!/usr/bin/env bash
# Phase A0: SATA 瓶颈证伪
# 在板卡上运行。需要 sudo 权限做 iostat -x sda。
set -u
LOGDIR=/tmp/phase_a0
mkdir -p "$LOGDIR"
cd "$LOGDIR"

echo "[A0] 采样 iostat / vmstat / pgmajfault / meminfo..."
iostat -xm 1 sda > iostat.log 2>&1 &
IOSTAT_PID=$!
vmstat 1 > vmstat.log 2>&1 &
VMSTAT_PID=$!
( while true; do
    echo "--- $(date +%T) ---"
    grep -E "pgmajfault|pgfault|pswpin|pswpout" /proc/vmstat
    echo "MemAvailable=$(grep MemAvailable /proc/meminfo | awk '{print $2}') kB"
    sleep 1
  done ) > vmstat_detail.log 2>&1 &
DETAIL_PID=$!

# 冷缓存：先刷掉页缓存，确保测的是真 I/O
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
echo "[A0] 页缓存已刷新，开始 LLM 推理..."

# 运行 Qwen3-4B 推理 200 token
cd /home/intel/QwenTalk
/home/intel/miniforge3/envs/openvino/bin/python - << 'PYEOF' > llm_output.log 2>&1
import openvino_genai as ov_genai
import time
print("[A0] 加载模型...")
t0 = time.perf_counter()
pipe = ov_genai.LLMPipeline("./Qwen3-4B-int4-ov", "GPU")
print(f"[A0] 加载耗时: {time.perf_counter() - t0:.2f}s")

cfg = ov_genai.GenerationConfig()
cfg.max_new_tokens = 200
cfg.do_sample = False

prompts = [
    "用中文详细介绍英特尔 Core Ultra 处理器的架构特点。",
]

# warmup 50 token
pipe.start_chat()
cfg.max_new_tokens = 50
pipe.generate(prompts[0], cfg)

# 正式计时 200 token
cfg.max_new_tokens = 200
t1 = time.perf_counter()
res = pipe.generate(prompts[0], cfg)
t_gen = time.perf_counter() - t1
pipe.finish_chat()
print(f"[A0] 200 token generate 耗时: {t_gen:.2f}s, ~{200/t_gen:.2f} tok/s")
PYEOF

cd "$LOGDIR"
echo "[A0] 推理结束，停止采样..."
kill $IOSTAT_PID $VMSTAT_PID $DETAIL_PID 2>/dev/null
sleep 2

echo ""
echo "========== A0 分析 =========="
echo ""
echo "[1] iostat %util 分布（sda）："
awk '/^sda/ {print $NF}' iostat.log | sort -n | awk '
  { a[NR]=$1 } END {
    n=NR; if(n<2){print "  数据不足"; exit}
    p50=a[int(n*0.5)]; p95=a[int(n*0.95)]; p99=a[int(n*0.99)]; mx=a[n]
    printf "  采样数=%d  p50=%.1f%%  p95=%.1f%%  p99=%.1f%%  max=%.1f%%\n", n, p50, p95, p99, mx
  }'

echo ""
echo "[2] iostat r_await (ms) 分布："
awk '/^sda/ {print $(NF-3)}' iostat.log | sort -n | awk '
  { a[NR]=$1 } END {
    n=NR; if(n<2){print "  数据不足"; exit}
    p50=a[int(n*0.5)]; p95=a[int(n*0.95)]; p99=a[int(n*0.99)]
    printf "  p50=%.1f ms  p95=%.1f ms  p99=%.1f ms\n", p50, p95, p99
  }'

echo ""
echo "[3] iostat rkB/s 峰值与均值："
awk '/^sda/ {sum+=$3; if($3>mx) mx=$3; n++} END {if(n>0) printf "  平均=%.0f kB/s  峰值=%.0f kB/s\n", sum/n, mx}' iostat.log

echo ""
echo "[4] pgmajfault 增速（每秒）："
grep -A0 "pgmajfault" vmstat_detail.log | awk '/pgmajfault/ {print $2}' | awk '
  NR==1 {prev=$1; next}
  { d=$1-prev; if(d<0) d=0; tot+=d; n++; if(d>mx) mx=d; prev=$1 }
  END { if(n>0) printf "  平均=%.0f /s  峰值=%.0f /s\n", tot/n, mx }'

echo ""
echo "[5] MemAvailable 最低点："
grep MemAvailable vmstat_detail.log | awk '{print $2}' | sort -n | head -1 | awk '{printf "  最低 %.2f GB\n", $1/1024/1024}'

echo ""
echo "[6] 推理结果："
grep -E "^\[A0\]" /tmp/phase_a0/../llm_output.log 2>/dev/null || grep -E "^\[A0\]" /home/intel/QwenTalk/llm_output.log 2>/dev/null || grep "^\[A0\]" "$LOGDIR/llm_output.log" 2>/dev/null

echo ""
echo "========== 判定 =========="
echo "按计划阈值表："
echo "  %util p95 > 80 + r_await p95 > 50 + pgmajfault > 1000/s  → SATA 饱和，换 NVMe 有效"
echo "  %util p95 < 30 + pgmajfault < 100/s                       → SATA 不是瓶颈"
