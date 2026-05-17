#!/usr/bin/env bash
# 夜间无人值守实验脚本 — Intel DK-2500
# Round 3: P0 修复 (B10/B11/B12/B13/B14/B15) + exp7 enable_thinking=false
set -E -o pipefail

# 1. oneAPI (先 source，它引用未设变量)
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1 || {
    echo "FATAL: oneAPI setvars.sh 加载失败"; exit 1
}
# 2. conda openvino env (修复: 所有 python3 走 openvino env)
source ~/miniforge3/etc/profile.d/conda.sh
conda activate openvino

# 3. B11: sudo NOPASSWD preflight (turbostat 需要)
sudo -n true 2>/dev/null || {
    echo "FATAL: sudo NOPASSWD 不可用，turbostat 会卡住"; exit 1
}

set -u

RESULTS=~/overnight_results/$(date +%Y%m%d_%H%M)
mkdir -p "$RESULTS"

BENCH=~/llama.cpp/build_sycl/bin/llama-bench
CLI=~/llama.cpp/build_sycl/bin/llama-cli
SERVER=~/llama.cpp/build_sycl/bin/llama-server
IQ2=~/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
IQ4=~/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
PALM=~/QwenTalk/models/omz/palm_detection_mediapipe_2023feb.onnx
HAND=~/QwenTalk/models/omz/handpose_estimation_mediapipe_2023feb.onnx
FACE=~/QwenTalk/models/omz/yolov8n-face.onnx
SYCL_ARGS="-ngl 99 -fa 0 --cache-type-k q8_0 --cache-type-v f16"
SERVER_PID=""

log() { echo "[$(date '+%H:%M:%S')] $*"; }
done_mark() { touch "$RESULTS/$1.done"; log "✓ $1"; }
fail_mark() { touch "$RESULTS/$1.failed"; log "✗ $1: ${2:-unknown}"; }

run_exp() {
    local name=$1 timeout_s=$2; shift 2
    log "▶ 开始 $name"
    # B13: 用 setsid 给子进程独立进程组，超时时 kill 整个组
    if timeout --foreground --kill-after=10s "$timeout_s" \
            setsid bash -c "$*" > "$RESULTS/${name}.log" 2>&1; then
        done_mark "$name"
    else
        local rc=$?
        fail_mark "$name" "exit=${rc} (timeout=${timeout_s}s)"
        # 兜底: 清掉 8080/8081 server 残留
        pkill -KILL -f "llama-server.*--port 80[0-9]+" 2>/dev/null || true
    fi
}

stop_server() {
    [[ -n "$SERVER_PID" ]] && { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; }
    SERVER_PID=""
}

start_server() {
    local model=$1 ctx=${2:-8192} port=${3:-8080}
    stop_server
    $SERVER -m "$model" $SYCL_ARGS -c "$ctx" --port "$port" \
        --host 127.0.0.1 -np 1 > "$RESULTS/server_${port}.log" 2>&1 &
    SERVER_PID=$!
    local i=0
    while ! curl -sf "http://127.0.0.1:${port}/health" > /dev/null 2>&1; do
        sleep 3; ((i+=3))
        [[ $i -ge 150 ]] && { log "server 启动超时"; return 1; }
    done
    # warmup: 首次推理触发 SYCL 编译
    curl -sf -X POST "http://127.0.0.1:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":8}' \
        > /dev/null 2>&1 || true
    log "llama-server 就绪+预热 (pid=$SERVER_PID)"
}

# ──────────────────────────────────────────────────────────────
health_check() {
    log "=== 健康检查 ==="
    local temp disk_free
    temp=$(cat /sys/class/thermal/thermal_zone1/temp 2>/dev/null || echo 0)
    disk_free=$(df -BG ~ | awk 'NR==2{print $4}' | tr -d G)
    log "CPU 温度: $((temp/1000))°C | 磁盘空闲: ${disk_free}GB"
    [[ $((temp/1000)) -gt 85 ]] && { log "FATAL: CPU 过热"; exit 1; }
    [[ $disk_free -lt 2 ]] && { log "FATAL: 磁盘不足"; exit 1; }
    $BENCH --list-devices 2>&1 | grep -q "Intel" || { log "FATAL: GPU 未找到"; exit 1; }
    python3 -c "import openvino; print('OV:', openvino.__version__)" || { log "FATAL: openvino 不可用"; exit 1; }
    log "健康检查通过"
}

# ──────────────────────────────────────────────────────────────
# exp1/exp4 已在 Round 1 完成，跳过

# ──────────────────────────────────────────────────────────────
exp2_thermal() {
    log "=== 实验 2: 热节流长稳 (sudo turbostat + llama-bench -r 200) ==="

    # 后台 turbostat 采样 (5s 间隔，总 30min)
    sudo turbostat --interval 5 --quiet \
        --show Core,CPU,Avg_MHz,Busy%,PkgWatt,PkgTmp \
        > "$RESULTS/exp2_turbostat.csv" 2>&1 &
    TURBO_PID=$!

    log "turbostat pid=$TURBO_PID 已启动"

    # llama-bench -r 200: 模型加载一次，跑 200 轮 tg32 (~30min)
    $BENCH -m "$IQ2" -ngl 99 -fa 0 \
        --cache-type-k q8_0 --cache-type-v f16 \
        -p 0 -n 32 -ub 512 -r 200 -o json \
        > "$RESULTS/exp2_bench.json" 2>&1

    kill "$TURBO_PID" 2>/dev/null || true
    wait "$TURBO_PID" 2>/dev/null || true

    # 分析: 取 turbostat 首尾 5 行对比温度/频率/功耗
    echo "=== 开始状态 ===" >> "$RESULTS/exp2_bench.json"
    head -6 "$RESULTS/exp2_turbostat.csv" >> "$RESULTS/exp2_bench.json"
    echo "=== 结束状态 ===" >> "$RESULTS/exp2_bench.json"
    tail -6 "$RESULTS/exp2_turbostat.csv" >> "$RESULTS/exp2_bench.json"
}

# ──────────────────────────────────────────────────────────────
exp3_cgroup() {
    log "=== 实验 3: cgroup 隔离效果 ==="

    # NPU 负载进程 PID
    local NPU_PID=""

    start_npu_load() {
        python3 -c "
import openvino as ov, numpy as np, time
core = ov.Core()
m = core.read_model('$PALM')
m.reshape({m.input(0): [1,192,192,3]})
c = core.compile_model(m, 'NPU')
inp = np.zeros((1,192,192,3), dtype=np.float32)
t0 = time.time()
while time.time()-t0 < 300:
    c(inp)
" > /dev/null 2>&1 &
        echo $!
    }

    # 函数: 跑 llama-bench -r 20 tg32，提取 avg_ts
    bench_tg20() {
        local label=$1
        local out
        out=$($BENCH -m "$IQ2" -ngl 99 -fa 0 \
            --cache-type-k q8_0 --cache-type-v f16 \
            -p 0 -n 32 -ub 512 -r 20 -o json 2>/dev/null)
        local avg stddev
        avg=$(echo "$out" | python3 -c "
import sys,json,statistics
data=[r['avg_ts'] for r in json.load(sys.stdin) if r.get('n_gen',0)>0]
print(f'{statistics.mean(data):.3f}') if data else print('err')" 2>/dev/null)
        stddev=$(echo "$out" | python3 -c "
import sys,json,statistics
data=[r['avg_ts'] for r in json.load(sys.stdin) if r.get('n_gen',0)>0]
print(f'{statistics.stdev(data):.3f}') if len(data)>1 else print('0')" 2>/dev/null)
        echo "${label}: avg=${avg} t/s  std=${stddev} t/s"
    }

    log "[A] 基线 (LLM only)"
    bench_tg20 "A_baseline" | tee -a "$RESULTS/exp3_cgroup.txt"

    log "[B] LLM + NPU 并发 (无 cgroup)"
    NPU_PID=$(start_npu_load)
    sleep 5  # 等 NPU 进入稳定推理
    bench_tg20 "B_no_cgroup" | tee -a "$RESULTS/exp3_cgroup.txt"
    kill "$NPU_PID" 2>/dev/null || true; sleep 3

    log "[C] cgroup 隔离: LLM=ai-inference.slice + NPU=ai-perception.slice"
    # B12 修复: 用 --unit 命名而非 $! 捕获包装 PID, --collect 自动清理
    local NPU_UNIT="exp3-npu-load.scope"
    systemd-run --user --slice=ai-perception.slice --scope \
        --unit="$NPU_UNIT" --collect \
        python3 -c "
import openvino as ov, numpy as np, time
core = ov.Core()
m = core.read_model('$PALM')
m.reshape({m.input(0): [1,192,192,3]})
c = core.compile_model(m, 'NPU')
inp = np.zeros((1,192,192,3), dtype=np.float32)
t0 = time.time()
while time.time()-t0 < 300: c(inp)
" > /dev/null 2>&1 &
    sleep 5

    # B15 修复: bench JSON 写文件再解析，避免 systemd-run --scope 的 stdout 污染
    local C_JSON="$RESULTS/exp3_C_cgroup.json"
    systemd-run --user --slice=ai-inference.slice --scope --pty=no \
        bash -c "$BENCH -m '$IQ2' -ngl 99 -fa 0 \
            --cache-type-k q8_0 --cache-type-v f16 \
            -p 0 -n 32 -ub 512 -r 20 -o json > '$C_JSON'" 2>/dev/null || true

    # 验证 JSON 有效再解析
    if [[ -s "$C_JSON" ]] && python3 -c "import json; json.load(open('$C_JSON'))" 2>/dev/null; then
        local avg stddev
        avg=$(python3 -c "
import json,statistics
data=[r['avg_ts'] for r in json.load(open('$C_JSON')) if r.get('n_gen',0)>0]
print(f'{statistics.mean(data):.3f}') if data else print('err')")
        stddev=$(python3 -c "
import json,statistics
data=[r['avg_ts'] for r in json.load(open('$C_JSON')) if r.get('n_gen',0)>0]
print(f'{statistics.stdev(data):.3f}') if len(data)>1 else print('0')")
        echo "C_cgroup_isolated: avg=${avg} t/s  std=${stddev} t/s" | tee -a "$RESULTS/exp3_cgroup.txt"
    else
        echo "C_cgroup_isolated: JSON 解析失败" | tee -a "$RESULTS/exp3_cgroup.txt"
    fi

    # B12: 用 unit 名停止 NPU load (而非 $! 捕获的包装 PID)
    systemctl --user stop "$NPU_UNIT" 2>/dev/null || true
    sleep 2

    log "cgroup 实验结果:"
    cat "$RESULTS/exp3_cgroup.txt"
}

# ──────────────────────────────────────────────────────────────
exp5_npu() {
    log "=== 实验 5: NPU 多模型并发 ==="

    python3 - > "$RESULTS/exp5_npu.txt" 2>&1 <<PYEOF
import openvino as ov, numpy as np, time, threading, statistics

core = ov.Core()

def load_model(path, shape):
    m = core.read_model(path)
    m.reshape({m.input(0): shape})
    return core.compile_model(m, "NPU")

spec = [
    ("$PALM", [1,192,192,3]),
    ("$HAND", [1,224,224,3]),
    ("$FACE", [1,3,640,640]),
]
compiled = [(load_model(p, s), np.zeros(s, dtype=np.float32)) for p, s in spec]
print(f"已加载 {len(compiled)} 个模型")

def bench_single(cm, inp, n=100):
    lat = []
    for _ in range(n):
        t0 = time.perf_counter()
        cm(inp)
        lat.append((time.perf_counter()-t0)*1000)
    return lat

for n_conc in [1, 2, 3]:
    results = [None]*n_conc

    def worker(idx):
        cm, inp = compiled[idx % len(compiled)]
        results[idx] = bench_single(cm, inp, 100)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_conc)]
    t0 = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    total = time.perf_counter()-t0

    all_lat = [x for r in results if r for x in r]
    p99_idx = int(len(all_lat)*0.99)
    print(f"并发={n_conc} | p50={statistics.median(all_lat):.1f}ms "
          f"p99={sorted(all_lat)[p99_idx]:.1f}ms "
          f"total_fps={len(all_lat)/total:.1f}")
PYEOF
    cat "$RESULTS/exp5_npu.txt"
}

# ──────────────────────────────────────────────────────────────
exp6_vlm() {
    log "=== 实验 6: SmolVLM2 连续帧稳定性 (300 帧) ==="

    # D435 帧捕获 (conda cv2 可用)
    python3 - > /dev/null 2>&1 <<'PYEOF'
import cv2, numpy as np
for dev in [2, 0, 4]:
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    if cap.isOpened():
        for _ in range(5): cap.read()
        ret, f = cap.read()
        cap.release()
        if ret:
            cv2.imwrite('/tmp/test_frame.jpg', f)
            print(f"D435 帧捕获成功 (/dev/video{dev})")
            break
else:
    # fallback
    img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    cv2.imwrite('/tmp/test_frame.jpg', img)
    print("使用合成帧 fallback")
PYEOF

    local SMOL_M=~/models/smolvlm2-2.2b-q4km.gguf
    local SMOL_MM=~/models/smolvlm2-2.2b-mmproj-f16.gguf

    $SERVER -m "$SMOL_M" --mmproj "$SMOL_MM" \
        -ngl 0 -t 8 -c 2048 --port 8081 --host 127.0.0.1 \
        > "$RESULTS/exp6_server.log" 2>&1 &
    VLM_PID=$!

    local i=0
    while ! curl -sf http://127.0.0.1:8081/health > /dev/null 2>&1; do
        sleep 3; ((i+=3))
        [[ $i -ge 120 ]] && { log "VLM server 启动超时"; kill "$VLM_PID" 2>/dev/null; return 1; }
    done
    log "SmolVLM2 server 就绪"

    # B14 修复: 用 /completion + <__image__> marker, 不用 /v1/chat/completions
    # /v1/chat/completions 在 SmolVLM2 模板下不注入 image marker, 返回 200 但 content 全空
    python3 - > "$RESULTS/exp6_vlm.txt" 2>&1 <<'PYEOF'
import requests, base64, time, resource, statistics

with open("/tmp/test_frame.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

PROMPT = ("<|im_start|>User: <__image__>描述图像，10字以内"
          "<end_of_utterance>\n<|im_start|>Assistant:")

times, mem_samples = [], []
errors, empty_contents = 0, 0

for i in range(300):
    t0 = time.perf_counter()
    content = ""
    try:
        r = requests.post("http://127.0.0.1:8081/completion", json={
            "prompt": PROMPT,
            "image_data": [{"data": b64, "id": 1}],
            "n_predict": 25,
            "temperature": 0.3,
            "stop": ["<end_of_utterance>"],
        }, timeout=30)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        content = (r.json().get("content") or "").strip()
        if not content:
            empty_contents += 1
    except Exception as e:
        errors += 1
        content = f"ERR:{e}"
        elapsed = time.perf_counter() - t0

    if i % 50 == 0:
        mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
        mem_samples.append(mem_mb)
        print(f"帧{i:3d}: {elapsed:.2f}s RSS={mem_mb}MB | {content[:30]}", flush=True)

print(f"\n--- 统计 ---")
print(f"总帧={len(times)} 错误={errors} 空响应={empty_contents}")
if times:
    print(f"均值={statistics.mean(times):.2f}s p99={sorted(times)[int(len(times)*0.99)]:.2f}s")
print(f"内存采样(MB): {mem_samples}")
leak = (mem_samples[-1] - mem_samples[0]) if len(mem_samples) >= 2 else 0
print(f"内存漂移: {leak:+d}MB ({'⚠ 可能泄漏' if leak > 100 else '✓ OK'})")
# 真实成功率检查
real_ok = len(times) - empty_contents - errors
print(f"真成功: {real_ok}/300 = {real_ok/300:.0%}")
PYEOF

    kill "$VLM_PID" 2>/dev/null || true
    wait "$VLM_PID" 2>/dev/null || true
}

# ──────────────────────────────────────────────────────────────
exp7_agent() {
    log "=== 实验 7: Agent / Function Calling 测试 ==="

    start_server "$IQ2" 8192 8080 || { log "server 启动失败"; return 1; }
    log "预热完成，开始测试"

    # 7a: 原生 function calling
    log "[7a] 原生 function calling"
    python3 - > "$RESULTS/exp7a_native.txt" 2>&1 <<'PYEOF'
import requests, time

tools = [{
    "type": "function",
    "function": {
        "name": "get_system_info",
        "description": "获取系统信息",
        "parameters": {"type":"object","properties":{
            "metric":{"type":"string","enum":["cpu","memory","temperature"]}
        },"required":["metric"]}
    }
},{
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "数学计算",
        "parameters": {"type":"object","properties":{
            "expression":{"type":"string"}
        },"required":["expression"]}
    }
}]

cases = [
    ("CPU状态查询", "查询 CPU 状态"),
    ("数学计算", "计算 2的32次方"),
    ("复合任务", "查询内存，然后计算 1024 乘以 1024"),
]

success = 0
for name, prompt in cases:
    t0 = time.time()
    try:
        # P0 修复: enable_thinking=false (之前 0/3 失败的根因)
        r = requests.post("http://127.0.0.1:8080/v1/chat/completions", json={
            "messages": [{"role":"user","content":prompt}],
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": 256,
            "temperature": 0.1,
            "chat_template_kwargs": {"enable_thinking": False},
        }, timeout=120).json()
        elapsed = time.time()-t0
        msg = r.get("choices",[{}])[0].get("message",{})
        tc = msg.get("tool_calls")
        called = tc[0]["function"]["name"] if tc else "(none)"
        ok = tc is not None
        success += ok
        print(f"[{'✓' if ok else '✗'}] {name}: → {called} ({elapsed:.1f}s)")
        if tc:
            print(f"    args: {tc[0]['function'].get('arguments','')[:80]}")
    except Exception as e:
        print(f"[✗] {name}: EXCEPTION {e}")
        elapsed = time.time()-t0

print(f"\n成功率: {success}/{len(cases)}")
PYEOF

    # 7b: IronClaw grammar 模式 (验证之前的 bug 是否修复)
    log "[7b] IronClaw grammar 模式"
    python3 - > "$RESULTS/exp7b_ironclaw.txt" 2>&1 <<'PYEOF'
import requests, time

# 测试 1: 简单 JSON grammar (短上下文)
def test_grammar(prompt, schema, label):
    t0 = time.time()
    try:
        # P0 修复: enable_thinking=false (grammar 模式同病)
        r = requests.post("http://127.0.0.1:8080/v1/chat/completions", json={
            "messages": [{"role":"user","content":prompt}],
            "max_tokens": 64,
            "response_format": {"type": "json_schema", "json_schema": {"schema": schema}},
            "chat_template_kwargs": {"enable_thinking": False},
        }, timeout=90)
        elapsed = time.time()-t0
        body = r.json()
        content = body.get("choices",[{}])[0].get("message",{}).get("content","")
        print(f"[{r.status_code}] {label} ({elapsed:.1f}s): {content[:80]}")
        return r.status_code == 200 and len(content) > 0
    except Exception as e:
        print(f"[ERR] {label}: {e}")
        return False

ok1 = test_grammar(
    "你好，用中文回一句话",
    {"type":"object","properties":{"reply":{"type":"string"}},"required":["reply"]},
    "简单 JSON 输出"
)
ok2 = test_grammar(
    "列出3种常见水果",
    {"type":"object","properties":{"fruits":{"type":"array","items":{"type":"string"}}},"required":["fruits"]},
    "JSON 数组输出"
)

print(f"\ngrammar 测试: {sum([ok1,ok2])}/2 通过")
if ok1 or ok2:
    print("✓ grammar 编译器可用（之前 16K bug 可能已修复）")
else:
    print("✗ grammar 仍有问题")
PYEOF

    stop_server
    echo "=== 7a ===" && cat "$RESULTS/exp7a_native.txt"
    echo "=== 7b ===" && cat "$RESULTS/exp7b_ironclaw.txt"
}

# ──────────────────────────────────────────────────────────────
summary() {
    log "=== 汇总报告 ==="
    {
        echo "# 夜间实验报告 Round 2 — $(date '+%Y-%m-%d %H:%M')"
        echo ""
        for e in exp2 exp3 exp5 exp6 exp7; do
            if [[ -f "$RESULTS/$e.done" ]]; then
                echo "✓ $e"
            elif [[ -f "$RESULTS/$e.failed" ]]; then
                echo "✗ $e (FAILED)"
            else
                echo "? $e"
            fi
        done
        echo ""
        echo "## cgroup 隔离"
        [[ -f "$RESULTS/exp3_cgroup.txt" ]] && cat "$RESULTS/exp3_cgroup.txt"
        echo "## NPU 并发"
        [[ -f "$RESULTS/exp5_npu.txt" ]] && cat "$RESULTS/exp5_npu.txt"
        echo "## Function Calling"
        [[ -f "$RESULTS/exp7a_native.txt" ]] && tail -5 "$RESULTS/exp7a_native.txt"
        echo "## Grammar 模式"
        [[ -f "$RESULTS/exp7b_ironclaw.txt" ]] && tail -5 "$RESULTS/exp7b_ironclaw.txt"
    } > "$RESULTS/summary.txt"
    cat "$RESULTS/summary.txt"
}

# ──────────────────────────────────────────────────────────────
main() {
    trap 'stop_server; log "退出"; summary' EXIT

    log "======================================"
    log "Round 2 开始 → $RESULTS"
    log "======================================"

    health_check

    # Round 1 已完成: exp1 (LLM bench) exp4 (zram)
    run_exp exp2 2400 'exp2_thermal'   # ~30min (turbostat + bench -r 200)
    run_exp exp3 2400 'exp3_cgroup'    # ~40min (A/B/C × 20 rounds each)
    run_exp exp5  600 'exp5_npu'       # ~5min
    run_exp exp6 2400 'exp6_vlm'       # ~30min (300 frames)
    run_exp exp7 1800 'exp7_agent'     # ~20min

    summary
    log "======================================"
    log "Round 2 完成 → $RESULTS/summary.txt"
    log "======================================"
}

export -f log done_mark fail_mark stop_server start_server
export -f exp2_thermal exp3_cgroup exp5_npu exp6_vlm exp7_agent
export RESULTS BENCH CLI SERVER IQ2 IQ4 PALM HAND FACE SYCL_ARGS SERVER_PID

main "$@"
