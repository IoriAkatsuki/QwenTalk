#!/bin/bash
# 板卡端 e2e smoke 测试 — 启动 webui server 后验证 6 个关键路径
# 用法（在板卡上）:
#   bash tests/test_smoke_board.sh
#
# 前提:
#   - pyrealsense2 / openvino / fastapi / uvicorn 装好
#   - llama-server 已在 8080 跑（/ws/chat 才能验）
#   - D435 接好（perception 真值才有数据）

set -u

PORT=${PORT:-8765}
HOST=${HOST:-127.0.0.1}
BASE="http://${HOST}:${PORT}"
SERVER_PID=""
PASS=0
FAIL=0
SKIP=0

log() { printf "[%s] %s\n" "$(date +%T)" "$*"; }
pass() { log "PASS: $*"; PASS=$((PASS+1)); }
fail() { log "FAIL: $*"; FAIL=$((FAIL+1)); }
skip() { log "SKIP: $*"; SKIP=$((SKIP+1)); }

cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    sleep 1
}
trap cleanup EXIT

cd "$(dirname "$0")/.."

log "Step 0: 启动 webui server (port $PORT)"
python -m webui.server --host "$HOST" --port "$PORT" > /tmp/intel_chan_smoke.log 2>&1 &
SERVER_PID=$!

code="000"
for i in $(seq 1 30); do
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 "$BASE/api/state" 2>/dev/null)
    [ "$code" = "200" ] && break
    sleep 1
done
if [ "$code" != "200" ]; then
    fail "server 未就绪 ($code)"
    log "log tail:"
    tail -20 /tmp/intel_chan_smoke.log
    exit 1
fi
pass "server ready (port $PORT)"

# Test 1: /api/state 返回 perception 字段
log "Test 1: /api/state"
resp=$(curl -s --max-time 3 "$BASE/api/state")
if echo "$resp" | grep -q "perception"; then
    pass "/api/state 含 perception 字段"
else
    fail "/api/state 缺 perception"
fi

# Test 2: /intel_chan 返回 HTML — 全文 grep 而不是 head 200（"Intel 酱"在 <title>，偏后）
log "Test 2: /intel_chan"
resp=$(curl -s --max-time 5 "$BASE/intel_chan")
if echo "$resp" | grep -qi "intel.*酱\|Live2D\|live2dcubismcore"; then
    pass "/intel_chan 返回 Live2D HTML"
else
    fail "/intel_chan 内容异常 (size=$(echo -n "$resp" | wc -c))"
fi

# Test 3: WS /ws/perception 能收到至少 1 帧
log "Test 3: WS /ws/perception"
python - <<EOF
import asyncio, json, sys
try:
    import websockets
except ImportError:
    sys.exit(2)

async def go():
    async with websockets.connect("ws://${HOST}:${PORT}/ws/perception") as ws:
        msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
        data = json.loads(msg)
        assert "hand" in data, f"missing hand"
        assert "emotion" in data, f"missing emotion"
        print("OK")

asyncio.run(go())
EOF
rc=$?
if [ "$rc" = "0" ]; then
    pass "/ws/perception 推送了 PerceptionState"
elif [ "$rc" = "2" ]; then
    skip "websockets 库未装"
else
    fail "/ws/perception 异常 (rc=$rc)"
fi

# Test 4: WS /ws/chat 流式回应（依赖 llama-server 8080）
log "Test 4: WS /ws/chat"
if curl -s -o /dev/null --max-time 2 "http://${HOST}:8080/v1/models"; then
    python - <<EOF
import asyncio, json, sys
try:
    import websockets
except ImportError:
    sys.exit(2)

async def go():
    async with websockets.connect("ws://${HOST}:${PORT}/ws/chat") as ws:
        await ws.send(json.dumps({"text": "hi", "max_tokens": 30, "thinking": "off"}))
        got_sentence = False
        got_done = False
        for _ in range(60):
            msg = await asyncio.wait_for(ws.recv(), timeout=120.0)
            m = json.loads(msg)
            if m.get("type") == "sentence":
                got_sentence = True
            elif m.get("type") == "done":
                got_done = True
                break
            elif m.get("type") == "error":
                print(f"  error: {m.get('message')}")
                sys.exit(3)
        assert got_sentence and got_done
        print("OK")

asyncio.run(go())
EOF
    rc=$?
    if [ "$rc" = "0" ]; then
        pass "/ws/chat 流式正常"
    elif [ "$rc" = "2" ]; then
        skip "websockets 库未装"
    else
        fail "/ws/chat 异常 (rc=$rc)"
    fi
else
    skip "llama-server 8080 不可达，跳过 chat 测试"
fi

# Test 5: /stream.mjpg multipart 头（依赖 D435 实际出帧，无 D435 则 skip）
# StreamingResponse 在第一个 yield 才发 header；用 GET + range 取头部
log "Test 5: /stream.mjpg"
ct=$(curl -s -o /dev/null -D - --max-time 6 -r 0-0 "$BASE/stream.mjpg" 2>/dev/null \
    | grep -i "content-type" | head -1)
if echo "$ct" | grep -qi "multipart"; then
    pass "/stream.mjpg 返回 multipart"
elif echo "$ct" | grep -qi "text/plain\|text/html"; then
    fail "/stream.mjpg content-type 异常: $ct"
else
    skip "/stream.mjpg 未出帧 (D435 未接 / encoder pending)"
fi

log "===================="
log "RESULT: $PASS passed, $FAIL failed, $SKIP skipped"
[ "$FAIL" = "0" ]
