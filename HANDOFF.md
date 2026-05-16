# 过夜 Handoff — Intel 酱 Phase A-I 完整版

**起始**: 2026-05-16 01:55
**完成**: 2026-05-16 02:35 板卡 e2e 验证通过 ✅
**目标 deadline**: 2026-05-16 08:00 — 提前完成

## 🎯 重点：板卡端到端验证通过

```
[02:33:35] RESULT: 6 passed, 0 failed, 0 skipped ✅
  - server ready (port 8000)
  - /api/state 含 perception
  - /intel_chan 返回 Live2D HTML
  - /ws/perception 推送 PerceptionState
  - /ws/chat 流式正常 (真实调 llama-server)
  - /stream.mjpg multipart
```

**意味着**: Intel 酱后端 + 前端在板卡上完整工作；浏览器打开
http://192.168.1.8:8000/intel_chan 应该能看到 Live2D + 聊天 UI 联动。

## 完成进度（所有 Phase 都 ✅）

| Phase | commit | 内容 |
|---|---|---|
| A | `847ab4e` | git baseline — 把 36 个未追踪文件首次入库 |
| B | `f508594` | `webui/perception_fusion.py` (179) — PerceptionFusion 融合 + 占位 NPU source |
| C | `58e24c2` | `webui/server.py` (+155 行) — WS /ws/perception + /ws/chat |
| D | `d2ff476` | `intel_chan.html` (267) — Live2D + WS client + 流式聊天 UI |
| E | `9b515d2` | `webui/head_pat_detector.py` (148) — 摸摸头状态机 |
| F | `e12936f` | `webui/event_bus.py` (172) — EventBus + 3 触发规则 |
| G | `0201db4` | `intel_chan_persona.py` (156) — 动态 system prompt + proactive 台词 |
|   | `21efe82` | smoke tests + HANDOFF baseline |
| H | `9ae7fcb` | **codex review 7 项 fix**（2 CRITICAL + 3 HIGH + 2 MED）|
| I | `0dc3ae2` | HANDOFF + board e2e smoke 脚本 |
| J | `fd6d2ee` | **板卡 e2e 验证 6/6 通过** + FastAPI lifespan |
| K | `52a4a3e` | HANDOFF 标记 e2e PASS |
| L | `3de37a6` | head.pat event 通道 (codex follow-up) |
| M | `3fba412` | **方案 A**: pipeline.start() 同步等 init + health_status (fan-out #1) |
| N | `6419faa` | smoke 加 init-failure 传播测试 (fan-out #3) |
| O | `bf66c27` | **方案 B**: `/api/health` endpoint (fan-out #2) |

## 终态验证

**板卡 unit smoke 23/23 ✅**（含 3 个新 contract 测试）
**板卡 e2e smoke 6/6 ✅** (`/api/state` + `/intel_chan` + WS + `/stream.mjpg`)
**`/api/health` 实测返回**:
```json
{
  "pipeline": {"alive": true, "ready": true, "init_error": null},
  "perception": {"thread_alive": true, "subscribers": 1},
  "llama_server": {"reachable": true, "url": "http://127.0.0.1:8080/v1/models"},
  "status": "ok"
}
```

Codex follow-up 标的 2 隐性问题全部修了：
- ✅ pipeline init 失败感知 — start() 现在同步等 ready event + re-raise；health_status 暴露
- ✅ head.pat 注入点 — PerceptionFusion.push_event() 接口已就位

## Codex Review 反馈处理（agent `ab73b4b8bd11b790e`）

**已 fix**：
- ✅ **CRITICAL-1**: `asyncio.get_event_loop()` 在 perception 线程会抛 RuntimeError → 全局 `_main_loop` + `run_coroutine_threadsafe`
- ✅ **HIGH-1**: `DistanceWakeRule` `distance<0.2` 误判 APPROACH → 直接 return + 回归测试
- ✅ **HIGH-2**: `run_in_executor` 未 await → `asyncio.ensure_future` + 180s timeout
- ✅ **HIGH-3**: `sys.path.insert` 每请求累加 → 移到 module 顶层 + 检查
- ✅ **MED-1**: `SilenceRule` 启动 30s 必触发 → 加 `user_present` 参数 + 回归测试
- ✅ **MED-2**: `_build_state` 无异常保护 → try/except 守住后台线程
- ✅ **MED-3**: `target.cheek` 永为 0 → happy 推断 base + cheekPulse 衰减 + event 触发

**留待板卡验证**（codex 标 MED）：
- ⚠️ `ParamCheek` 在 haru 模型是否存在（静默失败，不影响其他参数）
- ⚠️ NPU emotion 真实概率分布 → 用真值 calibrate sigmoid 拐点 (0.20)

## Smoke 测试

### 本机（已跑过）
```bash
cd /home/oasis/Documents/Intel/QwenTalk
python3 -m unittest tests.test_smoke -v
# 结果: 18 passed + 1 skipped (server route 需 pyrealsense2 在板卡)
```

### 板卡端 e2e（已跑过 ✅）
```bash
ssh intel@192.168.1.8 'cd /home/intel/QwenTalk && \
  PATH=/home/intel/miniforge3/envs/openvino/bin:$PATH \
  bash tests/test_smoke_board.sh'
# 已验证: 6/6 PASS at 2026-05-16 02:33
#   1) /api/state 含 perception        ✅
#   2) /intel_chan 返回 Live2D HTML     ✅
#   3) WS /ws/perception 推送            ✅
#   4) WS /ws/chat 流式回应 (llama调通)  ✅
#   5) /stream.mjpg multipart 头         ✅
```

## 板卡部署步骤（明早第一件事）

```bash
# 1. 同步代码到板卡
rsync -avz /home/oasis/Documents/Intel/QwenTalk/ intel@192.168.1.8:/home/intel/QwenTalk/

# 2. 板卡上装 webui 缺的依赖（如有）
ssh intel@192.168.1.8 '/home/intel/miniforge3/envs/openvino/bin/pip install fastapi uvicorn websockets'

# 3. 跑 board smoke
ssh intel@192.168.1.8 'cd /home/intel/QwenTalk && \
    /home/intel/miniforge3/envs/openvino/bin/python -m webui.server --host 0.0.0.0 --port 8000 &
    sleep 30 && bash tests/test_smoke_board.sh'

# 4. 浏览器 (本机/手机) 打开
#    http://192.168.1.8:8000/intel_chan
```

## 文件清单（本次新增）

| 文件 | 行数 | 用途 |
|---|---|---|
| `OVERNIGHT_PLAN.md` | 41 | 计划 |
| `webui/perception_fusion.py` | 181 | 5Hz 融合 + 异常保护 |
| `webui/server.py` | 256 | WS /ws/perception + /ws/chat + loop 桥接 fix |
| `webui/head_pat_detector.py` | 148 | 摸摸头状态机 |
| `webui/event_bus.py` | 180 | EventBus + Distance/Gaze/Silence 规则 |
| `intel_chan.html` | 287 | Live2D + WS + cheek 脉冲 |
| `intel_chan_persona.py` | 156 | 动态 system prompt |
| `tests/test_smoke.py` | 305 | 19 个 smoke test |
| `tests/test_smoke_board.sh` | 152 | 板卡 e2e smoke |
| `HANDOFF.md` | 本文件 | 交接 |

## 明日 Day 1 建议（按优先级）

1. ~~**跑板卡 smoke**~~ ✅ 已完成 6/6 PASS（02:33）
2. **浏览器实测**（30 min）— 启动 webui 后打开 http://192.168.1.8:8000/intel_chan
   看 Live2D 渲染 + 跟它聊几句（用 LLM streaming 返回）
3. **NPU gaze 编译**（半天）— `gaze-estimation-adas-0002`
4. **NPU emotion 替换** `_NullEmotionSource` → 真 emotion 概率喂 PerceptionFusion
5. **head_pat_detector 接进 GesturePipeline** → palm_3d 字段推 WS
6. **MeloTTS 集成** voice_pipeline → 替换 espeak
7. **形象替换** AIGC + Cubism rig (比赛前 1 周做)

## 重要时间戳 + 状态
- 板卡 llama-server 8080 仍在 tmux llmsrv 跑（8.18 t/s baseline）— **不要停**
- D435 未启动（本次过夜没动板卡硬件）
- 本机 git branch beta，**领先 origin/beta 9 个 commit**（明早决定是否 push）

## 重连命令
```bash
cd /home/oasis/Documents/Intel/QwenTalk
git log --oneline -15            # 看 overnight 9 commits
cat HANDOFF.md                   # 本文件
python3 -m unittest tests.test_smoke -v   # 跑 smoke 验证仍 OK
```
