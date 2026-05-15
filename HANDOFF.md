# 过夜 Handoff — Intel 酱 Phase A-I

**起始**: 2026-05-16 01:55
**完成**: 2026-05-16 02:30 左右（提前完成）
**目标 deadline**: 2026-05-16 08:00

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
| I | _本 commit_ | 最终 HANDOFF + board e2e smoke 脚本 |

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

### 板卡端 e2e（待跑）
```bash
ssh intel@192.168.1.8
cd /home/intel/QwenTalk
bash tests/test_smoke_board.sh
# 启动 webui server + 验证 5 个 path:
#   1) /api/state 含 perception
#   2) /intel_chan 返回 HTML
#   3) WS /ws/perception 推送 PerceptionState
#   4) WS /ws/chat 流式回应 (依赖 llama-server)
#   5) /stream.mjpg multipart 头
```

## 板卡部署步骤（明早第一件事）

```bash
# 1. 同步代码到板卡
rsync -avz /home/oasis/Documents/Intel/QwenTalk/ intel@192.168.1.8:/home/intel/QwenTalk/

# 2. 板卡上装 webui 缺的依赖（如有）
ssh intel@192.168.1.8 '/home/intel/miniforge3/envs/openvino/bin/pip install fastapi uvicorn websockets'

# 3. 跑 board smoke
ssh intel@192.168.1.8 'cd /home/intel/QwenTalk && \
    /home/intel/miniforge3/envs/openvino/bin/python -m webui.server --host 0.0.0.0 --port 8765 &
    sleep 30 && bash tests/test_smoke_board.sh'

# 4. 浏览器 (本机/手机) 打开
#    http://192.168.1.8:8765/intel_chan
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

1. **跑板卡 smoke**（30 min）— `bash tests/test_smoke_board.sh` 验证 4-5 项通过
2. **浏览器实测**（30 min）— 打开 /intel_chan 看 Live2D 跑动 + 聊天流畅
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
