# 过夜 Handoff — Intel 酱 Phase A-I

**起始**: 2026-05-16 01:55
**完成时间**: (TBD, target 08:00)

## 当前进度
✅ Phase A: git baseline (`847ab4e`)
✅ Phase B: PerceptionFusion 模块 (`f508594`)
✅ Phase C: WebSocket /ws/perception + /ws/chat (`58e24c2`)
✅ Phase D: intel_chan.html 前端 (`d2ff476`)
✅ Phase E: 摸摸头 hit-test (`9b515d2`)
✅ Phase F: event bus + trigger rules (`e12936f`)
✅ Phase G: Intel 酱 persona (`0201db4`)
🔄 Phase H: codex review + fix (in progress)
⏳ Phase I: HANDOFF + project.md 进度刷新

## 文件清单（本次新增）
| 文件 | 行数 | 用途 |
|---|---|---|
| `webui/perception_fusion.py` | 179 | 5Hz 融合 hand/emotion/gaze/face，订阅广播 |
| `webui/server.py` | 247 | FastAPI + WS perception/chat（原 106 行扩展） |
| `intel_chan.html` | 267 | Live2D + WS 前端，跨端兼容 |
| `webui/head_pat_detector.py` | 148 | 摸摸头状态机 |
| `webui/event_bus.py` | 172 | EventBus + Distance/Gaze/Silence 规则 |
| `intel_chan_persona.py` | 156 | 动态 system prompt + proactive 台词 |
| `OVERNIGHT_PLAN.md` | — | 本次过夜计划 |

## 现状（还未跑通）
本次过夜**完成了代码架构**，**未做实际部署测试**。下一步上板卡跑：

1. **scp webui/ + intel_chan.html + intel_chan_persona.py 到板卡**
2. 板卡装 fastapi / uvicorn / websockets 依赖（openvino env 应该有）
3. 启动 `python -m webui.server --host 0.0.0.0 --port 8000`
4. 浏览器打开 `http://192.168.1.8:8000/intel_chan` 看 Live2D 渲染
5. 看 `/ws/perception` 数据流（占位 source 都是默认值，模型不会动）
6. 测 `/ws/chat` 是否能流式回（依赖板卡 llama-server 8080 在跑）

## 已知问题（待 codex review 确认）
- `_broadcast_perception_sync` 用 `asyncio.get_event_loop()` 在非 asyncio 线程里 — **可能拿到错的 loop**，需要在 startup 时保存 loop 引用
- `intel_chan.html` 假设 haru 模型有 `ParamCheek` — **待板卡验证**（haru 可能没这个参数，需要 fallback）
- `_ws_perception_clients` 集合的并发安全 — handler 在 perception 线程里被调（同步），但 asyncio 端遍历可能 race

## 重连测试快速命令
```bash
# 在 oasis 本机
cd /home/oasis/Documents/Intel/QwenTalk
git log --oneline -10              # 看过夜 7 commits
cat OVERNIGHT_PLAN.md              # 看原始计划
cat HANDOFF.md                     # 这个文件

# 看 codex review 结果（agent id ab73b4b8bd11b790e）
# 在 claude 里 SendMessage(to='ab73b4b8bd11b790e', ...)
```

## 明日 Day 1 建议
1. **先部署测试**：把过夜代码 scp 板卡，跑通 WS 数据流，verify intel_chan.html 加载
2. **NPU gaze 编译**：下载 gaze-estimation-adas-0002 → NPU 编译 → 替换 _NullGazeSource 实现
3. **NPU emotion 接入**：替换 _NullEmotionSource → 真 emotions-recognition-retail-0003
4. **head_pat_detector 接进 GesturePipeline**：每帧 update palm_xy + depth，触发 event
5. **MeloTTS 集成**：装依赖进 openvino env，替换 espeak

## 重要时间戳
- 03:00 左右 LLM 还在 server tmux 跑（不要停）
- codex review agent 已 dispatch
