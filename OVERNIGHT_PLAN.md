# Intel 酱 过夜自动探索计划

**起始**: 2026-05-16 01:56
**deadline**: 2026-05-16 08:00 (用户回来)
**budget**: ~6 小时

## 约束
- 板卡 llama-server 8.37 baseline 不停 (free 仅 496MB UMA 紧)
- 重心放本机：写代码 + git commit + codex review/debug
- 板卡只做轻查询（不跑 OpenVINO/大模型）

## Phase 列表

| Phase | 目标 | 工时 | 产物 |
|---|---|---|---|
| **A** | OVERNIGHT_PLAN + git baseline | 20m | 0 commit (baseline) |
| **B** | perception_pipeline 扩展 (gaze fusion 占位 + emotion fusion) | 45m | 1 commit |
| **C** | webui server WS endpoints (/ws/perception + /ws/chat) | 30m | 1 commit |
| **D** | intel_chan.html 接 WS + 聊天 UI | 60m | 1 commit |
| **E** | 摸摸头 hit-test 模块 (head_pat_detector.py) | 30m | 1 commit |
| **F** | event bus + trigger framework | 45m | 1 commit |
| **G** | Intel 酱 persona + 记忆注入 | 30m | 1 commit |
| **H** | codex review + fix | 75m | 1+ commit |
| **I** | HANDOFF.md + project.md 进度更新 | 30m | 1 commit |

总: ~6h, 7-8 commits

## 设计原则
1. **不破坏现有 LLM baseline** — 板卡 llama-server 不动
2. **本机写代码 + git** — 板卡作 reference 只读
3. **失败 graceful** — 每 phase 失败写 FAIL_phase_X.md 跳到下一个，不 cascading
4. **代码可读** — 文件 < 300 行规范、函数 < 50、嵌套 ≤ 3
5. **codex review 必上** — Phase H 调 codex-rescue 审查所有代码

## 暂不做（明天工作）
- 板卡 NPU 编译 gaze/age-gender（需 X11 + free RAM）
- 板卡跑 perception_pipeline 实测
- 端到端 demo run (NPU + LLM + Live2D 全栈)
- MeloTTS venv 整合（需要装包，挤现有 disk）
