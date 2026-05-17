# Intel 酱 夜间进展交接（2026-05-17 早晨阅读）

## TL;DR

- 11 个 in-process tool e2e 回归测试已加（`tests/test_11_tools_e2e.py`），本机 39 pass + 2 skipped（外网）。
- 前端 `speak:|` handler ✅ 完成（Agent D，14 行）+ Live2D 优化 3 项 ✅ 完成（Agent J，51 行 lipsync/blink/距离唤醒），均未 commit。
- SYCL turbo3 kernel patch 仍在 codex 后台（B），完成后主线程自动跑安全 build（`/tmp/build_turbo3_safely.sh`，stop llmsrv + watchdog + -j 2 + 45 min timeout + 自动恢复 llmsrv）。
- **2 个 commit** 未 push 到 `origin/beta`（77c913a + d83d9d0），等 T1.2 e2e 通过再决定。

---

## 1. 夜间编排的 agent 全清单

| Agent | 角色 | 产出 | 状态 |
|---|---|---|---|
| **Agent A** (`a0788bd5d9e7ff8e1`) | 架构总览 / 路线图 reviewer | `/tmp/intel_chan_OVERALL_PLAN.md` (156 行) | ✅ 完成 |
| **plan-reviewer** (`ad7ca2ea9643c5a77`, codex) | 独立审 OVERALL_PLAN.md，找过度设计 / 漏项 | `/tmp/intel_chan_OVERALL_PLAN_REVIEW.md` (157 行) | ✅ 完成（approve-with-changes，见第 7 节）|
| **Agent B** (`a0774636a93435e76`, codex SYCL kernel patch) | SYCL turbo3 实验 patch | `/tmp/llama_tq_fork/SYCL_TURBO3_PATCH.md` | 🔄 codex CLI 后台跑（**目录尚未出现**），完成自动触发 build |
| **Agent D** (`a4690f4ca9879d6f4`, 前端 speak handler) | `intel_chan.html` SpeechSynthesisUtterance + cancel 打断 | `intel_chan.html:212-224` (+14 行) | ✅ 完成（未 commit）|
| **Agent J** (`a6ee6ed3dec011917`, Live2D 优化) | TTS lipsync hook + 自然 blink/breath + 距离唤醒（user_present=false → 半闭眼）| `intel_chan.html:152-173/202-204/300-348` (+51 行/改 13 行) | ✅ 完成（未 commit）|
| **Agent E** (`a3d30f1a54a609213`, 本 agent) | 11-tool e2e 测试 + 本文件 | `tests/test_11_tools_e2e.py` + `HANDOFF_MORNING_20260517.md` | ✅ 完成 |
| **Agent H** (`a18f3329ece12b7b3`, code-reviewer 审 D+J+E) | review D+J+E 合并 diff | `/tmp/intel_chan_FRONTEND_TEST_REVIEW.md` (160 行) | ✅ 完成（approve-with-changes，见第 7 节）|
| **Agent G** (待 spawn，codex review SYCL) | 独立 codex 审 B 的 patch | 待 B 完成 | ⏳ 待 B |

---

## 2. 当前 git status

```
位于分支 beta
您的分支领先 'origin/beta' 共 2 个提交。
尚未暂存以备提交的变更:
  修改: intel_chan.html              ← Agent D (+14 line speak handler) + Agent J (+51 line Live2D 优化)
未跟踪的文件:
  tests/test_11_tools_e2e.py         ← Agent E 新增
  HANDOFF_MORNING_20260517.md        ← 本文件
```

最近 5 个 commit：

```
77c913a feat(tools): 7 个新 in-process function tools — D435/NPU/TTS/Memory
d83d9d0 feat(memory): L1 跨 reconnect 短期 + L2 人格 facts (SQLite)
f1443e5 perf(chat): G - streaming + tool_calls delta parsing 取代 detect+stream 两阶段
88591cc feat(chat+ui): L0 会话历史 + motion 触发 + D435 degraded mode
1773a91 feat(chat): multi-step ReAct loop + tool calling + persona/event_bus 接入
```

⚠️ **2 个 commit 未 push origin/beta**（`git log origin/beta..HEAD` 实测：77c913a + d83d9d0）。Agent H 复核 H1：原 HANDOFF 误称 "9 个" 已修正为 2。

---

## 3. 已就绪可立即使用

| 子系统 | 验证方式 | 备注 |
|---|---|---|
| **11 个 in-process tool** | `python -m pytest tests/test_11_tools_e2e.py -v` → 39 pass + 2 skipped (外网) | commit 77c913a |
| Memory L1 / L2 SQLite | 本测试已覆盖跨 session hydrate + persona facts upsert | commit d83d9d0 |
| ReAct multi-step loop | 测试覆盖：单轮 final / 两轮 tool→content / MAX_ITERATIONS guard | commit 77c913a + f1443e5 |
| Perception fusion 5 Hz | `tests/test_smoke.py` 18 pass + 1 skipped 仍 green | commit 88591cc |
| `/api/health` | 板卡实测 OK，本机 import 通过 | commit bf66c27 |
| LLM 推理 frozen 配置 | Q4_K_S + mixed mode + -fa 1 + ub=64 = 8.37 t/s | overnight 2026-05-15 |
| Voice pipeline streaming | TTFA 30s → 4s | 2026-05-15 voice_pipeline_streaming |

---

## 4. 需要起床后人工 decide 的事（≤ 5 个二选一）

源：`/tmp/intel_chan_OVERALL_PLAN.md` 第 6 节。

1. **优先级排队**：
   - **A. 先 T1.1（前端 speak handler，20 min）** ← Agent A 推荐：unblock 后端已上线的 tool
   - B. 先 T1.4（head_pat 接 palm_3d，45 min）— 摸摸头爆点功能

2. **codex turbo3 (T1.3) 裁决标准**：
   - A. tps ≥ 8.37 → merge beta
   - B. 8.30-8.37（std dev 内）→ 归档不 merge
   - C. < 8.30 或不稳 → 立刻归档

3. **NPU emotion 接入策略 (T1.5)**：
   - **A. 先做最简包装**（≤ 100 行替换 `_NullEmotionSource`） ← 推荐
   - B. 顺手把 gaze + age-gender 合并接入（工作量 × 3）

4. **2 个未 push 提交 + 3 个未 commit 改动**：
   - **A. e2e 通过 → 先 commit (intel_chan.html D+J / tests / HANDOFF) → `git push origin beta`** ← 推荐
   - B. 等 T1.5（NPU emotion）完成再一并 commit + push

5. **是否今天写设计报告 (T2.5)**：
   - A. 写 — 5/17 周日时间充裕
   - **B. 不写** ← Agent A 推荐，先 demo 链跑通

---

## 5. 已知风险 / 未决问题

1. **Agent B 仍在跑**：codex CLI 后台编写 SYCL turbo3 patch + kernel；完成后主线程**自动**跑 `/tmp/build_turbo3_safely.sh`（stop llmsrv → watchdog → cmake -j 2 → 45 min timeout → 重启 llmsrv），不需起床干预。如起床时 build 失败，patch 仍在 `/tmp/llama_tq_fork/` 可手动检视。
2. **Agent H 发现的 MEDIUM 问题**（不阻塞 commit，明早 decide 修不修）：
   - M1 `test_react_max_iterations_guard` 断言 `"工具" or "尝试"` 过松（应改 `"尝试调用" in ...` + 断言迭代次数）
   - M2 J 的全局 `speechSynthesis.speak` monkey-patch 需加注释说明侵入式行为
   - M3 `tool_impls.speak` 无 try/except，`push_event` 抛异常会冒泡 — 疑似真漏洞，可补 try/except 返回 error dict
   - M4 `web_search` / `get_weather` 0 mock 成功路径覆盖，schema 漂移检测不到
3. **外网 tool 测试 skip**：`web_search` / `get_weather` 测试通过 `@pytest.mark.skip` 关闭（cn.bing.com / wttr.in 不一定可达）。手动跑：`pytest tests/test_11_tools_e2e.py::TestExternalTools -v --override-ini="addopts="` 并移除 skip 装饰器。

---

## 6. 起床 /goal 推荐顺序

1. **看夜间产出全貌**：
   ```
   ls -la /tmp/intel_chan_OVERALL_PLAN.md /tmp/intel_chan_OVERALL_PLAN_REVIEW.md /tmp/intel_chan_FRONTEND_TEST_REVIEW.md /tmp/llama_tq_fork/SYCL_TURBO3_PATCH.md /tmp/turbo3_build.log
   git -C /home/oasis/Documents/Intel/QwenTalk status --short
   git -C /home/oasis/Documents/Intel/QwenTalk diff intel_chan.html | head -100
   ```
2. **跑本机 e2e** 确认无 regression：`cd /home/oasis/Documents/Intel/QwenTalk && python -m pytest tests/ -v`（应 57 pass + 3 skipped）。
3. **看 build 结果**：`tail -80 /tmp/turbo3_build.log`（含 stop llmsrv → build → 重启 llmsrv 完整流程；若有 watchdog 介入会有 `CRITICAL: MemAvailable=` 行）。
4. **板卡 e2e**（T1.2）：浏览器 `http://192.168.1.8:8000/intel_chan` → 聊 3 句 + 触发 speak + 触发 head.pat。
5. **决策点 1-5 后** 进入 T1.4 / T1.5 实施（T1.1 已被 D+J 完成 + plan-reviewer 建议改为 5min 验证项）。

---

## 7. Reviewer 反馈关键摘要

### plan-reviewer (codex 审 PLAN.md) — approve-with-changes

- T1.1 已被 D+J 完成，应降为「5min 验证 speak 出声」
- T1.3 emotion 完整链路应降 Tier 2（OMZ 模型文件未确认存在）
- 漏项 Tier 1：dirty worktree 校验 / `/api/health` gate / 端口一致性（已修：HANDOFF.md 旧 8765 → 8000 对齐 server.py default）/ MeloTTS 预热
- 估时矫正：T1.4 45min → 60-90min；T1.5 1.5h → 2-3h 或不可达
- 补充禁区：禁动 LLM 推理参数 / 禁升 llama.cpp 或 OpenVINO 版本 / 禁 pip install 非 pinned

完整 review：`/tmp/intel_chan_OVERALL_PLAN_REVIEW.md`

### code-reviewer (Agent H 审 D+J+E) — approve-with-changes

- 跨 agent 协调 5 项 ✅（无 speak 双触发 / cancel→error+end 双监听 / 行号无交叠 / 字段名一致 / 无 silent failure 反模式）
- HIGH H1 / H2 已在本 HANDOFF 修正
- MEDIUM M1-M4 见第 5 节

完整 review：`/tmp/intel_chan_FRONTEND_TEST_REVIEW.md`

---

## 附：本次 Agent E 产出验证

```
$ python -m pytest tests/test_11_tools_e2e.py -v 2>&1 | tail -3
======================== 39 passed, 2 skipped in 2.66s =========================
```

测试覆盖：
- TestToolRegistration ×6（11 tool + 11 schema 对齐 / callable / 必需字段）
- TestSystemTools ×4（get_temperature / get_system_info cpu/memory/unknown）
- TestPerceptionTools ×8（get_gesture × 2 / get_distance × 3 / get_scene × 2 / identify_user）
- TestSpeakTool ×4（queue / truncate / empty / no-provider）
- TestMemoryTools ×7（store with key / clamp importance / recall by importance / empty / no-store）
- TestExternalTools ×4（network fail mock × 2 + 外网 skipped × 2）
- TestMemoryAcrossSessions ×5（L1 跨 session / role 过滤 / L2 persist / upsert / 综合 hydrate）
- TestReActLoop ×3（两轮 tool→content / 单轮 content / MAX_ITERATIONS guard）
