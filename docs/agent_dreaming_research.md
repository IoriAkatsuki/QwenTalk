# Agent Dreaming — 调研报告

**调研日期**: 2026-05-17
**目标**: 为 Intel DK-2500 边缘 agent (Intel 酱) 实现 idle-time 记忆整合 ("dreaming") 提供事实依据

## TL;DR

行业已经把 "agent dreaming" 从概念变成产品 — Anthropic 2026 年 5 月正式发布了 Claude Dreaming（scheduled 而非 idle-triggered），Letta 把 Stanford Generative Agents 的 reflection 机制工程化为 sleep-time compute（2 个并行 agent，5x 算力压缩）。开源侧 mem0 (55.9k★) 和 Hermes Agent 都已提供完整的 consolidation/retrieval/conflict 工具链。对 Intel 酱这种 24GB UMA 边缘场景，最务实路线是: **借用 Generative Agents 的 importance-threshold 触发 + Hermes Agent 的 MEMORY.md/USER.md 双文件结构 + mem0 的 ADD-only append 策略**，把 dreaming 限制为每天 1-2 次、单次 ≤30 LLM calls 的离线 job。

## 1. Anthropic 官方资料

- **Claude Dreaming**（2026-05 在 Code with Claude 大会发布，目前 research preview，仅限 Managed Agents 客户）。**触发是 scheduled cron，不是 idle detection** — 官方说法 "nightly, hourly, or whatever interval makes sense"，因为 idle 期间还可能有别的 agent 跑。
- 整合内容: 用户偏好、沟通风格、recurring patterns、task outcomes、对先前 belief 的修正
- 关键定位: "compression and judgment" — 官方明确不保证 perfect preservation
- 在 Claude 产品侧：2025-08 先在 Team/Enterprise 上线 Memory，2025-10 扩到 Pro/Max，2026-03 扩到 Free tier — **是 retrieval-based persistent memory，不是 dreaming**
- 公开 research blog 没有专门的 "dreaming" paper；最近的相关方向是 Constitutional AI 和 Sleeper Agents (这两个跟 dreaming 无关，是 alignment topic)

## 2. Hermes / Nous Research

- **Hermes 3 技术报告** (arxiv 2408.11857) 训练了 `<REFLECTION>` `<INNER_MONOLOGUE>` `<SCRATCHPAD>` 等专用 token — model-level 的 reflection 能力但**不是 dreaming**
- **Hermes Agent** (github.com/nousresearch/hermes-agent) 是真正可用的 agent 框架，提供:
  - `MEMORY.md` (2.2KB agent 自身笔记) + `USER.md` (1.4KB 用户偏好) 双文件 frozen snapshot 注入 prompt
  - 工具: `hindsight_retain` / `hindsight_recall` / `hindsight_reflect` (cross-memory synthesis)
  - **没有定时 dreaming**，是 proactive curation + 容量到顶时主动 consolidate
  - "Periodic nudges" + "autonomous skill creation after complex tasks" 在文档里被提到但实现未公开
- 用户项目里 `hermes_qwentalk_config.example.yaml` 应该指向 Hermes Agent 的 MCP 集成

## 3. OpenClaude / 开源 clone 项目

不存在严格意义的 Claude 开源 clone，但 Claude Code 周边的 memory 项目很活跃:

- **claude-mem** (thedotmack) — session 压缩 + 未来 session 注入 (跨 Claude Code/Codex/Gemini/Hermes)
- **claude-memory-compiler** (coleam00) — Claude Agent SDK hooks 捕获 session，LLM 编译成结构化 articles，参考 Karpathy LLM Knowledge Base
- **ClawMem** (yoloshii) — on-device 本地 RAG memory 层，纯本地无 API key，**最贴近 Intel 酱这种离线边缘场景**

## 4. GitHub 高 star 项目

| Repo | Star | 核心机制 | 与本项目相关性 |
|---|---|---|---|
| [mem0ai/mem0](https://github.com/mem0ai/mem0) | 55.9k | ADD-only append + LLM 抽取 atomic facts + hybrid (vector+graph+KV) + 多信号检索 (semantic/BM25/entity/temporal) | ★★★★★ 直接可用，但默认要 OpenAI embedding，需替换本地 Qwen 600M |
| [cpacker/MemGPT → Letta](https://github.com/letta-ai/letta) | ~16k+ | OS-style memory paging + sleep-time agent (主+sleep 双 agent) + 异步 memory mgmt | ★★★★ 架构参考价值高，但运行开销不适合 24GB UMA |
| [noahshinn/reflexion](https://github.com/noahshinn/reflexion) | ~3k | verbal RL: 把 binary feedback 转成 textual summary 进 episodic buffer | ★★★ 适合 skill 改进，不是日常 memory |
| [NousResearch/hermes-agent](https://github.com/nousresearch/hermes-agent) | growing | MEMORY.md+USER.md frozen snapshot + hindsight tool 三件套 | ★★★★★ 本项目已对接 |
| [letta-ai/sleep-time-compute](https://github.com/letta-ai/sleep-time-compute) | - | paper 配套 code，Stateful GSM-Symbolic / Stateful AIME 数据集 | ★★★ 学术参考 |

## 5. 学术 / 神经科学借鉴

- **Generative Agents (Park et al., arXiv 2304.03442, UIST 2023)** — 25 个 NPC 小镇 Smallville。**reflection 触发机制**: 最近 events 的 importance 累计分数 > 150 时触发 → 平均每天 2-3 次。importance 由 LLM 打 1-10 分（"刷牙=2","表白=8"）。reflection 操作最近 100 条记忆，三步: 生成候选问题 → 检索相关 memory → LLM 抽出 5 条 high-level insights + evidence citation。形成 tree-of-reflection 层级
- **Reflexion (Shinn et al., NeurIPS 2023, arXiv 2303.11366)** — 不更新权重，靠 verbal feedback 进 episodic buffer；HumanEval 91% pass@1
- **Sleep-time Compute (Lin et al., arXiv 2504.13171, Letta+UC Berkeley 2025)** — 把 raw context 转 learned context，5x test-time compute 压缩 / 2.5x cost 降低；前提是 "predictability of user query" 高
- **神经科学映射**: REM 睡眠期 hippocampus replay → cortex 形成 semantic memory。对 agent 的隐喻: episodic (L1 SQLite) → semantic (L2 facts) 的转换 + 冗余剪枝 + 矛盾检测。Dreaming 的标签本身就是 Anthropic 引用的这个类比

## 6. 设计维度回答（6 个问题）

1. **Trigger**: Anthropic 用 cron schedule（最稳）；Generative Agents 用 importance-threshold（最优雅但每天 2-3 次开销大）；Letta 是主+sleep 双 agent 异步常驻。**边缘设备推荐 cron + idle gate**: cron 每 N 小时一次，跑前检查最近 ≥30 min 无前端输入再启动
2. **Memory 选择**: Generative Agents 的 importance score (LLM 打 1-10) 是行业标准；mem0 用 LLM 抽 atomic facts + entity linking；Hermes 明确黑名单 (trivial / re-discoverable / 原始 code dump / 会话临时)
3. **冲突检测**: mem0 **采用 ADD-only 不覆盖**，靠 temporal reasoning 检索最新；早期 mem0/Letta 用 UPDATE/DELETE；Anthropic 承认 "no guarantee of perfect preservation"。**推荐策略**: 冲突先 ADD 双条带时间戳 → dreaming 中由 LLM 仲裁生成第三条 reconciliation memory
4. **整合形式**: Hermes 用 markdown 文件 frozen snapshot；mem0 用 atomic facts + embedding；Generative Agents 用自然语言 reflection tree。**最省 token**: markdown + 关键 fact 提取，**最强检索**: embedding + entity graph
5. **资源消耗**: sleep-time compute 论文显示能 5x 节省 test-time；但 dreaming 本身一晚要跑数十次 LLM call。Generative Agents 每次 reflection 大约 5-10 calls × 每天 2-3 次 = 15-30 calls/day。**Qwen3.6-A3B @ 8 t/s 跑 30 calls × 500 token = 30 min 离线 job**，可行
6. **产品案例**: Anthropic Claude Dreaming (managed agent, 2026-05)、Letta sleep-time agents (生产)、Replika 2.0 (2026-04 重构 memory)、Character.AI 无持久 memory、Notion AI 走 retrieval 不是 consolidation、Pi (Inflection) 闭源未披露

## 7. 推荐 Intel 酱项目 dreaming 方案

**Trigger**: cron 每天 03:00 + idle gate (近 30 min 无 ASR 输入 + 摄像头无人脸)，避免影响 wake-word 响应；额外保留 importance threshold 副触发器（连续高情感事件累计 > 阈值时立即整合）

**Memory 选择**: 沿用 Generative Agents 1-10 importance 打分（用 Qwen3.6-A3B 跑 batched scoring），≥6 分进 L2 facts；Hermes 黑名单过滤 trivial/可重发现的

**冲突检测**: 先 ADD-only 不覆盖（mem0 思路），dreaming 中用 LLM diff 两条带时间戳的矛盾 memory，生成 reconciliation 第三条；user 明确否定时打 invalidated tag 而不删除

**整合形式**: Hermes 双文件结构 (MEMORY.md / USER.md) + atomic facts SQLite 嵌入，markdown 进 prompt，facts 走 RAG

**资源消耗**: 单次 dreaming budget ≤30 LLM call ≤500 token avg ≤30 min wall time；用 Qwen3.6-A3B Q4_K_S mixed mode 跑 (实测 8.37 t/s)；PC801 NVMe 已存 L4 KV cache，可复用做 reflection 中间结果落盘

## Sources

- [Anthropic Claude Dreaming - MindStudio](https://www.mindstudio.ai/blog/what-is-claude-dreaming-anthropic-managed-agents) — Anthropic 2026-05 发布的 scheduled dreaming，目前 research preview
- [VentureBeat: Anthropic dreaming](https://venturebeat.com/technology/anthropic-introduces-dreaming-a-system-that-lets-ai-agents-learn-from-their-own-mistakes) — 第三方报道
- [Storyboard18: Anthropic Dreams feature](https://www.storyboard18.com/digital/anthropic-introduces-dreams-feature-for-claude-to-reorganise-memory-and-improve-ai-agents-97376.htm) — 印度媒体 confirm
- [Anthropic Claude Memory rollout](https://www.macrumors.com/2025/10/23/anthropic-automatic-memory-claude/) — 2025-10 Memory 扩到 Pro/Max
- [Sleep-time Compute arXiv 2504.13171](https://arxiv.org/abs/2504.13171) — Letta+Berkeley 2025 论文
- [Letta Blog: Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute) — 工程化实现：主 agent + sleep agent
- [letta-ai/sleep-time-compute github](https://github.com/letta-ai/sleep-time-compute) — 论文配套代码
- [Letta MemGPT blog](https://www.letta.com/blog/memgpt-and-letta) — MemGPT 已并入 Letta
- [Generative Agents arXiv 2304.03442](https://arxiv.org/abs/2304.03442) — Park et al. Smallville
- [Generative Agents ar5iv full text](https://ar5iv.labs.arxiv.org/html/2304.03442) — importance=150 阈值 + 1-10 评分细节
- [Reflexion arXiv 2303.11366](https://arxiv.org/abs/2303.11366) — Noah Shinn verbal RL
- [mem0ai/mem0 github](https://github.com/mem0ai/mem0) — 55.9k★, ADD-only memory layer
- [InfoWorld: Mem0 deep dive](https://www.infoworld.com/article/4026560/mem0-an-open-source-memory-layer-for-llm-applications-and-ai-agents.html) — 架构和 benchmark
- [Hermes Agent docs - Persistent Memory](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory) — MEMORY.md/USER.md 设计
- [Hermes 3 Technical Report arXiv 2408.11857](https://arxiv.org/pdf/2408.11857) — `<REFLECTION>` token training
- [Hermes 4 site](https://hermes4.nousresearch.com/) — 2026 hybrid reasoning model
- [claude-mem github](https://github.com/thedotmack/claude-mem) — 跨 agent 通用 session memory
- [claude-memory-compiler github](https://github.com/coleam00/claude-memory-compiler) — Karpathy LLM Knowledge Base 风格
- [ClawMem github](https://github.com/yoloshii/ClawMem) — on-device 离线 memory，最贴近边缘场景
- [Replika 2.0 architecture](https://www.roborhythms.com/replika-2-0-explained/) — 情感模式而非事实 memory
- [Analytics Vidhya: AI Agent Memory Systems](https://www.analyticsvidhya.com/blog/2026/04/memory-systems-in-ai-agents/) — semantic consolidation / intelligent forgetting / conflict resolution 综述
