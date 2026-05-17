# 滚动 KV 落盘 — 调研报告

> 调研对象：production / 学术上的 sliding-window KV checkpoint streaming 设计
> 调研目的：为 Intel DK-2500 上 hierarchical KV cache（Xe-LPG UMA hot / PC801 USB cold）实现提供事实依据
> 日期：2026-05-17

## TL;DR

1. **真正的"滚动 KV 落盘"行业标准答案有三套**：(a) ds4 用 **session 边界 + 间隔 token 触发**（cold/continued/evict/shutdown 四时机），(b) llama.cpp 服务端用 `--checkpoint-every-n-tokens` 在 **prefill 期间** 自动每 N tokens 写 RAM checkpoint（默认 8192，没原生写 disk），(c) **KVSwap (arXiv 2511.11907, 2025)** 是目前最贴本场景的工作 —— 边缘设备 layer-by-layer 异步写 NVMe/eMMC + top-k 预加载，恰好命中"Xe-LPG UMA + PC801 USB"架构。
2. **触发主流是双闸门**：阈值（≥ N tokens 或对齐 chunk 边界）+ 事件（session evict / prefill 完成）。粒度上 ds4/llama.cpp 是**整 KV state per checkpoint**，KVSwap/LMCache 是 **chunk 级（4-256 tokens）**；layer-by-layer 异步流水是边缘场景标配。
3. **对 Intel 酱**：先抄 llama.cpp 的 `--checkpoint-every-n-tokens 8192 --slot-save-path /mnt/pc801` 拿到 80% 收益，再借鉴 ds4 的 `cold/continued/evict/shutdown` 四时机和 SHA1-prefix 键设计跨 session 复用，**滚动 page-out attention 内部 hook 留到 Phase 2**（KVSwap 路线，需 fork llama.cpp）。

---

## 1. ds4 实现细节（antirez/ds4, 10.2k★, 2026-05）

### 1.1 设计哲学

> "The KV cache is actually a first-class disk citizen." — ds4 README

不再把 KV 当 RAM-only 临时态。在 128GB Mac 上模型本身 ~76GB，把 KV 写 SSD 才能撑近 1M tokens 上下文。

### 1.2 关键源码（`ds4_server.c`，15581 行单文件）

**触发时机（4 个 reason 枚举）**, `ds4_server.c:8227-8231`：
```c
KV_REASON_COLD      = 1,   // 初始 prompt 处理稳定后
KV_REASON_CONTINUED = 2,   // 长对话每 10k tokens 对齐边界
KV_REASON_EVICT     = 3,   // 新 session 替换 live cache 前
KV_REASON_SHUTDOWN  = 4,   // 服务器干净退出
```

**默认参数**（`ds4_server.c:8197-8208`）：
```c
KV_CACHE_DEFAULT_MIN_TOKENS               = 512      // 不存太短的
KV_CACHE_DEFAULT_COLD_MAX_TOKENS          = 30000    // cold 上限
KV_CACHE_DEFAULT_CONTINUED_INTERVAL_TOKENS= 10000    // continued 间隔
KV_CACHE_DEFAULT_BOUNDARY_TRIM_TOKENS     = 32       // 修剪尾部 token
KV_CACHE_DEFAULT_BOUNDARY_ALIGN_TOKENS    = 2048     // 对齐 prefill chunk 边界
KV_CACHE_DEFAULT_MB                       = 4096     // 默认磁盘预算 4 GiB
```

**触发位置**注释 `ds4_server.c:8156-8169`：
> "We persist reusable DS4 session snapshots **when a cold prompt reaches a useful prefix, when a long continued conversation has grown far enough, and when a request evicts the live session**. The cache key is the SHA1 of the rendered byte prefix."

**关键设计 trade-off**：
- **Live KV cache 只在 RAM 中保留一个**（"only one live KV cache in memory"），多 session 通过 disk 复活
- **使用 read/write IO 而非 mmap**（`ds4_server.c:8172-8174`），避免给已经 mmap 大 GGUF 的进程再加 VM mapping
- **同步写**，但只在 evict / 边界等"非热路径"时刻触发，所以不阻塞 prefill/decode
- **不允许往回卷 session** 来制造 cache：`"We never roll the session backward just to build a disk cache entry"` (`ds4_server.c:8170`)

**结论**：ds4 是 **session-level 异步 + chunk-aligned checkpoint**，不是 inference 内 page-out。这是因为 ds4 跑的 DeepSeek V4 的 KV cache 已经被 MLA 压得很小，分 layer 滚动收益不大；它的"first-class disk"是给**跨 session 持久化**用的。

---

## 2. llama.cpp 原生 ctx-checkpoint 行为

### 2.1 三个 flag 真实含义（PR #15293, 上游已合并）

| Flag | 默认 | 行为 |
|---|---|---|
| `--ctx-checkpoints N` | 32 | 每 slot 保留的 checkpoint **最大数量**（in-RAM 环） |
| `--checkpoint-every-n-tokens N` | 8192 | **prefill 期间**每处理 N tokens 自动建 1 个 checkpoint，`-1` 关 |
| `--slot-save-path PATH` | 关 | 用于 manual `/slots/{id}?action=save` REST 调用 |
| `--cache-ram N` | 8192 MiB | KV cache RAM 总上限 |
| `--swa-full` | false | 给 SWA 模型用完整 KV（不是 sliding 窗口） |

### 2.2 关键认知

- `--checkpoint-every-n-tokens` 是 **自动 + RAM-only**，主要用于 SWA 模型（Gemma 系列）和 prompt-prefix 复用
- `--slot-save-path` 是 **manual 触发**（HTTP `/slots/{id}?action=save`），整 slot KV 写盘 — 这就是 L4 用的
- **没有原生的 "auto-write-to-disk on threshold"** —— issue #20697 `--cache-disk` 仍是 open feature request，无实现
- `--ctx-checkpoints` 是 **滑动窗口在 RAM 中的"环"**，不是 disk

### 2.3 已知 bug（参考 issue #19794, #19977, #20225）

混合架构（Qwen3-Coder-Next、Gemma 4）的 ctx-checkpoint 经常被"invalidated"导致全量重处理。说明 **checkpoint 的失效语义对 SWA / hybrid attention 很脆弱**。

---

## 3. 学术 paper 综述

| Paper | 机制 | 是否 disk-backed | 备注 |
|---|---|---|---|
| **StreamingLLM** (Xiao et al., MIT, ICLR 2024, arXiv 2309.17453) | 保留前 4 个 attention sink + sliding window，丢弃中间 KV | 否（RAM-only） | 给"丢老 KV 不掉准确率"提供理论：sink token 是 softmax 累积偏置吸收器 |
| **H2O** (Zhang et al., NeurIPS 2023) | 按累积 attention 分数选 heavy hitter | 否 | "丢哪些"的算法，不解决"放哪" |
| **InfLLM** (Xiao et al., 2024) | sink + recent + block-wise top-k 检索 | RAM 内 CPU-offload | block 级 page-in，启发 KVSwap |
| **Quest** (Tang et al., MIT EfficientML 2024, ICML 2024) | query-aware block-wise top-k | 否 | top-k 用 pooled vector，被 KVSwap 评价为"underperform" |
| **⭐ KVSwap** (arXiv 2511.11907, 2025) | **layer-by-layer 写 NVMe/eMMC + 低秩压缩 K cache + top-k 预加载** | **是** | 直接对标本场景 |
| **Tutti** (arXiv 2605.03375, 2026) | SSD-backed KV 长上下文 serving | 是 | 数据中心场景 |
| **KV Cache Offloading I/O 特征** (atlarge 2025-cheops) | NVMe 带宽实测 | 是 | I/O 特征研究 |

### 3.1 KVSwap 深度（最重要）

**arXiv 2511.11907**：标题就是 *"KVSwap: Disk-aware KV Cache Offloading for Long-Context On-device Inference"* — 几乎是项目需求的精确镜像。

| 维度 | KVSwap 做法 |
|---|---|
| **触发** | prefill 期间持续写盘，decode 期间用 ring buffer 累积到 G 个 entry 再批量写 |
| **粒度** | group size G=**4 (NVMe)** 或 G=**8 (eMMC)**；layer-by-layer 写 |
| **同/异步** | **完全异步**："the next layer's I/O performed concurrently with current layer's attention + FFN" |
| **Page-in** | **预测式**：内存里维持 SVD 低秩压缩的 K cache，算近似 attention score，TopK 选 M=400 个最相关 group 预加载 |
| **元数据** | compressed K cache，压缩比 σ = Hk·d / r |
| **目标硬件** | Jetson Orin AGX (UMA 64GB + NVMe 1.8 GB/s) 和 eMMC (250 MB/s) |
| **精度** | RULER 平均损失 ≤4.4%，LongBench 1.1% |
| **加速** | 32K 上下文下比 ShadowKV 快 **1.8× (NVMe) / 4.1× (eMMC)**，vs vLLM 用 **11.0× 更少 KV 内存** |

**关键洞察**：KVSwap 证明在**带宽 250 MB/s** 的 eMMC 上也能跑（PC801 USB 3.0 实测 ~400-500 MB/s 在它和 NVMe 之间），所以"USB 太慢做不了 KV 落盘"这个直觉**是错的** —— 只要 prefetch 预测准 + layer 流水化 hide latency。

---

## 4. GitHub 项目对比

| Repo | ★ | 滚动写盘机制 | 边缘适配 | 备注 |
|---|---|---|---|---|
| **ggml-org/llama.cpp** | 75k+ | 部分（manual `/slots/save` + RAM ctx-checkpoint）；`--cache-disk` 是 open RFC #20697 | ★★★★★ | 项目当前 backend |
| **antirez/ds4** | 10.2k | session-level 4 触发时机 + SHA1 prefix 键 + read/write IO | ★★★ | Mac/CUDA only，无 SYCL；但策略可移植 |
| **LMCache/LMCache** | 数千 | **chunk-level 自动**（默认 `chunk_size=256` tokens），async put / blocking get + prefetch，LRU 驱逐 | ★★ | vLLM 生态，CMU 出品，最接近"production 边缘 KV cache"参考 |
| **vllm-project/vllm** | 50k+ | `swap_space` (CPU RAM)；NVMe via LMCache 或 KV connector | ★ | 数据中心场景，UMA 适配差 |
| **mit-han-lab/streaming-llm** | 6k+ | sliding window + attention sink，RAM-only | ★★★ | 算法基础，不解决落盘 |

### 4.1 LMCache 关键细节

- **粒度**：`chunk_size: 256` tokens（每 chunk 一个 file）
- **触发**：自动（"as they are stored"）
- **I/O**：write **async**，read **blocking**，配 `prefetch()` 主动预热
- **驱逐**：LRU
- **可参考度极高**，但项目 Python+vLLM 整合复杂，移植到 llama.cpp 需重写

### 4.2 llama.cpp `--cache-disk` 现状

Issue #20697（2026-03-17，open）明确指出 **AMD Strix Halo / Intel UMA 设备**需要 disk offload 才能避开和 VRAM 抢内存。**与本项目场景完全一致**。状态：仅 feature request，无 PR。是一个**潜在 upstream PR 机会**。

---

## 5. 五个实施问题回答

### Q1: 触发机制

**推荐两层组合**（ds4 + KVSwap 思路融合）：
- **阈值闸门**：`ctx_pos % 8192 == 0`（对齐 llama.cpp 默认）或 `ctx_pos >= 32768`（UMA pressure 临界）
- **事件闸门**：session evict、prompt 处理完成、对话轮次结束

**不推荐**：纯 UMA 水位监控 —— Xe-LPG UMA 没有明确的 VRAM 边界 watermark API。

### Q2: 粒度

**短期**（L4 已有路径）：整 slot KV state，每 8192 tokens 一份 → 直接用 `--slot-save-path` + cron 调 `/slots/save`。
**中期**：仿 LMCache 的 **chunk 256 tokens**，每 chunk 单 file。
**长期**（KVSwap 路线）：layer-by-layer + group G=4-8 KV entries，但需 fork llama.cpp 改 KV cache layout，工作量极大。

### Q3: 同步 vs 异步

- **写**：必须 **async**（io_uring 或 pthread + ring buffer）。Xe-LPG decode 每 token 125 ms（8 t/s），PC801 USB 写 4MB 也要 ~10ms，**不能阻塞**。
- **读 (page-in)**：**predictive prefetch** 最优（KVSwap 用 top-k 提前 1 layer 预取）；fallback 才是 blocking on-demand。
- **Hide latency**：layer-pipelined I/O 是金标准——下一 layer 的 I/O 与当前 layer 的 attention+FFN 并行。

### Q4: Attention 时 page-in

**预测式优于按需**。
- **简单版**：sink + recent 永驻 RAM（StreamingLLM 模式），只对"远 KV"做磁盘备份，attention 时**永不 page-in**（直接丢，靠 sink 保稳）。
- **进阶版**：维护低秩压缩 K（KVSwap 的 SVD），算近似 score，每 layer top-k 预取 400 group。这能在 LongBench 上把精度损失压到 1.1%。
- **on-demand 取**：仅用作 fallback 兜底，避免阻塞 hot path。

### Q5: 数学保证 / 精度

- **StreamingLLM 已证**：保留前 **4 个 attention sink tokens** 即可让 sliding window attention 在 4M tokens 长度下**不崩**（softmax 累积偏置被 sink 吸收）。
- **KVSwap 实测**：RULER ≤4.4% 损失，LongBench 1.1%。
- **attention sink 是必须的**：朴素丢老 KV（无 sink）会出现 "perplexity 指数爆炸"现象（Xiao et al. 2023 Fig 2）。
- **建议**：永驻 RAM = `sink(4) + recent_window(4096) + sliding_chunk_n`，老 chunk 写 PC801；若需 page-in，用 KVSwap 的低秩近似 top-k 选择。

---

## 6. 推荐 Intel 酱 项目方案

**Phase 1（1 周，零代码）**：直接用 llama-server `--checkpoint-every-n-tokens 8192 --slot-save-path /mnt/pc801/kv --ctx-checkpoints 32 --swa-full`，外加 cron/守护进程在 ctx 达到 24K（UMA 警戒）时调 `POST /slots/0?action=save&filename=auto_$(date).bin` —— 拿到 80% 收益。

**Phase 2（2-3 周）**：在 voice_pipeline 加 "rolling save" 守护：每对话轮结束按 ds4 的 4 时机决策（cold/continued/evict/shutdown），用 SHA1-prefix 命名，LRU evict，磁盘预算 8GB。**这一阶段不改 llama.cpp 源码**，完全在 Python 侧编排。

**Phase 3（可选, 1 个月）**：fork llama.cpp 实 KVSwap-style layer-pipelined async write to PC801 + StreamingLLM sink permanent + LRU sliding window。要改 `src/llama-context.cpp` 的 KV cache update 和 `tools/server/server.cpp`，可贡献为 #20697 的 upstream PR。

**禁忌**：在 SYCL backend 直接做 fattn hook 极不稳定（Xe-LPG graph 已知 mul_mat_id 兼容性坑，参见项目 memory `sycl_graph_disabled_finding.md`）；先做 Python/server 层编排再下沉。

---

## Sources

### ds4
- [antirez/ds4 GitHub](https://github.com/antirez/ds4) — `ds4_server.c` 15581 lines, KV cache impl at L7540-L8950
- [ds4 README — KV-cache section](https://github.com/antirez/ds4/blob/main/README.md)
- [ds4 AGENT.md](https://github.com/antirez/ds4/blob/main/AGENT.md)
- [ds4 commentary — Pasquale Pillitteri](https://pasqualepillitteri.it/en/news/2253/ds4-antirez-deepseek-v4-flash-inference-engine)

### llama.cpp
- [llama-server README master](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [Feature Request #20697 `--cache-disk`](https://github.com/ggml-org/llama.cpp/issues/20697)
- [Bug #19794 ctx-checkpoint invalidated (Qwen3-Coder)](https://github.com/ggml-org/llama.cpp/issues/19794)
- [Bug #19977 Qwen3.5-122B context cache loss](https://github.com/ggml-org/llama.cpp/issues/19977)
- [Bug #20225 Qwen 3.5 full re-process](https://github.com/ggml-org/llama.cpp/issues/20225)
- [Bug #21133 mmproj blocks slot save/restore](https://github.com/ggml-org/llama.cpp/issues/21133)

### 学术
- [Xiao et al. 2023 StreamingLLM arXiv 2309.17453](https://arxiv.org/abs/2309.17453)
- [⭐ KVSwap arXiv 2511.11907 (2025)](https://arxiv.org/html/2511.11907v1)
- [Tutti SSD-backed KV serving arXiv 2605.03375](https://arxiv.org/html/2605.03375)
- [NVMe KV Cache I/O Characterizing Study (atlarge 2025)](https://atlarge-research.com/pdfs/2025-cheops-llm.pdf)
- [SnapStream — sliding window + global top-K](https://www.emergentmind.com/topics/streamingllm)

### GitHub 项目
- [LMCache/LMCache GitHub](https://github.com/LMCache/LMCache)
- [LMCache local-storage docs](https://docs.lmcache.ai/kv_cache/local_storage.html)
- [LMCache tech report PDF](https://lmcache.ai/tech_report.pdf)
- [vLLM forum: nvme KV offload discussion](https://discuss.vllm.ai/t/possible-to-offload-kv-cache-to-dram-or-nvme/1682)
- [vLLM RFC #19854 KV cache offloading](https://github.com/vllm-project/vllm/issues/19854)
- [vLLM optimization & swap_space docs](https://docs.vllm.ai/en/stable/configuration/optimization/)
- [mit-han-lab/streaming-llm GitHub](https://github.com/mit-han-lab/streaming-llm)
- [Ceph + vLLM + LMCache integration blog](https://ceph.io/en/news/blog/2025/vllm-kv-caching/)
- [GKE Tiered KV Cache blog (Google Cloud)](https://cloud.google.com/blog/topics/developers-practitioners/boosting-llm-performance-with-tiered-kv-cache-on-google-kubernetes-engine/)
