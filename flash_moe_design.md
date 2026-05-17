# Flash-MoE 风格 Expert 顺序维护方案 — Intel DK-2500 + PC801 USB

**Target**: Qwen3.6-35B-A3B (Q4_K_S 20GB, 41 layers, 256 routed experts, top-8/token)
**Baseline**: 8.06 t/s bandwidth-bound (UMA 52 GB/s 已饱和)
**Storage**: PC801 USB 3.2 Gen 2x1, 960 MB/s seq read (4M block)

---

## 1. 调研结论

### flash-moe (danveloper) 核心机制
- **目标差异**: 397B 总参 ÷ 48GB MacBook = **5.5GB 非专家常驻 + 209GB 专家流式 SSD**
- **关键哲学** "Trust the OS": 不做自定义 cache，依赖 OS page cache LRU (~71% 自然命中率)
- **加载**: 仅 K=4 active experts 通过 **parallel `pread()` + GCD dispatch group** 按需读取 (每个 ~6.75MB)
- **抛弃的优化**:
  - `F_RDADVISE` 预取 net 0% (统一内存下 SSD DMA 拖累 GPU -73%)
  - Temporal expert prediction 25% 命中 → 性能 -18%
  - MLP routing predictor 31% 命中 → 仍劣于 baseline
- **`F_NOCACHE`** 仅在 2-bit 模式保留 (+3%, 避免 page thrash)
- **结论**: 串行 pipeline (GPU → SSD → GPU) 反而硬件最优

### ds4 项目
**无法确认是哪个项目**。搜索 "ds4 moe" 未发现直接对应 repo。最可能是用户对 **DeepSeek-V3/V4** 的简称 (DeepSeek 是 671B-A37B MoE，256 routed + 1 shared, top-8 — **与 Qwen3.6 完全同构**)。DeepSeek-V3 的相关性：
- **Auxiliary-loss-free 负载均衡**: 通过偏置项维持 expert 调用分布趋于均匀 → 意味着 **runtime 频次分布更平坦**，hot-expert 缓存收益上限受限
- 这恰好印证 flash-moe 抛弃 prediction 的实证

如用户原意指其他项目，请明示。

### Qwen3.6 MoE 结构（本机 gguf_dump 确认）
- 架构: `qwen35moe`, 41 blocks, embed 2048, ctx 262144
- Experts: `expert_count=256`, `expert_used_count=8`, `expert_ff_len=512`
- Tensor 命名规则: `blk.{0..40}.ffn_{gate,up,down}_exps.weight` (Q4_K, 256×512×2048 = 268M elements 单 tensor 含 256 experts)
- **关键洞察**: GGUF 已把 256 experts 打包成 **单一 3D tensor** 而非 256 独立 tensor — `--override-tensor` 正则只能整层粒度控制，**单 expert 粒度需改 llama.cpp graph**

---

## 2. 板卡现状定位

| 项 | 值 | 含义 |
|---|---|---|
| RAM 总量 | 24GB UMA | 模型 20GB + KV + Python = 紧 |
| 当前 used | 21GB / free 0.9GB | 已贴顶 |
| buff/cache | 16GB | mmap 命中良好 |
| swap used | 5.1GB | **已在用 swap → 任何 IO 优化都先要释放 swap** |
| llmsrv | `-ngl 99 -fa 1 -ub 64 -t 2` 全 GPU | 当前未用 `-ot` offload |
| 模型路径 | `/home/intel/models/qwen3.6-mtp-q4ks.gguf` 20GB on PC801 | seq read 960 MB/s |

**关键判断**: 当前是 **fit-UMA 全 GPU 路径**，bandwidth-bound 来自 **iGPU 显存子系统 (52 GB/s)**，**不是 SSD IO**。flash-moe 的核心收益是"模型 > RAM 时的 SSD 流式"，本机不直接适用。

但 **三个二阶机会** 值得拿：
1. **swap 替代**: 当前 5.1GB swap = OS 已被迫做 LRU。把 cold expert 主动放回 mmap 让 OS 看清楚边界，比 swap 盲驱逐好
2. **bigger context**: 25 GB swap-out → 让 21 GB 模型部分 mmap 化，可腾 4-6GB 给 ctx/agent stack
3. **future-proof**: 一旦上 Q5_K_M 25GB 或 Q6_K 28GB（超 UMA），flash-moe 路径 = 唯一选择

---

## 3. 设计方案

### Stage 0: 测量 — 不动 llmsrv, 旁路观察 (本周)

**A. Expert tensor 加载流量 baseline**（本次已测）
```
# ssh intel@192.168.1.8 'iostat -xm 1 3 sda'
# Sample 2-3 稳态:
# sda  r/s=0.00  rMB/s=0.00  %util=0.00
```
**实测确认: steady-state IO = 0**。Profile A 全 GPU 路径，模型完全驻 UMA，
当前 8.06 t/s **不是 SSD bandwidth bound**，flash-moe 流式改造无即时性能收益。
真正价值: **释放 RAM** 给 ctx/agent，**或** 上 Q5_K_M 25GB 后超 UMA 时的唯一可行路径。

**B. 真正 expert frequency 跟踪** (旁路推理)
llama.cpp **`--lookup-cache-static`** 不暴露 expert id, 但 `llama-server` 在 `verbose` 下 log 中 grep `gate_inp` / `top-k` 不可靠。**唯一干净接口**:
- 用 `llama-batched-bench` 或自写小程序，在 build 时 `#define GGML_DEBUG_MOE 1` patch `llama_set_expert_logging` (位于 `src/llama-graph.cpp` 的 `build_moe_ffn`) 打印 selected expert ids
- 或者在不改源码情况下，用 `perf record -e cache-misses` + symbol 推断 hot layer

**建议**: 跳过直接频次跟踪 (flash-moe 已验证 25-31% 命中率收益负向)，**改用 layer-level 静态策略**

### Stage 1: 配置生成器 + dry-run (无侵入)

把决策固化为 `--override-tensor` 正则生成器，决策维度:
- **Layer 范围**: Qwen3.6 41 层，attention/embedding 必须 GPU
- **共享 expert** (`ffn_shared_*`): 每 token 必激活 → GPU 必驻
- **routed expert** (`ffn_*_exps`): 256 中 8 激活 = 3.1% sparsity, **候选 offload 主体**

**配置策略 3 档**:

| Profile | -ot 正则 | 预期 GPU 占用 | 适用 |
|---|---|---|---|
| A. fit-UMA (当前) | (无) | 20GB | 已 OK 但贴顶 |
| B. spillover-late | `blk\.(3[0-9]\|40)\.ffn_.*_exps\.weight=CPU` | -2.4GB GPU | 释放 ctx/agent RAM |
| C. flash-stream | `blk\.([0-9]\|[12][0-9]\|3[0-5])\.ffn_.*_exps\.weight=CPU` (留前 5 层 GPU) | -16GB | 上 Q5/Q6 必用 |

**注意**: Q4_K_S 单层 ffn_*_exps 三个 tensor ≈ 3 × 67MB = **201MB/层 × 41 = 8.2GB pure experts** (剩 12GB 是 attn/embed/共享 expert/normalize)

### Stage 2: 编排器 (生成 launch script，不改 production)

产出 `flash_moe_launch.sh`:
1. 检测 `/proc/meminfo` available
2. 选择 profile (A/B/C)
3. 输出建议的 llama-server 启动参数
4. **不自动重启 llmsrv** — 让用户手动 `systemctl restart` 决定时机

### Stage 3: 验证矩阵

| 实验 | 命令 | 关注指标 |
|---|---|---|
| E1 | Profile A baseline | t/s (cold/hot), free, swap |
| E2 | Profile B drop_caches + iostat | t/s 下降幅度, sda r/s |
| E3 | Profile C 同上 | 同 + GPU util |
| E4 | E2 + `madvise(MADV_RANDOM)` on expert region | 比 E2 的 page-fault tail latency |

**成功标准**:
- Profile B: t/s ≥ 7.5 (-7% within tolerance), free RAM ≥ 4GB → 立项可用
- Profile C: t/s ≥ 4.0 → 留作 Q5_K_M fallback
- 任一: SSD r/s 稳态 < 200 MB/s (远低于 960 MB/s 上限, 不打满 USB)

---

## 4. 与 flash-moe 哲学的对齐与背离

| flash-moe | 本方案 | 理由 |
|---|---|---|
| Trust OS page cache | 同 | 无优势的自定义 cache |
| 无 prediction | 同 | DeepSeek auxiliary-loss-free → 频次平坦 |
| `pread()` 并行 | **N/A** | llama.cpp `mmap` 已隐式并行 page fault |
| 流式 209GB / 48GB RAM | **不适用** | 本机 20GB ≤ 24GB RAM，无需流式 |
| Metal serial pipeline | **SYCL graph** | 平台差异，graph 已默认开启 |

**核心差异**: 本机不是 flash-moe 主战场（model fits），收益主要在 **释放 RAM 给 agent stack** 和 **未来 Q5/Q6 升级**。

---

## 5. 实施清单

- [x] 调研 (本文档)
- [ ] `flash_moe_launch.sh` — 启动配置生成器 (无侵入)
- [ ] `moe_expert_tracker.py` — log 解析器 (仅在 patch 后有用，可选)
- [ ] 板卡冷启动验证 Profile B (需用户授权重启 llmsrv)
- [ ] 写入 memory: 结果落 MEMORY.md

---

## 6. 风险

- 当前 swap 5.1GB → **任何 drop_caches 实验前先 `swapoff -a && swapon -a`** 或重启
- llmsrv `-t 2 -ub 64` 是已调过的最优, **不要在测试时改 t/ub**, 否则 t/s 变化无法归因
- `--override-tensor` 在 SYCL 后端可能 fallback CPU 路径慢 — 需先小 batch 验证不挂掉
- 不要 push: 本机分支 `QwenTalk` 已 10 commits ahead

## 7. Next step

执行顺序: 写 `flash_moe_launch.sh` → 板卡 dry-run (Profile A baseline 重测) → 若用户授权再 Profile B 重启验证。
