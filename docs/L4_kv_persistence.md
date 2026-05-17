# L4 KV Cache Persistence Layer 设计

> 状态：**设计 + 原型**（不端到端集成；等 turbo3 SET_ROWS stage 5 完成后再接 production）
> 灵感来源：[antirez/ds4](https://github.com/antirez/ds4) — "The KV cache is actually a first-class disk citizen"
> 平台：Intel DK-2500，PC801 NVMe via USB 3.2 Gen 2x1，板卡上挂在 `/home`（848 GB 可用）

---

## 1. 背景与目标

### 1.1 痛点
- `Qwen3.6-A3B-Q4_K_S` ctx 32K 时，**cold prefill ≈ 16 min**（32 t/s prompt 处理速度）
- 板卡 RAM 24 GB UMA，重启 / OOM / 切 session 都会丢 KV cache
- L1/L2 记忆只存语义文本，**不存 token-level KV**，重启后必须重新 prefill

### 1.2 目标
让 KV cache 像 ds4 那样成为"一等公民磁盘资源"：
1. session end / ctx 接近上限 → **自动落盘 PC801**
2. session resume / 用户 `/load` → **直接热启动**，跳过 prefill
3. 与现有 L1/L2 SQLite memory 正交，不互相污染

### 1.3 非目标
- 不做跨模型迁移（KV 与模型 + quant 强耦合）
- 不做加密（板卡 single-user，本地盘）
- 不做分布式同步

---

## 2. 四层 Memory 架构

```
┌──────────────────────────────────────────────────────────────┐
│  L0  in-context KV (UMA, ≤ 32K turbo3 / ≤ 4K current)        │  ← llama-server slot
│      hot, ms-level access, 4-5 GB f16 / 0.3 GB q8_0          │
├──────────────────────────────────────────────────────────────┤
│  L1  short-term dialog (24h SQLite + 词法检索)               │  ← edge_memory.py
│      文本形式，半衰期 24h，重启不丢                          │
├──────────────────────────────────────────────────────────────┤
│  L2  persona facts (long_term SQLite k-v)                    │  ← edge_memory.py
│      "用户喜欢晚上工作"，永久                                │
├──────────────────────────────────────────────────────────────┤
│  L3  对话语料归档（预留）                                    │  ← 未来 RAG 索引
│      整段对话文本 + embedding，跨周月查询                    │
├──────────────────────────────────────────────────────────────┤
│  L4  KV cache 持久化（本设计）                               │  ← kv_persistence.py
│      .kvbin 文件 in PC801，session_id 索引，秒级 load        │
└──────────────────────────────────────────────────────────────┘
```

**层间关系**：
- L4 是 L0 的 **冷备份**，写时机 = session 结束或 ctx 超水位
- L4 与 L1/L2 **正交**：L1/L2 是"语义记忆"（人能读的文本），L4 是"神经记忆"（模型可直接消费的 KV state）
- L4 失败时 fallback 到 L1 RAG 重塞 prompt（见 §6）

---

## 3. llama-server 原生 slot save/restore（关键调研）

### 3.1 调研结论：**支持** ✓

板卡 SSH `intel@192.168.1.8`，二进制 `/home/intel/llama_mtp/build_sycl/bin/llama-server`（mtp-pr fork），`--help` 输出（已 source oneAPI）：

| 行号 | Flag | 含义 |
|---|---|---|
| 517 | `--slot-save-path PATH` | 持久化 slot KV 的目录（默认 disabled） |
| 400 | `-ctxcp / --ctx-checkpoints N` | 每个 slot 最多保留多少 ctx checkpoint（默认 32，PR [#15293](https://github.com/ggml-org/llama.cpp/pull/15293)） |
| 405 | `-cpent / --checkpoint-every-n-tokens N` | prefill 阶段每 N tokens 自动 checkpoint（默认 8192） |
| 414 | `--cache-idle-slots` | idle slot 自动 save+clear，依赖 `--cache-ram` |
| 507 | `--cache-reuse N` | KV-shifting 复用，已经在跑 |
| 515 | `--slots` endpoint | 监控/save/restore HTTP API（默认 enabled） |

### 3.2 当前 llmsrv 进程状态
```
16407 /home/intel/llama_mtp/build_sycl/bin/llama-server \
  -m /home/intel/models/qwen3.6-mtp-q4ks.gguf \
  -ngl 99 -fa 1 -ub 64 -t 2 -ctk f16 -ctv f16 \
  --port 8080 --host 0.0.0.0
```
**未加 `--slot-save-path`**。直接调 `POST /slots/0?action=save` 返回：
```json
{"error":{"code":501,"message":"This server does not support slots action.
                                Start it with `--slot-save-path`"}}
```

### 3.3 HTTP API
llama.cpp 标准 endpoint（参考 [server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#api-endpoints)）：

```
POST /slots/{slot_id}?action=save
  body: {"filename": "session_xxx.kvbin"}
  → 写到 {slot_save_path}/session_xxx.kvbin

POST /slots/{slot_id}?action=restore
  body: {"filename": "session_xxx.kvbin"}
  → 从盘加载，slot 立即 hot

POST /slots/{slot_id}?action=erase
GET  /slots
  → 列出所有 slot 状态
```

### 3.4 部署侧改动（设计层，不动 production）
重启 llmsrv 时追加：
```bash
--slot-save-path /home/intel/kv_sessions/ \
--ctx-checkpoints 16 \
--checkpoint-every-n-tokens 4096
```

---

## 4. KV 文件格式

llama.cpp 自己写的格式是不透明 binary（`llama_state_seq_save_file`），我们不解析它，只**包一层 metadata header**便于 GC 与跨模型校验：

```
{session_id}.meta.json   ← 我们写
  {
    "session_id": "intel-chan-20260517-1234",
    "model_hash": "sha1(qwen3.6-mtp-q4ks.gguf 头 1MB)",
    "ctx_used": 18432,
    "n_ctx": 32768,
    "created_at": 1779000000.0,
    "size_bytes": 314572800,
    "tags": ["voice", "live2d"],
    "user_summary": "继续昨天的板卡 KV 优化讨论"
  }

{session_id}.kvbin       ← llama-server 写
  llama.cpp 原生 sequence state binary
```

**命名规则**：`session_{slug}_{epoch}.{meta.json|kvbin}`

---

## 5. PC801 容量与路径规划

| 项 | 值 |
|---|---|
| PC801 mount | `/home/intel/` 即 root `/home`（SK Hynix HFS001TEJ9X101N） |
| 可用空间 | 848 GB |
| KV 存储目录 | `/home/intel/kv_sessions/` |
| 单 session 大小（f16 KV，32K ctx） | ≈ 5.4 GB |
| 单 session 大小（q8_0 KV，32K + SWA） | ≈ 0.3 GB（[memory pc801_updated_plan_2026_05_15](memory)） |
| 单 session 大小（turbo3 KV，32K） | ≈ 0.1 GB（推测，stage 5 完成后实测） |
| 容量预算 | 200 GB 上限（留 ≥ 600 GB 给模型/log）→ f16 ~37 session / turbo3 ~2000 session |
| GC 策略 | LRU + 永久标记，session > 30 天且无 pin 删除 |

---

## 6. 风险与回退

| 风险 | 触发条件 | Fallback |
|---|---|---|
| `.kvbin` 文件损坏 / sha 不一致 | 写盘中途崩溃 | 删除文件，走 L1 RAG 重 prefill |
| 模型版本不匹配 | 模型重训 / quant 变更 | `model_hash` 校验失败 → 拒绝 restore，标记 stale |
| ctx 超过 slot n_ctx | restore 后 user 继续聊到 > n_ctx | llama-server 自身 context-shift / kv-shift |
| PC801 USB 掉线 | enclosure 断电 | I/O error → 退化到 L1 RAG，记日志 |
| 多 session 抢同一 slot | 并发 voice + agent | slot_id 池化，session→slot 路由表（L4Store 维护） |
| llmsrv 未启用 `--slot-save-path` | 部署遗漏 | API 返回 501 → 抛 `KVPersistenceUnavailable`，agent 走 L1 |

---

## 7. 性能预算（基于板卡实测 PC801 顺序带宽）

> **实测数据**（2026-05-17，board idle，`dd bs=4M oflag=direct conv=fsync`，10 GB）：
> - 顺序写：**975 MB/s**
> - 顺序读：**2.0 GB/s**（O_DIRECT 单流，USB 3.2 Gen 2x1 satured）
>
> Python 层 `kv_persistence.py` dummy bench (`bench_kv_save_load.py`)：
> - write 100MB → 826 MB/s, 1GB → ~1.0 GB/s (page cache 影响数值偏高，物理墙仍 975 MB/s)


| 配置 | KV 大小 | 写入 (s, @975MB/s) | 读取 (s, @2GB/s) | 相比 cold prefill 加速比 |
|---|---|---|---|---|
| Qwen3.6 f16 KV @ 32K | 5.4 GB | 5.5 | 2.7 | **180×** |
| Qwen3.6 q8_0 + SWA @ 32K | 0.3 GB | 0.31 | 0.15 | **3200×** |
| turbo3 @ 32K（预期）| 0.1 GB | 0.10 | 0.05 | **10000×** |
| turbo3 @ 256K（预期）| 0.75 GB | 0.77 | 0.38 | **8000×+** |

> Cold prefill 锚点：32K @ 32 t/s prompt = 1000 s ≈ 16 min。
> 写盘比 prefill 快 100×～1000× → 显著工程收益。

---

## 8. Python 接口（详见 `kv_persistence.py`）

```python
from kv_persistence import KVStore

store = KVStore(base_dir="/home/intel/kv_sessions",
                llm_url="http://127.0.0.1:8080",
                model_path="/home/intel/models/qwen3.6-mtp-q4ks.gguf")

# 写
store.save_session("intel-chan-20260517-1234", slot_id=0,
                   tags=["voice"], user_summary="板卡 KV 优化")

# 读
ok = store.load_session("intel-chan-20260517-1234", slot_id=0)

# 列出
for s in store.list_sessions(): print(s["session_id"], s["ctx_used"])

# GC
store.gc_old_sessions(max_age_days=30, max_total_gb=200)
```

**离线 dummy 模式**：`KVStore(llm_url=None)` 跳过 HTTP，只做 metadata + dummy bytes 写盘（用于本地基准测试）。

---

## 9. 与 agent.py 集成（设计层，未实现）

```
voice_pipeline 收到 "继续昨天那个项目"
    ↓
agent.py 触发 tool: recall_session(query="昨天的板卡 KV")
    ↓
edge_memory.recall_memory(...) 命中 L1，拿到 session_id
    ↓
KVStore.load_session(session_id, slot_id=0)
    ├─ ok=True  → llm 直接热启动，回答续上下文
    └─ ok=False → 走 L1 RAG，把对话片段重塞 prompt 重 prefill
```

**不动的接口**：`agent.py`、`voice_pipeline.py`、`QwenTalk.py`、`intel_chan_persona.py`。L4 是新增独立模块。

---

## 10. 后续工作

1. **stage 5 turbo3 完成后**：实测 turbo3 KV 文件大小 + 读写时延（预期 < 0.2 s）
2. **PR llmsrv 启动脚本**：加 `--slot-save-path /home/intel/kv_sessions/`
3. **L3 设计**：对话归档 + embedding，跨周月 RAG
4. **session router**：多 slot 路由（voice 用 slot 0，agent 用 slot 1）

---

## 参考

- ds4: <https://github.com/antirez/ds4>
- llama.cpp server slots: <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>
- ctx-checkpoints PR: <https://github.com/ggml-org/llama.cpp/pull/15293>
- PC801 实验方案: memory `pc801_updated_plan_2026_05_15.md`
