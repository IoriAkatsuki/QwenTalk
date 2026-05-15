# Hermes Agent + QwenTalk MCP

This project exposes board-local perception and system tools through a stdio MCP server so Hermes Agent can orchestrate the DK-2500 edge-agent stack.

## Target Runtime

Hermes Agent 0.12 rejects model context windows below 64K. The default resident board profile uses Qwen3.6 with 64K context and q4_0 KV cache:

```bash
cd /home/intel/QwenTalk
./qwen36_64k_q4_resident.sh
```

Use `./qwen36_64k_q4_start.sh` when a foreground server process is preferred for log inspection.
`qwen36_128k_q4_start.sh` remains available for long-document experiments.

Equivalent raw command:

```bash
source /opt/intel/oneapi/setvars.sh
/home/intel/llama.cpp/build_sycl/bin/llama-server \
  -m /home/intel/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf \
  -ngl 99 -fa 1 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  -c 65536 -np 1 \
  --cache-ram 0 --no-cache-prompt --ctx-checkpoints 0 \
  --port 8080 --host 127.0.0.1 \
  --jinja --reasoning off
```

Notes:

- V-cache q4 requires flash attention, so `-fa 1` is mandatory.
- `-np 1` keeps one 64K slot; default auto slots can inflate memory pressure.
- `--cache-ram 0 --no-cache-prompt --ctx-checkpoints 0` disables prompt-cache/checkpoint overhead for the long-context server profile.
- For low-memory native QwenTalk tests, `-c 8192` is still useful, but Hermes should point at the 64K resident profile.

## Hermes Config

Copy or merge `hermes_qwentalk_config.example.yaml` into `~/.hermes/config.yaml`. The active board profile uses:

- custom OpenAI-compatible provider at `http://127.0.0.1:8080/v1`
- `model.default: Qwen3.6-35B-A3B-UD-IQ2_M`
- `model.context_length: 65536`
- `qwentalk_board` MCP stdio server
- `platform_toolsets.cli` with `qwentalk_board` enabled

Keep `web_search` disabled by default until board network timeout behavior is stabilized.

## MCP Server

Start manually for smoke tests:

```bash
cd /home/intel/QwenTalk
./qwentalk_mcp_start.sh --log-level INFO
```

Hermes launches the same script as a stdio MCP server. MCP stderr is collected in `~/.hermes/logs/mcp-stderr.log`.

## Exposed Tools

- `calculate`: safe math expression evaluation.
- `get_system_info`: CPU, memory, disk, and load metrics.
- `get_temperature`: sysfs thermal-zone temperatures.
- `get_distance`: RealSense D435 regional depth in meters.
- `get_gesture`: D435 + OpenVINO NPU hand gesture state.
- `get_scene`: D435 RGB-D context + SmolVLM2 scene text via `:8081`.
- `memory_store`: persist short-term or long-term agent memory in local SQLite.
- `memory_recall`: retrieve memory using lexical relevance, importance, retrieval reinforcement, and exponential decay.
- `memory_reinforce`: explicitly strengthen a memory; important or repeatedly retrieved short-term memories can become long-term.
- `memory_decay`: preview or delete expired/low-retention memories.
- `memory_status`: report memory database path and counts.
- `web_search`: available in QwenTalk but not recommended for default Hermes exposure on the board.

## SQLite Memory Model

The board-local memory layer is implemented in `edge_memory.py` and stores data at `~/.hermes/qwentalk_memory.sqlite3`.

Design notes:

- The implementation is dependency-free SQLite rather than a full MemPalace deployment, because the board runtime must stay offline-capable and disk-light.
- The layout follows the MemPalace idea of an explicit memory backend/palace and local SQLite temporal graph, but keeps the first board version as a deterministic SQL store exposed over MCP. Reference: <https://github.com/MemPalace/mempalace>.
- Memories are marked `short_term` or `long_term`.
- Retention uses exponential decay: `retention = exp(-ln(2) * elapsed_hours / half_life_hours)`.
- Defaults are 24 h half-life for short-term memory and 720 h for long-term memory.
- Retrieval reinforces memory by increasing stability and access count; high-importance or repeatedly recalled short-term memories can be promoted to long-term.

Manual MCP memory smoke:

```bash
cd /home/intel/QwenTalk
source ~/miniforge3/etc/profile.d/conda.sh
conda run -n openvino python tools.py memory_store '{"content":"用户偏好中文、紧凑、技术性强。","memory_type":"long_term","importance":0.9,"tags":["user","preference"]}'
conda run -n openvino python tools.py memory_recall '{"query":"用户偏好", "top_k": 3}'
conda run -n openvino python tools.py memory_status
```

Rolling compression retention smoke:

```bash
cd /home/intel/QwenTalk
./qwen36_64k_q4_resident.sh
source ~/miniforge3/etc/profile.d/conda.sh
conda run -n openvino python memory_compression_loop.py --rounds 6 --chunk-chars 4500
```

The default corpus is E. R. Eddison's public-domain Project Gutenberg text *The Worm Ouroboros*. Do not use copyrighted Tolkien text for automated download tests.

## Edge-Agent Personality

Hermes primary identity is `~/.hermes/SOUL.md`. The deployed board profile sets it to a DK-2500 local edge-agent persona. Optional named overlay lives under `agent.personalities.dk2500_edge` in `config.yaml`.

## Smoke Test

MCP-only smoke:

```bash
cd /home/intel/QwenTalk
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"smoke","version":"0"}}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"calculate","arguments":{"expression":"2+3*4"}}}' \
  | ./qwentalk_mcp_start.sh --log-level INFO
```

Hermes + local Qwen3.6 smoke:

```bash
cd /home/intel/QwenTalk
hermes -t qwentalk_board -z "必须调用本地 MCP 工具 calculate 计算 6*7。最后只输出 result 数字。"
```

Expected: MCP initialize result, selected board tools, calculate result `14` for MCP-only and `42` for Hermes one-shot.
