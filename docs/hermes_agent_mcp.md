# Hermes Agent + QwenTalk MCP

This project exposes board-local perception and system tools through a stdio MCP server so Hermes Agent can orchestrate the DK-2500 edge-agent stack.

## Target Runtime

Hermes Agent 0.12 rejects model context windows below 64K. The board profile now uses Qwen3.6 with 128K context and q4_0 KV cache:

```bash
cd /home/intel/QwenTalk
./qwen36_128k_q4_start.sh
```

Equivalent raw command:

```bash
source /opt/intel/oneapi/setvars.sh
/home/intel/llama.cpp/build_sycl/bin/llama-server \
  -m /home/intel/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf \
  -ngl 99 -fa 1 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  -c 128000 -np 1 \
  --cache-ram 0 --no-cache-prompt --ctx-checkpoints 0 \
  --port 8080 --host 127.0.0.1 \
  --jinja --reasoning off
```

Notes:

- V-cache q4 requires flash attention, so `-fa 1` is mandatory.
- `-np 1` keeps one 128K slot; default auto slots can inflate memory pressure.
- `--cache-ram 0 --no-cache-prompt --ctx-checkpoints 0` disables prompt-cache/checkpoint overhead for the long-context server profile.
- For low-memory native QwenTalk tests, `-c 8192` is still useful, but Hermes should point at the 128K profile.

## Hermes Config

Copy or merge `hermes_qwentalk_config.example.yaml` into `~/.hermes/config.yaml`. The active board profile uses:

- custom OpenAI-compatible provider at `http://127.0.0.1:8080/v1`
- `model.default: Qwen3.6-35B-A3B-UD-IQ2_M`
- `model.context_length: 128000`
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
- `web_search`: available in QwenTalk but not recommended for default Hermes exposure on the board.

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
