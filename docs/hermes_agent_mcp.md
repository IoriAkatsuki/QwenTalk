# Hermes Agent + QwenTalk MCP

This project exposes board-local perception and system tools through a stdio MCP server so Hermes Agent can orchestrate the DK-2500 edge-agent stack.

## Target runtime

LLM server stays local and should use the Qwen3.6 + compressed KV-cache profile:

```bash
source /opt/intel/oneapi/setvars.sh
/home/intel/llama.cpp/build_sycl/bin/llama-server \
  -m /home/intel/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf \
  -ngl 99 -fa 0 \
  --cache-type-k q8_0 --cache-type-v f16 \
  -c 8192 \
  --port 8080 --host 127.0.0.1 \
  --jinja --reasoning off
```

The MCP server does not start the LLM. It only exposes hardware/system tools.

## MCP server

Start manually for smoke tests:

```bash
cd /home/intel/QwenTalk
./qwentalk_mcp_start.sh
```

Hermes should launch it as a stdio MCP server. Add this block under `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  qwentalk_board:
    command: "/home/intel/QwenTalk/qwentalk_mcp_start.sh"
    args: []
    timeout: 180
    connect_timeout: 60
    enabled: true
    tools:
      include:
        - calculate
        - get_system_info
        - get_temperature
        - get_distance
        - get_gesture
        - get_scene
      resources: false
      prompts: false
```

Keep `web_search` disabled by default until board network timeout behavior is stabilized.

## Exposed tools

- `calculate`: safe math expression evaluation.
- `get_system_info`: CPU, memory, disk, and load metrics.
- `get_temperature`: sysfs thermal-zone temperatures.
- `get_distance`: RealSense D435 regional depth in meters.
- `get_gesture`: D435 + OpenVINO NPU hand gesture state.
- `get_scene`: D435 RGB-D context + SmolVLM2 scene text via `:8081`.
- `web_search`: DuckDuckGo instant-answer search, available but not recommended for default Hermes tool exposure on the board.

## Edge-agent personality

Hermes personality example:

```yaml
personalities:
  dk2500_edge: |
    你是运行在 Intel DK-2500 板卡上的本地边缘智能体。
    你通过本地 Qwen3.6、D435 深度相机、OpenVINO NPU 和本地语音链路与人交互。
    回答简洁，优先调用本地 MCP 工具获取实时状态。
    不把云端能力说成本地能力；传感器不可用时直接说明。
```

## Smoke test

```bash
cd /home/intel/QwenTalk
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"smoke","version":"0"}}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"calculate","arguments":{"expression":"2+3*4"}}}' \
  | ./qwentalk_mcp_start.sh
```

Expected: initialize result, 7 listed tools, and a calculate result of `14`.
