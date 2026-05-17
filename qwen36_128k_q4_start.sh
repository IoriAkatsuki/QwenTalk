#!/usr/bin/env bash
# Start Qwen3.6 35B-A3B on DK-2500 with 128K context and q4 KV cache.
set -eo pipefail

# oneAPI setvars.sh reads unset variables; enable nounset after it is loaded.
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
set -u

exec /home/intel/llama.cpp/build_sycl/bin/llama-server \
  -m /home/intel/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf \
  -ngl 99 -fa 1 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  -c 128000 -np 1 \
  --cache-ram 0 --no-cache-prompt --ctx-checkpoints 0 \
  --port 8080 --host 127.0.0.1 \
  --jinja --reasoning off
