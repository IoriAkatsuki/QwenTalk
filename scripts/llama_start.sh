#!/bin/bash
exec /home/intel/llama.cpp/build/bin/llama-server \
    -m /home/intel/models/gemma-4-E4B-it-Q4_K_M.gguf \
    -ngl 99 --port 8080 --host 127.0.0.1 \
    -c 16384 -n 500 \
    -ctk q8_0 -ctv q8_0 \
    --jinja --chat-template-kwargs '{"enable_thinking":false}'
