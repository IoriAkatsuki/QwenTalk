#!/usr/bin/env bash
# Start QwenTalk board-local tools as an MCP stdio server.
set -E -o pipefail

cd "$(dirname "$0")"

if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "$CONDA_DEFAULT_ENV" != "openvino" ]; then
    if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1090
        source "$HOME/miniforge3/etc/profile.d/conda.sh"
        conda activate openvino
    fi
fi

if [ -f /opt/intel/oneapi/setvars.sh ] && [ -z "${ONEAPI_ROOT:-}" ]; then
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
fi

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

exec python qwentalk_mcp_server.py "$@"
