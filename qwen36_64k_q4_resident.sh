#!/usr/bin/env bash
# Keep the 64K Qwen3.6 llama-server profile resident in the background.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

LOG_FILE="${QWEN36_64K_LOG:-/tmp/qwen36_64k_q4_start.log}"
PID_FILE="${QWEN36_64K_PID:-/tmp/qwen36_64k_q4_start.pid}"
HEALTH_URL="${QWEN36_64K_HEALTH:-http://127.0.0.1:8080/health}"

find_running_pids() {
  while read -r pid comm _; do
    if [[ "$comm" == "llama-server" ]] && pid_is_64k_server "$pid"; then
      echo "$pid"
    fi
  done < <(ps -eo pid=,comm=,args=)
}

stop_extra_matching() {
  local keep_pid="$1"
  local pid
  while read -r pid; do
    if [[ -n "$pid" ]] && [[ "$pid" != "$keep_pid" ]]; then
      echo "stopping extra matching 64K Qwen3.6 server pid=$pid" >&2
      stop_pid "$pid"
    fi
  done < <(find_running_pids)
}

pid_is_64k_server() {
  local pid="$1"
  local args
  args="$(ps -p "$pid" -o args= 2>/dev/null || true)"
  [[ "$args" == *"llama-server"* ]] &&
    [[ "$args" == *"Qwen3.6-35B-A3B-UD-IQ2_M.gguf"* ]] &&
    [[ "$args" == *"-c 65536"* ]] &&
    [[ "$args" == *"--cache-type-k q4_0"* ]] &&
    [[ "$args" == *"--cache-type-v q4_0"* ]] &&
    ([[ "$args" == *"--port 8080"* ]] || [[ "$args" == *"--port=8080"* ]]) &&
    [[ "$args" == *"--host 127.0.0.1"* ]] &&
    [[ "$args" == *"--reasoning off"* ]]
}

health_ok() {
  curl -fsS --max-time 2 "$HEALTH_URL" >/dev/null 2>&1
}

port_owned_by_pid() {
  local pid="$1"
  if ! command -v ss >/dev/null 2>&1; then
    return 0
  fi
  ss -ltnp 2>/dev/null | grep -Eq "127\\.0\\.0\\.1:8080.*pid=${pid},|\\[::1\\]:8080.*pid=${pid},"
}

wait_ready() {
  local pid="$1"
  for _ in $(seq 1 120); do
    if ! kill -0 "$pid" 2>/dev/null; then
      return 1
    fi
    if pid_is_64k_server "$pid" && port_owned_by_pid "$pid" && health_ok; then
      echo "64K Qwen3.6 server ready: pid=$pid log=$LOG_FILE"
      return 0
    fi
    sleep 2
  done
  return 2
}

stop_pid() {
  local pid="$1"
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 10); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 1
  done
  kill -KILL "$pid" 2>/dev/null || true
}

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${old_pid:-}" ]] && [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null && pid_is_64k_server "$old_pid"; then
    if wait_ready "$old_pid"; then
      stop_extra_matching "$old_pid"
      exit 0
    fi
    echo "stale or unhealthy 64K Qwen3.6 server, replacing pid=$old_pid" >&2
    stop_pid "$old_pid"
  fi
  rm -f "$PID_FILE"
fi

running_pids=()
while read -r running_pid; do
  [[ -n "$running_pid" ]] && running_pids+=("$running_pid")
done < <(find_running_pids)
if ((${#running_pids[@]} > 0)); then
  healthy_pid=""
  for running_pid in "${running_pids[@]}"; do
    echo "$running_pid" > "$PID_FILE"
    if [[ -z "$healthy_pid" ]] && wait_ready "$running_pid"; then
      healthy_pid="$running_pid"
    else
      echo "unhealthy or extra matching 64K Qwen3.6 server, replacing pid=$running_pid" >&2
      stop_pid "$running_pid"
    fi
  done
  if [[ -n "$healthy_pid" ]]; then
    echo "$healthy_pid" > "$PID_FILE"
    stop_extra_matching "$healthy_pid"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

nohup bash ./qwen36_64k_q4_start.sh >> "$LOG_FILE" 2>&1 &
pid="$!"
echo "$pid" > "$PID_FILE"

if wait_ready "$pid"; then
  exit 0
fi

echo "64K Qwen3.6 server did not become healthy in time; tail follows:" >&2
stop_pid "$pid"
rm -f "$PID_FILE"
tail -n 80 "$LOG_FILE" >&2 || true
exit 1
