#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"

SERVICES=(
  "sidecar-7b:8001"
  "gemma4:8002"
  "qwen3-coder:8003"
  "qwen36-35b:8004"
)

usage() {
  echo "Usage: $0 {up|down|status}"
  exit 1
}

wait_for_health() {
  local port="$1"
  local service="$2"
  local max_retries=60
  local retry=0

  while [ $retry -lt $max_retries ]; do
    if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
      echo "  ${service} is healthy on port ${port}"
      return 0
    fi
    retry=$((retry + 1))
    sleep 1
  done

  echo "  ERROR: ${service} failed to become healthy on port ${port} after ${max_retries}s"
  return 1
}

sleep_model() {
  local port="$1"
  local service="$2"

  if curl -sf "http://localhost:${port}/is_sleeping" > /dev/null 2>&1; then
    local sleeping
    sleeping=$(curl -sf "http://localhost:${port}/is_sleeping" 2>/dev/null || echo "false")
    if [ "$sleeping" = "true" ]; then
      echo "  ${service} is already sleeping"
      return 0
    fi
  fi

  echo "  Sleeping ${service} on port ${port}..."
  local resp
  resp=$(curl -sf -X POST "http://localhost:${port}/sleep?level=2&mode=wait" 2>/dev/null || true)
  if [ -n "$resp" ]; then
    echo "  ${service} sleep response: ${resp}"
  else
    echo "  ${service} sleep request sent"
  fi
}

cmd_up() {
  echo "=== Starting vLLM Model Zoo ==="

  mkdir -p ~/.cache/vllm ~/.cache/triton

  for entry in "${SERVICES[@]}"; do
    IFS=':' read -r service port <<< "$entry"
    echo ""
    echo "Starting ${service} on port ${port}..."
    docker compose -f "$COMPOSE_FILE" up -d "$service"

    if ! wait_for_health "$port" "$service"; then
      echo "ERROR: Failed to start ${service}"
      exit 1
    fi

    sleep_model "$port" "$service"
  done

  echo ""
  echo "=== All models started and sleeping ==="
}

cmd_down() {
  echo "=== Stopping vLLM Model Zoo ==="
  docker compose -f "$COMPOSE_FILE" down
  echo "=== All models stopped ==="
}

cmd_status() {
  echo "=== vLLM Model Zoo Status ==="
  printf "%-15s %-8s %-20s\n" "SERVICE" "PORT" "STATE"
  printf "%-15s %-8s %-20s\n" "-------" "----" "-----"

  for entry in "${SERVICES[@]}"; do
    IFS=':' read -r service port <<< "$entry"

    if ! curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
      printf "%-15s %-8s %-20s\n" "$service" "$port" "STOPPED"
      continue
    fi

    local is_sleeping
    is_sleeping=$(curl -sf "http://localhost:${port}/is_sleeping" 2>/dev/null || echo "false")

    if [ "$is_sleeping" = "true" ]; then
      printf "%-15s %-8s %-20s\n" "$service" "$port" "SLEEPING (L2)"
    else
      printf "%-15s %-8s %-20s\n" "$service" "$port" "AWAKE"
    fi
  done
}

if [ $# -lt 1 ]; then
  usage
fi

case "$1" in
  up)
    cmd_up
    ;;
  down)
    cmd_down
    ;;
  status)
    cmd_status
    ;;
  *)
    usage
    ;;
esac
