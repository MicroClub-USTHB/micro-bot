#!/usr/bin/env bash
# Wait until the API answers on /api/v1/health, then print its status.
set -euo pipefail
url="${1:-http://localhost:8000}/api/v1/health"
timeout="${2:-180}"

for ((i = 0; i < timeout; i += 2)); do
  if body=$(curl -fsS "$url" 2>/dev/null); then
    echo "API up: $body"
    exit 0
  fi
  sleep 2
done
echo "API not healthy after ${timeout}s; see: make logs" >&2
exit 1
