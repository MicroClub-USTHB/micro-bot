#!/usr/bin/env bash
# Upload every file of a folder (subfolders included) to the running API.
#   scripts/ingest.sh <dir>
set -euo pipefail
cd "$(dirname "$0")/.."

dir="${1:?usage: scripts/ingest.sh <dir>}"
base="${API_URL:-http://localhost:8000}"
key=$(grep '^ADMIN_API_KEY=' .env | cut -d= -f2-)
[[ -n "$key" ]] || { echo "ADMIN_API_KEY is empty in .env" >&2; exit 1; }

failed=0
while IFS= read -r -d '' f; do
  code=$(curl -sS -o /tmp/micro-bot-ingest.json -w '%{http_code}' "$base/api/v1/documents" \
    -H "X-Admin-Key: $key" -F "file=@$f")
  printf '%s  %s\n' "$code" "$f"
  [[ "$code" == 200 ]] || { cat /tmp/micro-bot-ingest.json; echo; failed=1; }
done < <(find "$dir" -type f ! -name '.*' -print0 | sort -z)
exit "$failed"
