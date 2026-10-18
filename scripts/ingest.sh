#!/usr/bin/env bash
# Upload every file of a folder to the running API.
#   scripts/ingest.sh <dir> [public|members]
# Subfolders named public/ and members/ are uploaded with that visibility.
set -euo pipefail
cd "$(dirname "$0")/.."

dir="${1:?usage: scripts/ingest.sh <dir> [public|members]}"
default_vis="${2:-public}"
base="${API_URL:-http://localhost:8000}"
key=$(grep '^ADMIN_API_KEY=' .env | cut -d= -f2-)
[[ -n "$key" ]] || { echo "ADMIN_API_KEY is empty in .env" >&2; exit 1; }

failed=0
while IFS= read -r -d '' f; do
  vis="$default_vis"
  case "$f" in
    */members/*) vis=members ;;
    */public/*) vis=public ;;
  esac
  code=$(curl -sS -o /tmp/micro-bot-ingest.json -w '%{http_code}' "$base/api/v1/documents" \
    -H "X-Admin-Key: $key" -F "file=@$f" -F "visibility=$vis")
  printf '%-8s %s  %s\n' "$vis" "$code" "$f"
  [[ "$code" == 200 ]] || { cat /tmp/micro-bot-ingest.json; echo; failed=1; }
done < <(find "$dir" -type f ! -name '.*' -print0 | sort -z)
exit "$failed"
