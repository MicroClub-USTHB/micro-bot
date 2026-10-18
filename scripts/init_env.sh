#!/usr/bin/env bash
# Create .env from .env.example with random MEMBER_TOKEN and ADMIN_API_KEY. Never overwrites.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  echo ".env already exists, left untouched"
  exit 0
fi

rand() { openssl rand -hex 24 2>/dev/null || python3 -c 'import secrets; print(secrets.token_hex(24))'; }

sed -e "s|^MEMBER_TOKEN=.*|MEMBER_TOKEN=$(rand)|" \
    -e "s|^ADMIN_API_KEY=.*|ADMIN_API_KEY=$(rand)|" \
    .env.example > .env
echo "created .env with fresh MEMBER_TOKEN and ADMIN_API_KEY"
grep -q '^OPENROUTER_API_KEY=.\+' .env || echo "tip: set OPENROUTER_API_KEY in .env to get written answers"
