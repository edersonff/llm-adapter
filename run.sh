#!/usr/bin/env bash
set -euo pipefail

dir="$(cd "$(dirname "$0")" && pwd)"
cd "$dir"

if [ ! -f target/release/llm-adapter ]; then
  echo "building llm-adapter (first run)..." >&2
  cargo build --release
fi

exec target/release/llm-adapter "$@"
