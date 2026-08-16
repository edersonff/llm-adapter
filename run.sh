#!/usr/bin/env bash
set -euo pipefail
dir="$(cd "$(dirname "$0")" && pwd)"
cd "$dir"

CONFIG="${LLM_CONFIG:-config.yaml}"
if [ ! -f "$CONFIG" ] && [ -f config.yaml.template ]; then
  cp config.yaml.template "$CONFIG"
  echo "created $CONFIG from the template — fill in one provider key, or export LLM_CONFIG=/path/to/your/config.yaml" >&2
fi

if [ ! -f target/release/llm-adapter ]; then
  echo "building llm-adapter (first run)..." >&2
  cargo build --release
fi

exec target/release/llm-adapter --config "$CONFIG" "$@"
