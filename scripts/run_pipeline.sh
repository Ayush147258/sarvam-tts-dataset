#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/the pipeline${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONIOENCODING="utf-8"

SOURCES_FILE="data/sources_en.yaml"
LANGUAGE="en-IN"
LIMIT=""
STAGE_FROM="download"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sources)
      SOURCES_FILE="$2"
      shift 2
      ;;
    --language)
      LANGUAGE="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --stage-from)
      STAGE_FROM="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

args=(--sources "$SOURCES_FILE" --language "$LANGUAGE" --stage-from "$STAGE_FROM")
if [[ -n "$LIMIT" ]]; then
  args+=(--limit "$LIMIT")
fi

uv run python -m src.pipeline "${args[@]}"
