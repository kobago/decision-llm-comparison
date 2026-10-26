#!/bin/bash
# Run every model one at a time (each in its own process so memory is released).
set -u
cd "$(dirname "$0")/.."
export NEEDLE_TELEMETRY=0
for m in "$@"; do
  echo "===== $m $(date +%T)"
  uv run python bench/run.py "$m" > "logs/run-$m.log" 2>&1 && echo "OK $m" || echo "FAIL $m (see logs/run-$m.log)"
done
echo ALL_DONE
