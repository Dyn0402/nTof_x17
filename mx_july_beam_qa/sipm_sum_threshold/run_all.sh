#!/bin/bash
# Memory-safe driver: one arm per process so the OS reclaims RAM between arms
# (peak ~1.75 GB each). Usage: ./run_all.sh [run_file]
set -e
RUN="${1:-$HOME/x17/beam_july/data/run224460.root}"
PY="$(dirname "$0")/../../.venv/bin/python"
for A in A B C D; do
  echo "=== arm $A ==="
  "$PY" "$(dirname "$0")/build_sum.py" "$A" "$RUN"
done
"$PY" "$(dirname "$0")/plot_threshold.py" "$(basename "$RUN" .root)"
