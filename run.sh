#!/usr/bin/env bash
# Minimal runner for CADRL on Unix-like shells
set -euo pipefail

# Usage:
#   ./run.sh --run-name myrun --lr 0.001 --epochs 10

PYTHON=${PYTHON:-python}

exec "$PYTHON" "$(dirname "$0")/main.py" "$@"
