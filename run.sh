#!/usr/bin/env bash
# Minimal runner for CADRL on Unix-like shells
set -euo pipefail


# Usage:
#   ./run.sh --mode train --run-name myrun
#   ./run.sh --mode test  --run-name myrun

PYTHON=${PYTHON:-python}

exec "$PYTHON" "$(dirname "$0")/main.py" "$@"
