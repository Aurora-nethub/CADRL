#!/usr/bin/env bash
set -euo pipefail

# Usage: ./data/run_generate.sh --n-episodes 100
PYTHON=${PYTHON:-python}
exec "$PYTHON" "$(dirname "$0")/generate_multi_sim_orca.py" "$@"
