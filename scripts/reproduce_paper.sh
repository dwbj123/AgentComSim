#!/usr/bin/env sh
set -eu

# Run from any directory; resolve the repository root from this script.
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python -m agentcomsim run -c "$ROOT/configs/paper.yaml" -o "$ROOT/results/paper"
