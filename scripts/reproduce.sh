#!/usr/bin/env bash
set -euo pipefail

# Release verification only. This script never prepares data, trains a model,
# selects a checkpoint, or evaluates a new candidate.
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

"${PYTHON_BIN}" -m pytest -q -p no:cacheprovider
"${PYTHON_BIN}" -m src.quality_control --output reports/v2_quality_control.json
