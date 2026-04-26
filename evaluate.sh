#!/usr/bin/env bash
set -euo pipefail

# Project root = directory containing this script
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${ROOT_DIR}/scripts/evaluate.py"

# Optional override, for example:
# PYTHON_BIN="/c/Users/letru/miniconda3/envs/ml_p1/python.exe" ./evaluate.sh --split test
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${ROOT_DIR}"
"${PYTHON_BIN}" "${EVAL_SCRIPT}" "$@"
