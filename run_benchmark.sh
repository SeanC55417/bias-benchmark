#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_BENCHTOOLS_DIR="${WORKSPACE_DIR}/benchtools"
BENCHTOOLS_DIR="${BENCHTOOLS_DIR:-${DEFAULT_BENCHTOOLS_DIR}}"
BENCHTOOLS_INSTALL="${BENCHTOOLS_INSTALL:-git+https://github.com/ml4sts/benchtools.git}"
MODEL_NAME="${MODEL_NAME:-llama3}"
OLLAMA_API_URL="${OLLAMA_API_URL:-http://localhost:11434}"

if [[ -d "${BENCHTOOLS_DIR}" ]]; then
  echo "Installing local benchtools from ${BENCHTOOLS_DIR}..."
  python3 -m pip install -e "${BENCHTOOLS_DIR}"
else
  echo "Local benchtools not found at ${BENCHTOOLS_DIR}."
  echo "Installing benchtools from ${BENCHTOOLS_INSTALL}..."
  python3 -m pip install "${BENCHTOOLS_INSTALL}"
fi

echo "Installing tabulate..."
python3 -m pip install tabulate

echo "Ensuring Ollama model '${MODEL_NAME}' is available..."
ollama pull "${MODEL_NAME}"

echo "Running benchmark..."
cd "${SCRIPT_DIR}"
benchtool run . -r ollama -m "${MODEL_NAME}" -a "${OLLAMA_API_URL}" -l logs/
