#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCHTOOLS_DIR="${WORKSPACE_DIR}/benchtools"
MODEL_NAME="${MODEL_NAME:-llama3}"
OLLAMA_API_URL="${OLLAMA_API_URL:-http://localhost:11434}"

if [[ ! -d "${BENCHTOOLS_DIR}" ]]; then
  echo "Expected benchtools at: ${BENCHTOOLS_DIR}"
  exit 1
fi

echo "Installing local benchtools..."
python3 -m pip install -e "${BENCHTOOLS_DIR}"

echo "Installing tabulate..."
python3 -m pip install tabulate

echo "Ensuring Ollama model '${MODEL_NAME}' is available..."
ollama pull "${MODEL_NAME}"

echo "Running benchmark..."
cd "${SCRIPT_DIR}"
benchtool run . -r ollama -m "${MODEL_NAME}" -a "${OLLAMA_API_URL}" -l logs/
