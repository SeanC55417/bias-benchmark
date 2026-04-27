#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_BENCHTOOLS_DIR="${WORKSPACE_DIR}/benchtools"
BENCHTOOLS_DIR="${BENCHTOOLS_DIR:-${DEFAULT_BENCHTOOLS_DIR}}"
BENCHTOOLS_INSTALL="${BENCHTOOLS_INSTALL:-git+https://github.com/ml4sts/benchtools.git}"
DEFAULT_MODELS=("llama3.1:8b" "qwen2.5:7b" "llama3.2:3b")
OLLAMA_API_URL="${OLLAMA_API_URL:-http://localhost:11434}"

trim_whitespace() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

MODELS=()

if [[ -n "${MODEL_NAME:-}" ]]; then
  MODELS=("${MODEL_NAME}")
elif [[ -n "${MODEL_NAMES:-}" ]]; then
  IFS=',' read -r -a RAW_MODELS <<< "${MODEL_NAMES}"
  for raw_model in "${RAW_MODELS[@]}"; do
    model="$(trim_whitespace "${raw_model}")"
    if [[ -n "${model}" ]]; then
      MODELS+=("${model}")
    fi
  done
else
  MODELS=("${DEFAULT_MODELS[@]}")
fi

if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "No models configured. Set MODEL_NAME or MODEL_NAMES."
  exit 1
fi

if [[ -d "${BENCHTOOLS_DIR}" ]]; then
  echo "Installing local benchtools from ${BENCHTOOLS_DIR}..."
  python3 -m pip install -e "${BENCHTOOLS_DIR}"
else
  echo "Local benchtools not found at ${BENCHTOOLS_DIR}."
  echo "Installing benchtools from ${BENCHTOOLS_INSTALL}..."
  python3 -m pip install "${BENCHTOOLS_INSTALL}"
fi

echo "Installing pandas<3 for benchtools compatibility..."
python3 -m pip install "pandas<3"

echo "Installing tabulate..."
python3 -m pip install tabulate

cd "${SCRIPT_DIR}"

for model in "${MODELS[@]}"; do
  echo "Ensuring Ollama model '${model}' is available..."
  ollama pull "${model}"

  echo "Running benchmark for ${model}..."
  benchtool run . -r ollama -m "${model}" -a "${OLLAMA_API_URL}" -l logs/
done

