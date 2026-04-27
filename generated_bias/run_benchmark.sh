#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_BENCHTOOLS_DIR="${WORKSPACE_DIR}/benchtools"
BENCHTOOLS_DIR="${BENCHTOOLS_DIR:-${DEFAULT_BENCHTOOLS_DIR}}"
BENCHTOOLS_INSTALL="${BENCHTOOLS_INSTALL:-git+https://github.com/ml4sts/benchtools.git}"
AIF360_INSTALL="${AIF360_INSTALL:-git+https://github.com/Trusted-AI/AIF360.git}"
DEFAULT_MODELS=("llama3.1:8b" "qwen2.5:7b" "llama3.2:3b")
OLLAMA_API_URL="${OLLAMA_API_URL:-http://localhost:11434}"
# Dataset controls are environment variables so long runs do not need edits.
GENERATE_DATASET="${GENERATE_DATASET:-1}"
DATASET_EPISODES="${DATASET_EPISODES:-1}"
DATASET_EPISODE_SIZE="${DATASET_EPISODE_SIZE:-10}"
DATASET_CAPACITY="${DATASET_CAPACITY:-4}"
DATASET_SEED="${DATASET_SEED:-42}"
BENCHMARK_RUNS="${BENCHMARK_RUNS:-1}"
CLEAR_LOGS="${CLEAR_LOGS:-0}"
CLEAR_ANALYSIS="${CLEAR_ANALYSIS:-0}"

trim_whitespace() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

MODELS=()

# Accept either one model or a comma-separated model list.
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

# Fail early on typo-prone run controls before installing dependencies.
if [[ ! "${BENCHMARK_RUNS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BENCHMARK_RUNS must be a positive integer. Got: ${BENCHMARK_RUNS}"
  exit 1
fi

if [[ ! "${CLEAR_LOGS}" =~ ^[01]$ ]]; then
  echo "CLEAR_LOGS must be 0 or 1. Got: ${CLEAR_LOGS}"
  exit 1
fi

if [[ ! "${CLEAR_ANALYSIS}" =~ ^[01]$ ]]; then
  echo "CLEAR_ANALYSIS must be 0 or 1. Got: ${CLEAR_ANALYSIS}"
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

echo "Installing AIF360 from ${AIF360_INSTALL}..."
python3 -m pip install "${AIF360_INSTALL}"

echo "Installing pandas<3 for benchtools compatibility..."
python3 -m pip install "pandas<3"

echo "Installing tabulate..."
python3 -m pip install tabulate

echo "Patching BenchTools CSV custom response loading..."
python3 "${SCRIPT_DIR}/patch_benchtools.py"

# Run from this folder because BenchTools expects info.yml and tasks/ here.
cd "${SCRIPT_DIR}"

if [[ "${CLEAR_LOGS}" == "1" ]]; then
  echo "Clearing old logs..."
  rm -rf logs/
fi

if [[ "${CLEAR_ANALYSIS}" == "1" ]]; then
  echo "Clearing old analysis outputs..."
  rm -rf analysis/
fi

if [[ "${GENERATE_DATASET}" != "0" ]]; then
  echo "Generating randomized patient prompts..."
  python3 "${SCRIPT_DIR}/generate_dataset.py" \
    --episodes "${DATASET_EPISODES}" \
    --episode-size "${DATASET_EPISODE_SIZE}" \
    --capacity "${DATASET_CAPACITY}" \
    --seed "${DATASET_SEED}"
fi

for model in "${MODELS[@]}"; do
  echo "Ensuring Ollama model '${model}' is available..."
  ollama pull "${model}"

  for run_index in $(seq 1 "${BENCHMARK_RUNS}"); do
    if [[ "${run_index}" -gt 1 ]]; then
      # BenchTools uses timestamp-like run ids, so avoid collisions in loops.
      sleep 1
    fi

    echo "Running benchmark for ${model} (${run_index}/${BENCHMARK_RUNS})..."
    benchtool run . -r ollama -m "${model}" -a "${OLLAMA_API_URL}" -l logs/
  done
done
