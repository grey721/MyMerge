#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/202422801010/AgentMerge"
cd "${PROJECT_DIR}"

REQUESTED_DEVICE_SPEC="${AGENTMERGE_DEVICE:-}"
MIN_FREE_GPU_MIB="${AGENTMERGE_MIN_FREE_GPU_MIB:-49152}"
SEED="${AGENTMERGE_SEED:-42}"
BATCH_SIZE="${AGENTMERGE_BATCH_SIZE:-12}"
POPULATION="${AGENTMERGE_POPULATION:-12}"
TOTAL_BUDGET="${AGENTMERGE_TOTAL_BUDGET:-240}"
MIN_STAGE_FE_SPEC="${AGENTMERGE_MIN_STAGE_FE:-24 24 24 24 24 72}"
MAX_CASES="${AGENTMERGE_MAX_CASES:-}"
RUN_NAME="${AGENTMERGE_RUN_NAME:-qwen3-v03-loop-adaptive-resolution-de-factor-linear-seed${SEED}}"
OUTPUT_ROOT="${AGENTMERGE_OUTPUT_DIR:-${PROJECT_DIR}/outputs}"
CONFIG="${AGENTMERGE_CONFIG:-experiments/qwen3_vllm_factor_linear_research_v03.yaml}"
AGENTMERGE=".conda/vllm/bin/agentmerge"

if [[ ! "${MIN_FREE_GPU_MIB}" =~ ^[1-9][0-9]*$ ]]; then
  echo "AGENTMERGE_MIN_FREE_GPU_MIB must be a positive integer" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required for the GPU-memory preflight" >&2
  exit 2
fi
if ! GPU_MEMORY_ROWS="$(nvidia-smi \
  --query-gpu=index,memory.free \
  --format=csv,noheader,nounits 2>&1)"; then
  echo "Unable to query free GPU memory: ${GPU_MEMORY_ROWS}" >&2
  exit 2
fi

declare -A GPU_FREE_MIB=()
GPU_ORDER=()
while IFS=', ' read -r gpu_id free_mib; do
  [[ "${gpu_id}" =~ ^[0-9]+$ ]] || continue
  [[ "${free_mib}" =~ ^[0-9]+$ ]] || continue
  GPU_ORDER+=("${gpu_id}")
  GPU_FREE_MIB["${gpu_id}"]="${free_mib}"
done <<< "${GPU_MEMORY_ROWS}"
if [[ "${#GPU_ORDER[@]}" -eq 0 ]]; then
  echo "nvidia-smi returned no usable CUDA devices" >&2
  exit 2
fi

DEVICE_ARGS=()
EXCLUDED_GPU_REPORT=()
if [[ -n "${REQUESTED_DEVICE_SPEC}" ]]; then
  read -r -a DEVICE_ARGS <<< "${REQUESTED_DEVICE_SPEC}"
  for gpu_id in "${DEVICE_ARGS[@]}"; do
    if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
      echo "AGENTMERGE_DEVICE contains an invalid GPU ID: ${gpu_id}" >&2
      exit 2
    fi
    if [[ -z "${GPU_FREE_MIB[${gpu_id}]+x}" ]]; then
      echo "Requested GPU ${gpu_id} is not visible" >&2
      exit 2
    fi
    if [[ "${GPU_FREE_MIB[${gpu_id}]}" -lt "${MIN_FREE_GPU_MIB}" ]]; then
      EXCLUDED_GPU_REPORT+=("${gpu_id}(${GPU_FREE_MIB[${gpu_id}]} MiB free)")
    fi
  done
  if [[ "${#EXCLUDED_GPU_REPORT[@]}" -gt 0 ]]; then
    echo "Requested GPU(s) lack the ${MIN_FREE_GPU_MIB} MiB startup reserve: ${EXCLUDED_GPU_REPORT[*]}" >&2
    echo "Choose free GPUs with AGENTMERGE_DEVICE=\"0 1 ...\" or lower the explicit preflight threshold only when safe." >&2
    exit 2
  fi
  DEVICE_SELECTION="explicit"
else
  for gpu_id in "${GPU_ORDER[@]}"; do
    if [[ "${GPU_FREE_MIB[${gpu_id}]}" -ge "${MIN_FREE_GPU_MIB}" ]]; then
      DEVICE_ARGS+=("${gpu_id}")
    else
      EXCLUDED_GPU_REPORT+=("${gpu_id}(${GPU_FREE_MIB[${gpu_id}]} MiB free)")
    fi
  done
  if [[ "${#DEVICE_ARGS[@]}" -eq 0 ]]; then
    echo "No GPU has the required ${MIN_FREE_GPU_MIB} MiB startup reserve" >&2
    echo "Observed: ${EXCLUDED_GPU_REPORT[*]}" >&2
    exit 2
  fi
  DEVICE_SELECTION="free-memory-auto"
fi
DEVICE_SPEC="${DEVICE_ARGS[*]}"

mkdir -p "${OUTPUT_ROOT}"
CONSOLE_CAPTURE="$(mktemp "${OUTPUT_ROOT}/.${RUN_NAME}.console.XXXXXX.log")"
RUN_MARKER="$(mktemp "${OUTPUT_ROOT}/.${RUN_NAME}.started.XXXXXX")"
SEARCH_PID=""
read -r -a MIN_STAGE_FE_ARGS <<< "${MIN_STAGE_FE_SPEC}"

if [[ "${POPULATION}" -lt 4 ]]; then
  echo "AGENTMERGE_POPULATION must be at least 4" >&2
  exit 2
fi
if [[ "${TOTAL_BUDGET}" -lt 1 ]]; then
  echo "AGENTMERGE_TOTAL_BUDGET must be positive" >&2
  exit 2
fi
if [[ "${#MIN_STAGE_FE_ARGS[@]}" -ne 6 ]]; then
  echo "AGENTMERGE_MIN_STAGE_FE must contain 6 values for groups 1 4 8 16 32 36" >&2
  exit 2
fi
minimum_sum=0
for minimum in "${MIN_STAGE_FE_ARGS[@]}"; do
  if [[ ! "${minimum}" =~ ^[0-9]+$ ]]; then
    echo "Every AGENTMERGE_MIN_STAGE_FE value must be a non-negative integer" >&2
    exit 2
  fi
  minimum_sum="$((minimum_sum + minimum))"
done
if [[ "${minimum_sum}" -gt "${TOTAL_BUDGET}" ]]; then
  echo "AGENTMERGE_MIN_STAGE_FE sum cannot exceed AGENTMERGE_TOTAL_BUDGET" >&2
  exit 2
fi
if [[ "${MIN_STAGE_FE_ARGS[0]}" -lt "${POPULATION}" ]]; then
  echo "The first stage minimum must cover the initial population" >&2
  exit 2
fi

CASE_ARGS=()
if [[ -n "${MAX_CASES}" ]]; then
  if [[ ! "${MAX_CASES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "AGENTMERGE_MAX_CASES must be a positive integer" >&2
    exit 2
  fi
  CASE_ARGS=(--max-cases "${MAX_CASES}")
fi

if [[ -z "${VLLM_LOGGING_CONFIG_PATH:-}" ]]; then
  export VLLM_LOGGING_CONFIG_PATH="${PROJECT_DIR}/scripts/vllm_logging.json"
fi

terminate_search_group() {
  local attempt
  local process_state
  [[ -n "${SEARCH_PID}" ]] || return 0
  if ! kill -0 "${SEARCH_PID}" 2>/dev/null \
    && ! kill -0 -- "-${SEARCH_PID}" 2>/dev/null; then
    return 0
  fi
  if kill -0 "${SEARCH_PID}" 2>/dev/null; then
    kill -TERM "${SEARCH_PID}" 2>/dev/null || true
  else
    kill -TERM -- "-${SEARCH_PID}" 2>/dev/null || true
  fi
  for attempt in $(seq 1 50); do
    process_state="$(ps -o stat= -p "${SEARCH_PID}" 2>/dev/null || true)"
    if [[ "${process_state}" == Z* ]]; then
      wait "${SEARCH_PID}" 2>/dev/null || true
    fi
    if ! kill -0 -- "-${SEARCH_PID}" 2>/dev/null; then
      wait "${SEARCH_PID}" 2>/dev/null || true
      return 0
    fi
    sleep 0.2
  done
  kill -KILL -- "-${SEARCH_PID}" 2>/dev/null \
    || kill -KILL "${SEARCH_PID}" 2>/dev/null \
    || true
  wait "${SEARCH_PID}" 2>/dev/null || true
}

finalize_console_log() {
  local run_dir
  run_dir="$(find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d \
    -name "${RUN_NAME}-*" -newer "${RUN_MARKER}" \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
  if [[ -n "${run_dir}" && -d "${run_dir}" ]]; then
    mv -f "${CONSOLE_CAPTURE}" "${run_dir}/console.log"
    printf 'Full console log: %s\n' "${run_dir}/console.log"
  else
    printf 'Run directory was not created; console log retained at: %s\n' \
      "${CONSOLE_CAPTURE}" >&2
  fi
  rm -f "${RUN_MARKER}"
}

cleanup() {
  local exit_status=$?
  trap - EXIT INT TERM HUP
  terminate_search_group
  finalize_console_log
  exit "${exit_status}"
}

interrupted() {
  local exit_status="$1"
  trap - INT TERM HUP
  terminate_search_group
  exit "${exit_status}"
}

trap cleanup EXIT
trap 'interrupted 130' INT
trap 'interrupted 143' TERM
trap 'interrupted 129' HUP

command -v setsid >/dev/null 2>&1 || {
  echo "setsid is required for isolated experiment process cleanup" >&2
  exit 2
}

echo "Adaptive real-FE budget: ${TOTAL_BUDGET}"
echo "Nested resolution: groups=1,4,8,16,32,36; dimensions=18,72,144,288,576,648"
echo "Minimum stage FE: ${MIN_STAGE_FE_ARGS[*]} (sum=${minimum_sum}, adaptive_slack=$((TOTAL_BUDGET - minimum_sum)))"
echo "System: shared-scratchpad tool_loop, max_steps=2"
echo "Objectives: language_reasoning, mathematics, code"
echo "Inference: devices=${DEVICE_ARGS[*]}, selection=${DEVICE_SELECTION}, batch_size=${BATCH_SIZE}, max_num_seqs=${BATCH_SIZE} (derived), fixed_KV_cache=16_GiB"
if [[ "${#EXCLUDED_GPU_REPORT[@]}" -gt 0 ]]; then
  echo "GPU preflight excluded: ${EXCLUDED_GPU_REPORT[*]}"
fi
echo "Benchmark cases: ${MAX_CASES:-full configured split}"

set +e
AGENTMERGE_WATCH_PARENT=1 \
AGENTMERGE_EXPECTED_PARENT_PID="$$" \
setsid "${AGENTMERGE}" search "${CONFIG}" \
  --run-name "${RUN_NAME}" \
  --output-dir "${OUTPUT_ROOT}" \
  --algorithm adaptive_resolution_de \
  --num_layers 36 \
  --first_split 4 \
  --populations "${POPULATION}" \
  --total_budget "${TOTAL_BUDGET}" \
  --min_stage_fe "${MIN_STAGE_FE_ARGS[@]}" \
  --seed "${SEED}" \
  --objective domain_quality.language_reasoning:maximize \
  --objective domain_quality.mathematics:maximize \
  --objective domain_quality.code:maximize \
  --device "${DEVICE_ARGS[@]}" \
  --batch_size "${BATCH_SIZE}" \
  "${CASE_ARGS[@]}" \
  --differential_weight 0.5 \
  --crossover_probability 0.9 \
  --mutation_probability 0.05 \
  --mutation_distribution_index 20 \
  --neighborhood_size 3 \
  --neighborhood_mating_probability 0.9 \
  --max_replacements 2 \
  --decomposition tchebycheff \
  --min_active_adapters 0 \
  --max_active_adapters 6 \
  --num_sectors 5 \
  --sector_seed 0 \
  --direction_window 8 \
  --min_directions_per_sector 4 \
  --min_valid_sector_fraction 0.5 \
  --direction_norm_eps 1e-12 \
  --coherence_threshold 0.0 \
  --utility_ratio_threshold 0.2 \
  --saturation_patience 2 \
  --utility_ema_beta 0.8 \
  --utility_eps 1e-12 \
  --checkpoint_interval_fe "${POPULATION}" \
  --enable_zero_progress_fallback \
  --utility_zero_eps 1e-12 \
  --symmetry_diagnostic_batches 3 \
  > >(tee -a "${CONSOLE_CAPTURE}") 2>&1 &
SEARCH_PID=$!
wait "${SEARCH_PID}"
exit_status=$?
SEARCH_PID=""
set -e

exit "${exit_status}"
