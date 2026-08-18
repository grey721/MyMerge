#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/202422801010/AgentMerge"
cd "${PROJECT_DIR}"

BACKEND="${AGENTMERGE_BACKEND:-vllm}"
DEVICE_SPEC="${AGENTMERGE_DEVICE:-1 2 3 4 5}"
SEED="${AGENTMERGE_SEED:-42}"
BATCH_SIZE="${AGENTMERGE_BATCH_SIZE:-12}"
MAX_CASES="${AGENTMERGE_MAX_CASES:-}"
POPULATION="${AGENTMERGE_POPULATION:-12}"
# Seven progressive stages share exactly 20 candidate batches. Stage changes
# only rebuild and inherit the evaluated population; every configured batch
# is therefore a standard DE candidate batch. With 12 individuals this gives
# a nominal budget of 12 x 20 = 240 expensive FEs.
GENERATION_SCHEDULE="${AGENTMERGE_GENERATIONS:-3 2 2 3 3 3 4}"
RUN_NAME="${AGENTMERGE_RUN_NAME:-qwen3-v03-loop-progressive-de-factor-linear-seed${SEED}}"
OUTPUT_ROOT="${AGENTMERGE_OUTPUT_DIR:-${PROJECT_DIR}/outputs}"
mkdir -p "${OUTPUT_ROOT}"
CONSOLE_CAPTURE="$(mktemp "${OUTPUT_ROOT}/.${RUN_NAME}.console.XXXXXX.log")"
RUN_MARKER="$(mktemp "${OUTPUT_ROOT}/.${RUN_NAME}.started.XXXXXX")"
SEARCH_PID=""
read -r -a DEVICE_ARGS <<< "${DEVICE_SPEC}"
read -r -a GENERATION_ARGS <<< "${GENERATION_SCHEDULE}"
OBJECTIVE_ARGS=()
OBJECTIVE_VALUES=()
CASE_ARGS=()
if [[ -n "${AGENTMERGE_OBJECTIVES:-}" ]]; then
  read -r -a OBJECTIVE_VALUES <<< "${AGENTMERGE_OBJECTIVES}"
elif [[ "${BACKEND}" == "vllm" ]]; then
  OBJECTIVE_VALUES=(
    "domain_quality.language_reasoning:maximize"
    "domain_quality.mathematics:maximize"
    "domain_quality.code:maximize"
  )
fi
for objective in "${OBJECTIVE_VALUES[@]}"; do
  OBJECTIVE_ARGS+=(--objective "${objective}")
done
if [[ -n "${MAX_CASES}" ]]; then
  if [[ ! "${MAX_CASES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "AGENTMERGE_MAX_CASES must be a positive integer" >&2
    exit 2
  fi
  CASE_ARGS=(--max-cases "${MAX_CASES}")
fi

if [[ "${#GENERATION_ARGS[@]}" -ne 7 ]]; then
  echo "AGENTMERGE_GENERATIONS must contain exactly 7 stage values" >&2
  exit 2
fi
generation_total=0
for stage_generations in "${GENERATION_ARGS[@]}"; do
  if [[ ! "${stage_generations}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Every AGENTMERGE_GENERATIONS value must be a positive integer" >&2
    exit 2
  fi
  generation_total="$((generation_total + stage_generations))"
done
if [[ "${generation_total}" -ne 20 ]]; then
  echo "AGENTMERGE_GENERATIONS must sum to 20 for the 240-FE protocol" >&2
  exit 2
fi
if [[ "${POPULATION}" -ne 12 ]]; then
  echo "AGENTMERGE_POPULATION must be 12 for the 12 x 20 protocol" >&2
  exit 2
fi

terminate_search_group() {
  local attempt
  local process_state
  [[ -n "${SEARCH_PID}" ]] || return 0
  if ! kill -0 "${SEARCH_PID}" 2>/dev/null \
    && ! kill -0 -- "-${SEARCH_PID}" 2>/dev/null; then
    return 0
  fi

  # Let the controller run its Python cleanup first. If it has already died,
  # signal its isolated process group so orphaned descendants also exit.
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

case "${BACKEND}" in
  vllm)
    AGENTMERGE=".conda/vllm/bin/agentmerge"
    CONFIG="${AGENTMERGE_CONFIG:-experiments/qwen3_vllm_factor_linear_research_v03.yaml}"
    if [[ -z "${VLLM_LOGGING_CONFIG_PATH:-}" ]]; then
      export VLLM_LOGGING_CONFIG_PATH="${PROJECT_DIR}/scripts/vllm_logging.json"
    fi
    ;;
  peft)
    AGENTMERGE=".conda/qwen3/bin/agentmerge"
    CONFIG="${AGENTMERGE_CONFIG:-experiments/qwen3_de_pool_research_v01.yaml}"
    ;;
  *)
    echo "AGENTMERGE_BACKEND must be vllm or peft" >&2
    exit 2
    ;;
esac

command -v setsid >/dev/null 2>&1 || {
  echo "setsid is required for isolated experiment process cleanup" >&2
  exit 2
}

echo "Search budget: population=${POPULATION}, candidate_batches=${generation_total}, nominal_FE=$((POPULATION * generation_total))"
echo "Freedom schedule: groups=1,2,4,8,16,32,36; dimensions=18,36,72,144,288,576,648"
echo "Stage generations: ${GENERATION_ARGS[*]}"
if [[ "${BACKEND}" == "vllm" ]]; then
  echo "System: shared-scratchpad tool_loop, max_steps=2"
  echo "Objectives:"
  for objective in "${OBJECTIVE_VALUES[@]}"; do
    echo "  ${objective}"
  done
  echo "Inference: batch_size=${BATCH_SIZE}, max_num_seqs=${BATCH_SIZE} (derived), fixed_KV_cache=16_GiB"
else
  echo "System and objectives: loaded from ${CONFIG}"
  echo "Inference: batch_size=${BATCH_SIZE}, backend=${BACKEND}"
fi
echo "Benchmark cases: ${MAX_CASES:-full configured split}"

set +e
AGENTMERGE_WATCH_PARENT=1 \
AGENTMERGE_EXPECTED_PARENT_PID="$$" \
setsid "${AGENTMERGE}" search "${CONFIG}" \
  --run-name "${RUN_NAME}" \
  --output-dir "${OUTPUT_ROOT}" \
  --algorithm progressive_de \
  --refinement_groups 1 2 4 8 16 32 36 \
  --num_layers 36 \
  --populations "${POPULATION}" \
  --generations "${GENERATION_ARGS[@]}" \
  --seed "${SEED}" \
  "${OBJECTIVE_ARGS[@]}" \
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
  > >(tee -a "${CONSOLE_CAPTURE}") 2>&1 &
SEARCH_PID=$!
wait "${SEARCH_PID}"
exit_status=$?
SEARCH_PID=""
set -e

exit "${exit_status}"
