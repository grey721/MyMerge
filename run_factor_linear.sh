#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/202422801010/AgentMerge"
cd "${PROJECT_DIR}"

# Run two matched 240-FE searches serially. The child scripts retain their
# process-group cleanup, console capture and candidate-worker lifecycle guards.
FACTOR_CONFIG="${AGENTMERGE_CONFIG:-experiments/qwen3_vllm_factor_linear_research_v03.yaml}"
SEED="${AGENTMERGE_SEED:-42}"

echo "Factor-linear experiment suite started"
echo "Step 1/2: progressive recursive-refinement DE (240 nominal FEs)"
AGENTMERGE_BACKEND=vllm \
AGENTMERGE_CONFIG="${FACTOR_CONFIG}" \
AGENTMERGE_RUN_NAME="qwen3-progressive-de-6-factor-linear-vllm-seed${SEED}" \
AGENTMERGE_GENERATIONS="${AGENTMERGE_PROGRESSIVE_GENERATIONS:-3 2 2 3 3 3 4}" \
./run_progressive_de.sh

echo "Step 2/2: ordinary single-stage adapter-level DE (240 nominal FEs)"
AGENTMERGE_CONFIG="${FACTOR_CONFIG}" \
AGENTMERGE_RUN_NAME="qwen3-adapter-de-6-factor-linear-vllm-seed${SEED}" \
AGENTMERGE_GENERATIONS="${AGENTMERGE_SINGLE_GENERATIONS:-20}" \
./run_de.sh

echo "Factor-linear experiment suite completed"
