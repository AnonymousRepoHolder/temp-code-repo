#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 The NeuralVirtualizer Authors
#
# Run graph matching evaluation for all 10 models with both LLMs
# Usage: bash security_eval/ModelObfuscator/struct/run_all.sh

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

LLMS=("gpt5" "claude")
MODEL_IDS=(1 2 3 4 5 6 7 8 9 10)

echo "========================================"
echo "Graph Matching Evaluation - Full Run"
echo "========================================"
echo "Models: ${#MODEL_IDS[@]}"
echo "LLMs: ${LLMS[@]}"
echo "Total tasks: $((${#MODEL_IDS[@]} * ${#LLMS[@]}))"
echo ""

START_TIME=$(date +%s)
TASK_NUM=0
TOTAL_TASKS=$((${#MODEL_IDS[@]} * ${#LLMS[@]}))

for llm in "${LLMS[@]}"; do
    echo ""
    echo "========================================"
    echo "LLM: $llm"
    echo "========================================"

    for model_id in "${MODEL_IDS[@]}"; do
        TASK_NUM=$((TASK_NUM + 1))
        echo ""
        echo "[$TASK_NUM/$TOTAL_TASKS] Running: Model $model_id, LLM $llm"
        echo "----------------------------------------"

        python3 "$SCRIPT_DIR/visualize_struct.py" \
            --LLM "$llm" \
            --model_name "$model_id"

        echo "[$TASK_NUM/$TOTAL_TASKS] Completed: Model $model_id, LLM $llm"
    done
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "========================================"
echo "All tasks completed!"
echo "Total time: ${ELAPSED}s"
echo "Average time per task: $((ELAPSED / TOTAL_TASKS))s"
echo "========================================"
