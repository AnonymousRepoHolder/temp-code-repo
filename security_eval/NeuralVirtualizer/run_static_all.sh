#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MODEL_IDS=(1 2 3 4 5 6 7 8 9 10)
MODEL_NAMES=(squeezenet posenet fruit lenet mobilenet skin mnasnet efficientnet ssd depth_estimation)

for model_name in "${MODEL_NAMES[@]}"; do
  python3 security_eval/tflite2json.py --model_name "${model_name}"
done

for model_id in "${MODEL_IDS[@]}"; do
  python3 security_eval/NeuralVirtualizer/opTypes/static_attack.py \
    --model_name "${model_id}" \
    --artifact original \
    --tag static_original
  python3 security_eval/NeuralVirtualizer/opTypes/compare_eval.py \
    --model_name "${model_id}" \
    --LLM static_original \
    --real_subdir real_original

  python3 security_eval/NeuralVirtualizer/struct/static_attack.py \
    --model_name "${model_id}" \
    --artifact original \
    --tag static_original
  python3 security_eval/NeuralVirtualizer/struct/visualize_struct.py \
    --model_name "${model_id}" \
    --LLM static_original

  python3 security_eval/NeuralVirtualizer/opTypes/static_attack.py \
    --model_name "${model_id}" \
    --artifact virtualized \
    --tag static_virtualized
  python3 security_eval/NeuralVirtualizer/opTypes/compare_eval.py \
    --model_name "${model_id}" \
    --LLM static_virtualized \
    --real_subdir real

  python3 security_eval/NeuralVirtualizer/struct/static_attack.py \
    --model_name "${model_id}" \
    --artifact virtualized \
    --tag static_virtualized
  python3 security_eval/NeuralVirtualizer/struct/visualize_struct.py \
    --model_name "${model_id}" \
    --LLM static_virtualized
done

python3 security_eval/NeuralVirtualizer/summarize_static_results.py \
  --orig_tag static_original \
  --virt_tag static_virtualized
