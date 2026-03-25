# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 The NeuralVirtualizer Authors

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


MODEL_ID_TO_NAME: Dict[int, str] = {
    1: 'squeezenet',
    2: 'posenet',
    3: 'fruit',
    4: 'lenet',
    5: 'mobilenet',
    6: 'skin',
    7: 'mnasnet',
    8: 'efficientnet',
    9: 'ssd',
    10: 'depth_estimation',
}


def load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_model_ids(model_ids_arg: str | None) -> List[int]:
    if not model_ids_arg:
        return list(MODEL_ID_TO_NAME.keys())

    model_ids: List[int] = []
    for raw_part in model_ids_arg.split(','):
        part = raw_part.strip()
        if not part:
            continue
        model_id = int(part)
        if model_id not in MODEL_ID_TO_NAME:
            raise ValueError(f'Unknown model id in --model_ids: {model_id}')
        model_ids.append(model_id)

    if not model_ids:
        raise ValueError('--model_ids is empty after parsing')
    return model_ids


def find_ground_truth_similarity(similarity_data: Dict[str, Any], model_name: str) -> float:
    for row in similarity_data.get('ranking', []):
        if row.get('model_name') == model_name:
            return float(row.get('similarity', 0.0))
    return 0.0


def summarize_tag(tag: str, model_ids: List[int]) -> Dict[str, Any]:
    per_model: List[Dict[str, Any]] = []

    for model_id in model_ids:
        model_name = MODEL_ID_TO_NAME[model_id]
        op_eval_path = os.path.join(
            BASE_DIR,
            'opTypes',
            'eval',
            tag,
            f'{model_id}_{model_name}_eval.json',
        )
        struct_eval_path = os.path.join(
            BASE_DIR,
            'struct',
            'eval',
            tag,
            f'{model_id}_similarity.json',
        )

        op_eval = load_json(op_eval_path)
        struct_eval = load_json(struct_eval_path)

        operator_recovery = float(op_eval['metrics']['top1_acc'])
        structure_recovery = find_ground_truth_similarity(struct_eval, model_name)
        static_reconstruction = (
            operator_recovery >= 0.999999 and structure_recovery >= 0.999999
        )

        per_model.append({
            'model_id': model_id,
            'model_name': model_name,
            'operator_recovery': round(operator_recovery, 6),
            'structure_recovery': round(structure_recovery, 6),
            'static_reconstruction': static_reconstruction,
        })

    count = len(per_model)
    op_mean = sum(row['operator_recovery'] for row in per_model) / count if count else 0.0
    struct_mean = sum(row['structure_recovery'] for row in per_model) / count if count else 0.0
    reconstruction_successes = sum(1 for row in per_model if row['static_reconstruction'])

    return {
        'per_model': per_model,
        'aggregate': {
            'mean_operator_recovery': round(op_mean, 6),
            'mean_structure_recovery': round(struct_mean, 6),
            'static_reconstruction_successes': reconstruction_successes,
            'static_reconstruction_total': count,
            'static_reconstruction_ratio': f'{reconstruction_successes}/{count}',
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarize static reverse-engineering results')
    parser.add_argument('--orig_tag', default='static_original', help='Tag for original .tflite results')
    parser.add_argument('--virt_tag', default='static_virtualized', help='Tag for virtualized artifact results')
    parser.add_argument(
        '--model_ids',
        default=None,
        help='Comma-separated model ids to summarize, e.g., 5 or 1,2,3. Default: all 10 models.',
    )
    parser.add_argument(
        '--output',
        default=os.path.join(BASE_DIR, 'static_summary.json'),
        help='Output summary JSON path',
    )
    args = parser.parse_args()

    model_ids = parse_model_ids(args.model_ids)
    original = summarize_tag(args.orig_tag, model_ids)
    virtualized = summarize_tag(args.virt_tag, model_ids)

    summary = {
        'original': original,
        'virtualized': virtualized,
        'paper_table': [
            {
                'artifact': 'Original .tflite',
                'operator_recovery': original['aggregate']['mean_operator_recovery'],
                'structure_recovery': original['aggregate']['mean_structure_recovery'],
                'static_reconstruction': original['aggregate']['static_reconstruction_ratio'],
            },
            {
                'artifact': 'NeuralVirtualizer artifacts',
                'operator_recovery': virtualized['aggregate']['mean_operator_recovery'],
                'structure_recovery': virtualized['aggregate']['mean_structure_recovery'],
                'static_reconstruction': virtualized['aggregate']['static_reconstruction_ratio'],
            },
        ],
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f'Wrote summary: {args.output}')


if __name__ == '__main__':
    main()
