# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 The NeuralVirtualizer Authors

from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import subprocess
import sys
from typing import Any, Dict, List, Optional


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from utils.utils import op_type_mapping  # type: ignore


MODEL_ID_TO_NAME: Dict[int, str] = {
    0: 'security_test',
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


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def safe_b64decode(value: str) -> Optional[bytes]:
    try:
        return base64.b64decode(value, validate=True)
    except Exception:
        try:
            return base64.b64decode(value)
        except Exception:
            return None


def decode_u32_le(raw: Optional[bytes]) -> Optional[int]:
    if raw is None or len(raw) != 4:
        return None
    return struct.unpack('<I', raw)[0]


def model_name_from_id(model_id: str) -> str:
    model_key = int(model_id)
    if model_key not in MODEL_ID_TO_NAME:
        raise ValueError(f'Unknown model id: {model_id}')
    return MODEL_ID_TO_NAME[model_key]


def ensure_original_model_json(model_name: str) -> str:
    json_path = os.path.join(REPO_ROOT, 'security_eval', 'model_json', f'{model_name}.json')
    if os.path.exists(json_path):
        return json_path

    script_path = os.path.join(REPO_ROOT, 'security_eval', 'tflite2json.py')
    subprocess.run(
        [sys.executable, script_path, '--model_name', model_name],
        cwd=REPO_ROOT,
        check=True,
    )
    return json_path


def extract_original_truth_rows(model_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    subgraphs = model_json.get('subgraphs', [])
    operators = subgraphs[0].get('operators', []) if subgraphs else []
    operator_codes = model_json.get('operator_codes', [])

    rows: List[Dict[str, Any]] = []
    for idx, op in enumerate(operators):
        op_type = 'UNKNOWN'
        opcode_index = op.get('opcode_index')
        if isinstance(opcode_index, int) and 0 <= opcode_index < len(operator_codes):
            code_entry = operator_codes[opcode_index] or {}
            code = code_entry.get('deprecated_builtin_code')
            if code is None:
                code = code_entry.get('builtin_code')
            if isinstance(code, int):
                op_type = op_type_mapping.get(code, f'UNKNOWN_OP_{code}')
        rows.append({'idx': idx, 'type': op_type})
    return rows


def build_predict_rows_from_truth(truth_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            'idx': row['idx'],
            'topk': [{'type': row['type'], 'p': 1.0}],
        }
        for row in truth_rows
    ]


def build_virtualized_predict_rows(v_infos: Dict[str, Any]) -> List[Dict[str, Any]]:
    operators = v_infos.get('operators', [])
    rows: List[Dict[str, Any]] = []

    for op in operators:
        idx = op.get('index')
        pred_type = 'UNKNOWN'
        if idx is None:
            continue

        ciphertext = safe_b64decode(str(op.get('v_op_code_data', '')))
        ciphertext_u32 = decode_u32_le(ciphertext)
        if ciphertext_u32 is not None:
            pred_type = op_type_mapping.get(ciphertext_u32, 'UNKNOWN')

        rows.append({
            'idx': idx,
            'topk': [{'type': pred_type, 'p': 1.0}],
        })

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description='Schema-aware static operator recovery attack')
    parser.add_argument('--model_name', required=True, help='Numeric model id, e.g., 1')
    parser.add_argument(
        '--artifact',
        required=True,
        choices=['original', 'virtualized'],
        help='Artifact type to attack',
    )
    parser.add_argument(
        '--tag',
        default=None,
        help='Prediction subdirectory name under predict/. Default: static_<artifact>',
    )
    args = parser.parse_args()

    model_id = str(args.model_name)
    tag = args.tag or f'static_{args.artifact}'
    pred_path = os.path.join(BASE_DIR, 'predict', tag, f'{model_id}_predict.json')

    if args.artifact == 'original':
        model_name = model_name_from_id(model_id)
        model_json_path = ensure_original_model_json(model_name)
        model_json = load_json(model_json_path)

        truth_rows = extract_original_truth_rows(model_json)
        predict_rows = build_predict_rows_from_truth(truth_rows)

        truth_path = os.path.join(BASE_DIR, 'real_original', f'{model_id}_real.json')
        save_json(truth_path, truth_rows)
        save_json(pred_path, predict_rows)
        print(f'Wrote truth: {truth_path}')
        print(f'Wrote prediction: {pred_path}')
        return

    v_infos_path = os.path.join(BASE_DIR, 'v_infos', f'{model_id}_v_infos.json')
    v_infos = load_json(v_infos_path)
    predict_rows = build_virtualized_predict_rows(v_infos)
    save_json(pred_path, predict_rows)
    print(f'Wrote prediction: {pred_path}')


if __name__ == '__main__':
    main()
