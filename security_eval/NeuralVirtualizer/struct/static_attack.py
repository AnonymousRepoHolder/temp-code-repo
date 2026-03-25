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
from typing import Any, Dict, List, Optional, Set, Tuple


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..'))


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


def build_original_prediction(model_json: Dict[str, Any]) -> Dict[str, Any]:
    subgraphs = model_json.get('subgraphs', [])
    operators = subgraphs[0].get('operators', []) if subgraphs else []

    producer: Dict[int, int] = {}
    for idx, op in enumerate(operators):
        for tensor_id in op.get('outputs', []) or []:
            if isinstance(tensor_id, int):
                producer[tensor_id] = idx

    edges: Set[Tuple[int, int]] = set()
    for dst_idx, op in enumerate(operators):
        for tensor_id in op.get('inputs', []) or []:
            if isinstance(tensor_id, int) and tensor_id in producer:
                src_idx = producer[tensor_id]
                if src_idx != dst_idx:
                    edges.add((src_idx, dst_idx))

    return {
        'nodes': list(range(len(operators))),
        'edges': [{'src': src, 'dst': dst} for src, dst in sorted(edges)],
    }


def build_virtualized_prediction(v_infos: Dict[str, Any]) -> Dict[str, Any]:
    operators = v_infos.get('operators', [])

    node_ids: List[int] = []
    visible_nodes: Set[int] = set()
    for op in operators:
        idx = op.get('index')
        if isinstance(idx, int):
            node_ids.append(idx)
            visible_nodes.add(idx)

    edges: Set[Tuple[int, int]] = set()
    for op in operators:
        src_idx = op.get('index')
        if not isinstance(src_idx, int):
            continue

        for conn_data in op.get('v_forward_connections', []) or []:
            ciphertext = safe_b64decode(str(conn_data))
            candidate_dst = decode_u32_le(ciphertext)
            if candidate_dst is None:
                continue
            if candidate_dst in visible_nodes and candidate_dst != src_idx:
                edges.add((src_idx, candidate_dst))

    return {
        'nodes': sorted(node_ids),
        'edges': [{'src': src, 'dst': dst} for src, dst in sorted(edges)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Schema-aware static structure recovery attack')
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
        prediction = build_original_prediction(model_json)
        save_json(pred_path, prediction)
        print(f'Wrote prediction: {pred_path}')
        return

    v_infos_path = os.path.join(BASE_DIR, 'v_infos', f'{model_id}_v_infos.json')
    v_infos = load_json(v_infos_path)
    prediction = build_virtualized_prediction(v_infos)
    save_json(pred_path, prediction)
    print(f'Wrote prediction: {pred_path}')


if __name__ == '__main__':
    main()
