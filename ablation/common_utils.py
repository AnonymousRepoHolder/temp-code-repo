#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 The NeuralVirtualizer Authors

"""
Common utilities for ablation experiments.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from utils.utils import op_type_mapping, get_op_type_from_deprecated_builtin_code

# Model selection for ablation study
# Using 7 models to ensure sufficient cross-model collision samples
# model_0: mobilenet - mobile architecture with depthwise convolutions
# model_1: squeezenet - lightweight architecture with fire modules
# model_2: mnasnet - NAS-designed architecture
# model_3: efficientnet - efficient scaling architecture
# model_4: posenet - pose estimation model
# model_5: fruit - classification model
# model_6: skin - classification model
MODEL_NAMES = [
    'mobilenet',
    'squeezenet',
    'mnasnet',
    'efficientnet',
    'posenet',
    'fruit',
    'skin'
]
MODEL_IDS = [f'model_{i}' for i in range(len(MODEL_NAMES))]


def get_model_mapping():
    """
    Get mapping between model IDs and actual model names.

    Returns:
        dict: Mapping from model_id to model_name
    """
    return dict(zip(MODEL_IDS, MODEL_NAMES))


def get_model_id(model_name):
    """
    Get anonymized model ID from model name.

    Args:
        model_name: Actual model name (e.g., 'mobilenet')

    Returns:
        str: Anonymized model ID (e.g., 'model_0')
    """
    try:
        idx = MODEL_NAMES.index(model_name)
        return MODEL_IDS[idx]
    except ValueError:
        return None


def extract_opcodes_from_tflite(model_path):
    """
    Extract operator type codes from a TFLite model.

    Args:
        model_path: Path to the .tflite file

    Returns:
        list: List of operator type dictionaries with 'index' and 'deprecated_builtin_code'
    """
    import tensorflow as tf

    # Load interpreter
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()

    # Get model name for JSON conversion
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    # Create temporary directory for JSON conversion
    temp_dir = tempfile.mkdtemp()
    try:
        # Copy model to temp directory
        temp_model_path = os.path.join(temp_dir, f'{model_name}.tflite')
        shutil.copy(model_path, temp_model_path)

        # Convert to JSON using flatc
        schema_path = os.path.join(PROJECT_ROOT, 'schema.fbs')
        subprocess.run(
            ['flatc', '-t', schema_path, '--', temp_model_path],
            cwd=temp_dir,
            check=True,
            capture_output=True
        )

        # Read and parse JSON
        json_path = os.path.join(temp_dir, f'{model_name}.json')

        # Reduce JSON size (remove buffers)
        with open(json_path, 'r', encoding='utf-8') as f:
            lines = []
            for line in f:
                if 'buffers:' in line:
                    lines.append('}\n')
                    break
                lines.append(line)

        with open(json_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        # Repair JSON
        subprocess.run(
            ['jsonrepair', json_path, '--overwrite'],
            check=True,
            capture_output=True
        )

        # Load JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            model_json = json.load(f)

        # Extract operator codes
        operator_codes = model_json.get('operator_codes', [])
        operators = model_json['subgraphs'][0]['operators']

        # Build reverse mapping for placeholder fix
        op_name_to_code = {v: k for k, v in op_type_mapping.items()}

        # Get ops details for fallback
        try:
            ops_details = interpreter._get_ops_details()
        except Exception:
            ops_details = [{}] * len(operators)

        all_op_types = []
        for i, op in enumerate(operators):
            deprecated_builtin_code = None

            if 'opcode_index' in op:
                deprecated_builtin_code = operator_codes[op['opcode_index']].get('deprecated_builtin_code')

                # Fix placeholder (127)
                if deprecated_builtin_code == 127:
                    builtin_code_name = operator_codes[op['opcode_index']].get('builtin_code', '')
                    if builtin_code_name in op_name_to_code:
                        deprecated_builtin_code = op_name_to_code[builtin_code_name]

                # Fallback to ops_details
                if deprecated_builtin_code is None:
                    op_name = ops_details[i].get('op_name')
                    deprecated_builtin_code = op_name_to_code.get(op_name)
            else:
                op_name = ops_details[i].get('op_name')
                deprecated_builtin_code = op_name_to_code.get(op_name)

            if deprecated_builtin_code is not None:
                all_op_types.append({
                    'index': i,
                    'deprecated_builtin_code': deprecated_builtin_code
                })

        return all_op_types

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


def load_prior_knowledge():
    """
    Load common operator frequency distribution as prior knowledge.

    Returns:
        dict: Prior knowledge dictionary
    """
    return {
        "common_operator_frequencies": {
            "CONV_2D": 0.22,
            "DEPTHWISE_CONV_2D": 0.18,
            "RELU": 0.15,
            "RELU6": 0.12,
            "ADD": 0.08,
            "MUL": 0.05,
            "RESHAPE": 0.04,
            "CONCATENATION": 0.03,
            "AVERAGE_POOL_2D": 0.03,
            "MAX_POOL_2D": 0.02,
            "FULLY_CONNECTED": 0.02,
            "SOFTMAX": 0.01,
            "OTHER": 0.05
        }
    }
