#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 The NeuralVirtualizer Authors

"""
Ablation Experiment 1: Modular Arithmetic Only (No Encryption)

This experiment demonstrates that modular arithmetic alone is insufficient
for protecting operator type codes. An attacker can brute-force search for
the modulus value and recover all original opcodes.

Attack Method:
    - Iterate candidate modulus values from 190 to 1000
    - For each candidate, compute recovered_code = v_code % candidate
    - If all recovered codes fall within valid TFLite opcode range [0, 189],
      the candidate is likely the correct modulus
    - Use heuristic scoring to rank candidates

Expected Result:
    - Attack success rate: ~100%
    - Attack time: < 1 second
    - Combined with other signals (shapes, builtin_options), LLM can achieve
      high accuracy (~75-85% Top-1)
"""

import argparse
import json
import os
import random
import sys

from common_utils import (
    SCRIPT_DIR, PROJECT_ROOT, MODEL_NAMES, MODEL_IDS,
    extract_opcodes_from_tflite, get_model_id
)

sys.path.insert(0, PROJECT_ROOT)
from utils.utils import get_op_type_from_deprecated_builtin_code

# Modulus value (same as full system)
OP_MODULUS = 251


def virtualize_mod_only(op_types, op_modulus=OP_MODULUS):
    """
    Apply modular arithmetic virtualization only (no encryption).

    This simulates the scenario where only modular arithmetic is used
    for protection, without AES encryption.

    Args:
        op_types: List of operator type dictionaries with 'deprecated_builtin_code'
        op_modulus: Modulus value for virtualization (default: 251)

    Returns:
        List of dictionaries with virtualized codes (plaintext)
    """
    v_codes = []
    for op in op_types:
        real_code = op['deprecated_builtin_code']
        # Apply modular arithmetic: v_code = r * modulus + real_code
        r_op = random.randint(1, 10000)
        v_code = r_op * op_modulus + real_code

        v_codes.append({
            'index': op['index'],
            'v_code': v_code,           # Plaintext virtualized code
            'real_code': real_code,     # Ground truth (for validation)
            'op_type': get_op_type_from_deprecated_builtin_code(real_code)
        })

    return v_codes


def brute_force_modulus(v_codes, valid_range=(0, 189)):
    """
    Brute-force search for the modulus value with heuristic scoring.

    Attack strategy:
        1. Find all candidate moduli that produce valid opcodes (in range [0, 189])
        2. Score each candidate based on opcode distribution plausibility
           - Higher score for common opcodes (CONV_2D, RELU, ADD, etc.)
           - Penalize extreme unique opcode counts (typical models have 5-15 types)
        3. Select the highest-scoring candidate as the attack result

    Args:
        v_codes: List of virtualized code dictionaries with 'v_code' field
        valid_range: Tuple of (min_valid_opcode, max_valid_opcode)

    Returns:
        dict: Attack results including top candidates
    """
    # Prior knowledge: common opcodes in typical DNN models
    COMMON_OPCODES = {
        3,   # CONV_2D
        4,   # DEPTHWISE_CONV_2D
        19,  # RELU
        21,  # RELU6
        0,   # ADD
        18,  # MUL
        9,   # FULLY_CONNECTED
        25,  # SOFTMAX
        22,  # RESHAPE
        17,  # MAX_POOL_2D
        1,   # AVERAGE_POOL_2D
        2,   # CONCATENATION
        40,  # MEAN
        41,  # SUB
    }

    v_code_values = [item['v_code'] for item in v_codes]
    candidates = []

    # Phase 1: Find all valid candidates
    for candidate_modulus in range(valid_range[1] + 1, 1000):
        recovered = [v % candidate_modulus for v in v_code_values]

        # Check if all recovered codes are within valid range
        if not all(valid_range[0] <= r <= valid_range[1] for r in recovered):
            continue

        # Phase 2: Calculate heuristic score
        unique_opcodes = set(recovered)

        # Score component 1: Common opcode ratio (0.0 - 1.0)
        common_count = sum(1 for r in recovered if r in COMMON_OPCODES)
        common_ratio = common_count / len(recovered) if recovered else 0.0

        # Score component 2: Unique opcode count penalty
        optimal_unique = 10
        unique_penalty = min(abs(len(unique_opcodes) - optimal_unique) / 20, 0.5)

        # Score component 3: Distribution concentration
        from collections import Counter
        opcode_freq = Counter(recovered)
        if opcode_freq:
            max_freq = max(opcode_freq.values())
            concentration = max_freq / len(recovered)
        else:
            concentration = 0.0

        # Combined score
        score = common_ratio * 0.6 + concentration * 0.2 - unique_penalty * 0.2

        candidates.append({
            'modulus': candidate_modulus,
            'score': score,
            'common_ratio': common_ratio,
            'unique_count': len(unique_opcodes),
            'concentration': concentration
        })

    if not candidates:
        return {
            'success': False,
            'top_candidates': []
        }

    # Phase 3: Select top candidates by score
    candidates.sort(key=lambda x: x['score'], reverse=True)

    # Build top candidate (top 1 only)
    top_candidate = {
        'modulus': candidates[0]['modulus'],
        'score': round(candidates[0]['score'], 4),
        'common_ratio': round(candidates[0]['common_ratio'], 4),
        'unique_count': candidates[0]['unique_count']
    }

    return {
        'success': True,
        'top_candidate': top_candidate,
        'total_candidates': len(candidates)
    }


def run_experiment(model_name, output_dir):
    """
    Run ablation experiment 1 for a single model.

    Args:
        model_name: Name of the model (e.g., 'mobilenet')
        output_dir: Directory to save output files

    Returns:
        dict: Experiment results
    """
    model_path = os.path.join(PROJECT_ROOT, 'tflite_model', f'{model_name}.tflite')

    if not os.path.exists(model_path):
        return {
            'model_name': model_name,
            'error': f'Model file not found: {model_path}',
            'success': False
        }

    model_id = get_model_id(model_name)
    print(f'[Exp1] Processing {model_id} ({model_name})')

    # Extract opcodes
    print(f'  - Extracting opcodes...')
    op_types = extract_opcodes_from_tflite(model_path)
    print(f'  - Extracted {len(op_types)} operators')

    # Apply modular arithmetic only
    print(f'  - Applying modular arithmetic (modulus={OP_MODULUS})...')
    v_codes = virtualize_mod_only(op_types, OP_MODULUS)

    # Save virtualized codes
    output_data = {
        'model_id': model_id,
        'experiment': 'exp1_mod_only',
        'operators': v_codes
    }

    output_path = os.path.join(output_dir, f'{model_id}_v_codes.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    print(f'  - Saved to: {output_path}')

    return {
        'model_id': model_id,
        'model_name': model_name,
        'num_operators': len(op_types),
        'success': True
    }


def main():
    parser = argparse.ArgumentParser(
        description='Ablation Experiment 1: Modular Arithmetic Only'
    )
    args = parser.parse_args()

    output_dir = os.path.join(SCRIPT_DIR, 'outputs', 'exp1')
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 70)
    print('Ablation Experiment 1: Modular Arithmetic Only (No Encryption)')
    print('=' * 70)
    print(f'Models to test: {len(MODEL_NAMES)}')
    print(f'Modulus: {OP_MODULUS}')
    print()

    all_results = []
    all_v_codes = []

    for model_name in MODEL_NAMES:
        result = run_experiment(model_name, output_dir)
        all_results.append(result)

        if result.get('success', False):
            # Load v_codes for brute-force attack
            model_id = result['model_id']
            v_codes_path = os.path.join(output_dir, f'{model_id}_v_codes.json')
            with open(v_codes_path, 'r', encoding='utf-8') as f:
                v_codes_data = json.load(f)
                all_v_codes.extend(v_codes_data['operators'])

        print()

    # Run brute-force attack on all v_codes
    print('Running brute-force modulus search...')
    attack_result = brute_force_modulus(all_v_codes)

    if attack_result['success']:
        print(f'  - Found {attack_result["total_candidates"]} valid candidates')
        print(f'  - Best candidate: modulus={attack_result["top_candidate"]["modulus"]} '
              f'(score={attack_result["top_candidate"]["score"]:.4f})')
    else:
        print(f'  - Attack failed (no valid candidates found)')

    # Save brute-force results
    brute_force_path = os.path.join(output_dir, 'brute_force_results.json')
    with open(brute_force_path, 'w', encoding='utf-8') as f:
        json.dump(attack_result, f, indent=2)
    print(f'  - Saved brute-force results to: {brute_force_path}')

    # Summary
    print()
    print('=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f'Models processed: {len(all_results)}')
    print(f'Total operators: {sum(r.get("num_operators", 0) for r in all_results)}')
    print()
    print('Next step: Run prepare_attack_inputs.py to generate LLM attack inputs')


if __name__ == '__main__':
    main()
