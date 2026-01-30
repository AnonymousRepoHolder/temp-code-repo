#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 The NeuralVirtualizer Authors

"""
Prepare attack input files for LLM-based attacks on ablation experiments.

This script generates attack input files that will be provided to LLMs
for operator type inference. Each experiment has different attack scenarios:

- Exp1: Modulus brute-force attack (opcode directly recoverable)
- Exp2: Cross-model frequency analysis (high collision rate)
- Exp3: Cross-model frequency analysis (low collision rate due to modular arithmetic)

IMPORTANT: Attack inputs contain ONLY opcode information (index + v_code/cipher).
No shape, input/output, or other context information is provided.
This ensures the ablation study accurately measures the contribution of
each protection mechanism (modular arithmetic vs encryption).
"""

import json
import os
import sys
from collections import Counter, defaultdict

from common_utils import (
    SCRIPT_DIR, PROJECT_ROOT, MODEL_NAMES, MODEL_IDS,
    get_model_id, load_prior_knowledge
)

sys.path.insert(0, PROJECT_ROOT)
from utils.utils import op_type_mapping


def generate_opcode_mapping():
    """
    Generate opcode to operator type mapping file for Exp1.

    This file provides the mapping from TFLite opcode (deprecated_builtin_code)
    to operator type name, which is needed for the modulus brute-force attack.
    """
    print('Generating opcode mapping file...')

    # op_type_mapping is {opcode: op_type_name}
    mapping = {
        'description': 'TFLite opcode to operator type mapping',
        'note': 'opcode is the deprecated_builtin_code in TFLite schema',
        'mapping': {str(k): v for k, v in sorted(op_type_mapping.items())}
    }

    output_path = os.path.join(SCRIPT_DIR, 'attack_inputs', 'opcode_mapping.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2)

    print(f'  Saved to: {output_path}')
    return True


def prepare_exp1_input():
    """
    Prepare attack input for Exp1 (modular arithmetic only).

    Attack scenario: Attacker has brute-forced the modulus and can
    compute real_code = v_code % modulus. The real_code directly
    maps to operator type via opcode lookup table.

    Input contains ONLY: index, v_code, modulus, prior_knowledge
    """
    print('Preparing Exp1 attack input...')

    exp1_dir = os.path.join(SCRIPT_DIR, 'outputs', 'exp1')

    # Load brute-force results
    brute_force_path = os.path.join(exp1_dir, 'brute_force_results.json')
    with open(brute_force_path, 'r', encoding='utf-8') as f:
        brute_force_result = json.load(f)

    if not brute_force_result.get('success', False):
        print('  Error: Brute-force attack failed, cannot prepare input')
        return False

    # Build attack input (NO shape information)
    attack_input = {
        'experiment': 'exp1_mod_only',
        'attack_context': {
            'protection_removed': 'Encryption (AES-256-CTR)',
            'protection_remaining': 'Modular arithmetic virtualization',
            'attacker_knowledge': {
                'modulus': brute_force_result['top_candidate'],
                'modular_arithmetic_rule': 'real_code = v_code % modulus, where real_code in [0, 189]',
                'note': 'Attacker has brute-forced the modulus. Compute real_code = v_code % modulus, then lookup operator type from opcode.'
            }
        },
        'prior_knowledge': load_prior_knowledge(),
        'models': []
    }

    # Load each model's data (ONLY index and v_code)
    for model_id in MODEL_IDS:
        v_codes_path = os.path.join(exp1_dir, f'{model_id}_v_codes.json')
        with open(v_codes_path, 'r', encoding='utf-8') as f:
            v_codes_data = json.load(f)

        # Extract ONLY index and v_code
        operators = [
            {
                'index': op['index'],
                'v_code': op['v_code']
            }
            for op in v_codes_data['operators']
        ]

        attack_input['models'].append({
            'model_id': model_id,
            'num_operators': len(operators),
            'operators': operators
        })

    # Save attack input
    output_path = os.path.join(SCRIPT_DIR, 'attack_inputs', 'exp1_attack_input.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(attack_input, f, indent=2)

    print(f'  Saved to: {output_path}')
    return True


def analyze_collision_patterns(models_data):
    """
    Analyze ciphertext collision patterns across models.

    Args:
        models_data: List of model data dictionaries (containing only index and cipher)

    Returns:
        dict: Collision analysis results (NO shape information)
    """
    # Collect all ciphertexts
    cipher_to_positions = defaultdict(list)
    all_ciphertexts = []

    for model_data in models_data:
        model_id = model_data['model_id']
        for op in model_data['operators']:
            cipher = op['v_op_code_data']
            all_ciphertexts.append(cipher)
            cipher_to_positions[cipher].append({
                'model_id': model_id,
                'op_index': op['index']
            })

    # Calculate collision statistics
    cipher_freq = Counter(all_ciphertexts)
    total = len(all_ciphertexts)
    unique = len(cipher_freq)

    # Extract high-frequency ciphertexts (top 20)
    high_freq_ciphers = []
    for cipher, count in cipher_freq.most_common(20):
        positions = cipher_to_positions[cipher]

        high_freq_ciphers.append({
            'cipher': cipher,
            'occurrence_count': count,
            'occurrence_percentage': round(count / total * 100, 2),
            'appears_in_models': list(set(p['model_id'] for p in positions)),
            'positions': positions  # Only model_id and op_index
        })

    return {
        'total_ciphertexts': total,
        'unique_ciphertexts': unique,
        'collision_rate': round(1 - unique / total, 4) if total > 0 else 0,
        'num_models': len(models_data),
        'high_frequency_ciphertexts': high_freq_ciphers
    }


def prepare_exp2_input():
    """
    Prepare attack input for Exp2 (encryption only, no modular arithmetic).

    Attack scenario: Cross-model frequency analysis. Without modular arithmetic,
    same opcode at same position produces same ciphertext across models.
    High collision rate enables frequency-based inference.

    Input contains ONLY: index, v_op_code_data, collision_analysis, prior_knowledge
    """
    print('Preparing Exp2 attack input...')

    exp2_dir = os.path.join(SCRIPT_DIR, 'outputs', 'exp2')

    # Load all models' data (ONLY index and cipher)
    models_data = []
    for model_id in MODEL_IDS:
        v_infos_path = os.path.join(exp2_dir, f'{model_id}_v_infos.json')
        with open(v_infos_path, 'r', encoding='utf-8') as f:
            v_infos_data = json.load(f)

        # Extract ONLY index and v_op_code_data
        operators = [
            {
                'index': op['index'],
                'v_op_code_data': op['v_op_code_data']
            }
            for op in v_infos_data['operators']
        ]

        models_data.append({
            'model_id': model_id,
            'num_operators': len(operators),
            'operators': operators
        })

    # Analyze collision patterns
    collision_analysis = analyze_collision_patterns(models_data)

    # Build attack input (NO shape information)
    attack_input = {
        'experiment': 'exp2_enc_only',
        'attack_context': {
            'protection_removed': 'Modular arithmetic randomization (r_op factor)',
            'protection_remaining': 'AES-256-CTR encryption',
            'vulnerability': 'Same opcode at same position produces SAME ciphertext across models',
            'attacker_strategy': 'Cross-model frequency analysis: high-frequency ciphertexts likely correspond to common operators'
        },
        'global_collision_analysis': collision_analysis,
        'prior_knowledge': load_prior_knowledge(),
        'models': models_data
    }

    # Save attack input
    output_path = os.path.join(SCRIPT_DIR, 'attack_inputs', 'exp2_attack_input.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(attack_input, f, indent=2)

    print(f'  Saved to: {output_path}')
    print(f'  Collision rate: {collision_analysis["collision_rate"]*100:.1f}%')
    return True


def prepare_exp3_input():
    """
    Prepare attack input for Exp3 (full protection: modular arithmetic + encryption).

    Attack scenario: Same as Exp2 (frequency analysis), but with full protection.
    The random r_op factor breaks ciphertext correlation, resulting in very low
    collision rate. Frequency analysis becomes ineffective.

    Input format is IDENTICAL to Exp2 for fair comparison.
    """
    print('Preparing Exp3 attack input...')

    exp3_dir = os.path.join(SCRIPT_DIR, 'outputs', 'exp3')

    # Load all models' data (ONLY index and cipher)
    models_data = []
    for model_id in MODEL_IDS:
        v_infos_path = os.path.join(exp3_dir, f'{model_id}_v_infos.json')
        with open(v_infos_path, 'r', encoding='utf-8') as f:
            v_infos_data = json.load(f)

        # Extract ONLY index and v_op_code_data
        operators = [
            {
                'index': op['index'],
                'v_op_code_data': op['v_op_code_data']
            }
            for op in v_infos_data['operators']
        ]

        models_data.append({
            'model_id': model_id,
            'num_operators': len(operators),
            'operators': operators
        })

    # Analyze collision patterns (same method as Exp2 for fair comparison)
    collision_analysis = analyze_collision_patterns(models_data)

    # Build attack input (same format as Exp2)
    attack_input = {
        'experiment': 'exp3_full_protection',
        'attack_context': {
            'protection_removed': 'None (full protection enabled)',
            'protection_remaining': 'Modular arithmetic + AES-256-CTR encryption',
            'vulnerability': 'Same opcode at same position produces SAME ciphertext across models',
            'attacker_strategy': 'Cross-model frequency analysis: high-frequency ciphertexts likely correspond to common operators'
        },
        'global_collision_analysis': collision_analysis,
        'prior_knowledge': load_prior_knowledge(),
        'models': models_data
    }

    # Save attack input
    output_path = os.path.join(SCRIPT_DIR, 'attack_inputs', 'exp3_attack_input.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(attack_input, f, indent=2)

    print(f'  Saved to: {output_path}')
    print(f'  Collision rate: {collision_analysis["collision_rate"]*100:.1f}%')
    return True


def prepare_ground_truth():
    """
    Prepare ground truth files for evaluation.
    """
    print('Preparing ground truth files...')

    ground_truth_dir = os.path.join(SCRIPT_DIR, 'ground_truth')
    os.makedirs(ground_truth_dir, exist_ok=True)

    # Use exp1 outputs as ground truth source (they have real_code and op_type)
    exp1_dir = os.path.join(SCRIPT_DIR, 'outputs', 'exp1')

    for model_id in MODEL_IDS:
        v_codes_path = os.path.join(exp1_dir, f'{model_id}_v_codes.json')
        with open(v_codes_path, 'r', encoding='utf-8') as f:
            v_codes_data = json.load(f)

        # Build ground truth
        ground_truth = [
            {
                'idx': i,
                'type': op['op_type']
            }
            for i, op in enumerate(v_codes_data['operators'])
        ]

        # Save
        output_path = os.path.join(ground_truth_dir, f'{model_id}_real.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ground_truth, f, indent=2)

    print(f'  Saved {len(MODEL_IDS)} ground truth files to: {ground_truth_dir}')
    return True


def main():
    print('=' * 70)
    print('Preparing Attack Inputs for Ablation Experiments')
    print('=' * 70)
    print()

    # Generate opcode mapping (needed for Exp1)
    success_mapping = generate_opcode_mapping()
    print()

    # Prepare inputs for each experiment
    success_exp1 = prepare_exp1_input()
    print()

    success_exp2 = prepare_exp2_input()
    print()

    success_exp3 = prepare_exp3_input()
    print()

    # Prepare ground truth
    success_gt = prepare_ground_truth()
    print()

    # Summary
    print('=' * 70)
    print('SUMMARY')
    print('=' * 70)
    if all([success_mapping, success_exp1, success_exp2, success_exp3, success_gt]):
        print('All attack inputs and ground truth files prepared successfully!')
        print()
        print('Next steps:')
        print('1. Provide EXP1-ATTACK-Prompt.md to LLM')
        print('   - LLM reads: attack_inputs/exp1_attack_input.json')
        print('   - LLM reads: attack_inputs/opcode_mapping.json')
        print('   - LLM writes: attack_outputs/exp1/model_*_predict.json')
        print()
        print('2. Provide EXP2-EXP3-ATTACK-Prompt.md to LLM')
        print('   - For Exp2: LLM reads attack_inputs/exp2_attack_input.json')
        print('   - For Exp3: LLM reads attack_inputs/exp3_attack_input.json')
        print('   - LLM writes: attack_outputs/exp2|exp3/model_*_predict.json')
        print()
        print('3. Run evaluate_ablation.py to evaluate results')
    else:
        print('Some steps failed. Please check the error messages above.')


if __name__ == '__main__':
    main()
