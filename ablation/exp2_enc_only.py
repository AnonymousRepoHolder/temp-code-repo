#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 The NeuralVirtualizer Authors

"""
Ablation Experiment 2: Encryption Only (No Modular Arithmetic)

This experiment demonstrates that AES encryption with semantic nonces alone
is vulnerable to cross-model frequency analysis when the random factor (r_op)
from modular arithmetic is removed.

Key Insight:
    - Semantic nonce = nonce_base (8B) + op_index (8B)
    - Without r_op, same (op_index, real_code) produces same ciphertext
    - Different models using the same key will have ciphertext collisions
      at the same op_index positions if they share the same opcode

Attack Method:
    - Collect ciphertexts from multiple models encrypted with the same key
    - Analyze global ciphertext frequency distribution
    - High collision rate indicates vulnerability to frequency analysis
    - LLM infers operator types from collision patterns and context

Expected Result:
    - High global collision rate (30-60%)
    - LLM can achieve moderate accuracy (~50-65% Top-1) using frequency analysis
"""

import argparse
import base64
import json
import os
import struct
import sys

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

from common_utils import (
    SCRIPT_DIR, PROJECT_ROOT, MODEL_NAMES, MODEL_IDS,
    extract_opcodes_from_tflite, get_model_id
)
from crypto_utils import get_shared_crypto_params

sys.path.insert(0, PROJECT_ROOT)
from utils.utils import get_op_type_from_deprecated_builtin_code


def virtualize_enc_only(op_types, crypto_params):
    """
    Apply AES encryption only (no modular arithmetic).

    This simulates the scenario where encryption is used but the random
    factor r_op from modular arithmetic is removed. The plaintext is
    directly the real opcode.

    CRITICAL DIFFERENCE from full scheme:
        Full scheme:  v_code = r_op * modulus + real_code  (random each time)
        This ablation: v_code = real_code                  (deterministic)

    This means same (op_index, real_code) will always produce same ciphertext,
    enabling cross-model frequency analysis.

    Args:
        op_types: List of operator type dictionaries
        crypto_params: Cryptographic parameters

    Returns:
        List of dictionaries with encrypted opcodes
    """
    v_op_types = []

    for op in op_types:
        real_code = op['deprecated_builtin_code']

        # KEY ABLATION: No modular arithmetic, use real_code directly
        v_code = real_code  # This is the critical difference!

        # Construct 16-byte nonce: 8-byte base + 8-byte counter (semantic op index)
        nonce_counter = struct.pack('<Q', op['index'])
        nonce = crypto_params['nonce_bases']['op'] + nonce_counter

        # Create AES-256-CTR cipher
        cipher = Cipher(
            algorithms.AES(crypto_params['aes_keys']['op']),
            modes.CTR(nonce),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()

        # Encrypt the opcode (4-byte unsigned integer)
        plaintext = struct.pack('<I', v_code)
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()

        # Encode as Base64
        v_op_code_data = base64.b64encode(ciphertext[:4]).decode('ascii')

        v_op_types.append({
            'index': op['index'],
            'v_op_code_data': v_op_code_data,
            'real_code': real_code,  # Ground truth for validation
            'op_type': get_op_type_from_deprecated_builtin_code(real_code)
        })

    return v_op_types


def run_experiment(model_name, output_dir, crypto_params):
    """
    Run ablation experiment 2 for a single model.

    Args:
        model_name: Name of the model (e.g., 'mobilenet')
        output_dir: Directory to save output files
        crypto_params: Shared cryptographic parameters

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
    print(f'[Exp2] Processing {model_id} ({model_name})')

    # Extract opcodes
    print(f'  - Extracting opcodes...')
    op_types = extract_opcodes_from_tflite(model_path)
    print(f'  - Extracted {len(op_types)} operators')

    # Apply encryption only (no modular arithmetic)
    print(f'  - Applying encryption only (no modular arithmetic)...')
    v_op_types = virtualize_enc_only(op_types, crypto_params)

    # Build output structure
    output_data = {
        'model_id': model_id,
        'experiment': 'exp2_enc_only',
        'operators': v_op_types
    }

    # Save to output directory
    output_path = os.path.join(output_dir, f'{model_id}_v_infos.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    print(f'  - Saved to: {output_path}')

    # Collect statistics
    ciphertexts = [op['v_op_code_data'] for op in v_op_types]
    unique_ciphertexts = len(set(ciphertexts))

    result = {
        'model_id': model_id,
        'model_name': model_name,
        'num_operators': len(op_types),
        'unique_ciphertexts': unique_ciphertexts,
        'success': True
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Ablation Experiment 2: Encryption Only (No Modular Arithmetic)'
    )
    args = parser.parse_args()

    output_dir = os.path.join(SCRIPT_DIR, 'outputs', 'exp2')
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 70)
    print('Ablation Experiment 2: Encryption Only (No Modular Arithmetic)')
    print('=' * 70)
    print(f'Models to test: {len(MODEL_NAMES)}')
    print()

    # Generate shared crypto parameters for all models
    # IMPORTANT: All models must use the same key for cross-model analysis
    print('Loading/generating shared cryptographic parameters...')
    crypto_params = get_shared_crypto_params()
    print('  - Using shared AES-256 keys and nonce bases')
    print('  - Same keys will be used by exp3 for fair comparison')
    print()

    all_results = []
    total_operators = 0

    for model_name in MODEL_NAMES:
        result = run_experiment(model_name, output_dir, crypto_params)
        all_results.append(result)

        if result.get('success', False):
            total_operators += result.get('num_operators', 0)

        print()

    # Summary
    print('=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f'Models processed: {len(all_results)}')
    print(f'Total operators: {total_operators}')
    print()
    print('Next step: Run prepare_attack_inputs.py to generate LLM attack inputs')


if __name__ == '__main__':
    main()
