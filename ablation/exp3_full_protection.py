#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 The NeuralVirtualizer Authors

"""
Ablation Experiment 3: Full Protection (Modular Arithmetic + Encryption)

This experiment demonstrates that the combination of modular arithmetic
and AES encryption effectively prevents cross-model frequency analysis.

Key Insight:
    - Full scheme: v_code = r_op * modulus + real_code
    - r_op is random for each encryption, even for the same opcode
    - This breaks the deterministic relationship between opcode and ciphertext
    - Cross-model collision rate should be very low (<1%)

This serves as the baseline for comparison with exp1 and exp2.

Expected Result:
    - Very low global collision rate (<1%)
    - Frequency analysis attack is ineffective
    - LLM can only achieve low accuracy (~20-30% Top-1) using local signals
"""

import argparse
import base64
import json
import os
import random
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


def virtualize_full_scheme(op_types, crypto_params):
    """
    Apply full virtualization scheme (modular arithmetic + encryption).

    This is the complete protection mechanism:
        v_code = r_op * modulus + real_code
        ciphertext = AES_CTR(v_code, key, nonce)

    The random factor r_op ensures that even the same opcode at the same
    position will produce different ciphertexts across different models
    (or even different encryptions of the same model).

    Args:
        op_types: List of operator type dictionaries
        crypto_params: Cryptographic parameters

    Returns:
        List of dictionaries with encrypted opcodes
    """
    v_op_types = []
    op_modulus = crypto_params['op_modulus']

    for op in op_types:
        real_code = op['deprecated_builtin_code']

        # FULL SCHEME: Apply modular arithmetic with random factor
        r_op = random.randint(1, 10000)
        v_code = r_op * op_modulus + real_code

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

        # Encrypt the virtualized code (4-byte unsigned integer)
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
    Run ablation experiment 3 for a single model.

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
    print(f'[Exp3] Processing {model_id} ({model_name})')

    # Extract opcodes
    print(f'  - Extracting opcodes...')
    op_types = extract_opcodes_from_tflite(model_path)
    print(f'  - Extracted {len(op_types)} operators')

    # Apply full protection (modular arithmetic + encryption)
    print(f'  - Applying full protection (modular arithmetic + encryption)...')
    v_op_types = virtualize_full_scheme(op_types, crypto_params)

    # Build output structure
    output_data = {
        'model_id': model_id,
        'experiment': 'exp3_full_protection',
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
        description='Ablation Experiment 3: Full Protection (Modular Arithmetic + Encryption)'
    )
    args = parser.parse_args()

    output_dir = os.path.join(SCRIPT_DIR, 'outputs', 'exp3')
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 70)
    print('Ablation Experiment 3: Full Protection (Modular Arithmetic + Encryption)')
    print('=' * 70)
    print(f'Models to test: {len(MODEL_NAMES)}')
    print()

    # Generate shared crypto parameters for all models
    # IMPORTANT: Use same keys as exp2 for fair comparison
    print('Loading/generating shared cryptographic parameters...')
    crypto_params = get_shared_crypto_params()
    print('  - Using shared AES-256 keys and nonce bases')
    print('  - Same keys as exp2 for fair comparison')
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
