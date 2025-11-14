# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 The NeuralVirtualizer Authors
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import random
import numpy as np
import json
import struct
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

from utils.utils import extract_all_parameters, convert_dtype_for_json


# Encrypt parameter position and length information using AES-256-CTR
def virtualize_params(params_info, total_param_size, crypto_params, all_op_indices=None):
    """
    Encrypt parameter position/length and append Phase 3 dummies to ensure each
    operator has at least two parameter entries.

    Args:
        params_info: list of real parameter descriptors extracted from the model
        total_param_size: total number of elements in params.bin (for bounds)
        crypto_params: AES keys and nonce bases
        all_op_indices: optional iterable of all operator indices to cover ops
                        with zero real params (recommended). When None, only
                        operators appearing in params_info are considered.
    """
    v_params_info = []

    # Phase 3: ensure at least 2 parameter entries per operator by appending dummy entries
    # Build per-op aggregation to count real entries and plan dummy additions
    per_op = {}
    for p in params_info:
        per_op.setdefault(p['op_index'], []).append(p)

    # Determine the operator set to cover; fallback to existing keys if not provided
    if all_op_indices is None:
        op_set = set(per_op.keys())
    else:
        op_set = set(all_op_indices)

    # Compose a new list with dummies appended where needed (0->+2, 1->+1)
    augmented_params = list(params_info)
    for op_index in op_set:
        real_count = len(per_op.get(op_index, []))
        if real_count < 2:
            need = 2 - real_count
            for _ in range(need):
                # Generate dummy start_pos/length within valid bounds to avoid accidental OOB
                if total_param_size > 0:
                    start_pos = random.randint(0, max(total_param_size - 1, 0))
                    max_len = min(1024, max(total_param_size - start_pos, 1))
                    length = random.randint(1, max_len)
                else:
                    start_pos = 0
                    length = 1
                # Reserve slot value -1 (encoded unsigned for counter construction later)
                dummy = {
                    'index': -1,                 # no backing tensor index
                    'op_index': op_index,
                    'start_pos': start_pos,
                    'length': length,
                    'shape': [],                 # num_dims==0 is carried in encrypted v_shape elsewhere
                    'dtype': 'float32',
                    'input_slot': -1             # reserved slot for dummy
                }
                augmented_params.append(dummy)

    # Encrypt each parameter's position information (including dummies)
    for param in augmented_params:
        # Construct 16-byte nonce: 8-byte base + 8-byte counter (using semantic identifiers)
        # Combined counter = (op_index << 32) | input_slot
        op_u32 = (int(param['op_index']) & 0xFFFFFFFF)
        slot_u32 = (int(param.get('input_slot', 0)) & 0xFFFFFFFF)
        combined_counter = (op_u32 << 32) | slot_u32
        nonce_counter = struct.pack('<Q', combined_counter)
        nonce = crypto_params['nonce_bases']['param'] + nonce_counter  # Total: 16 bytes

        # Create AES-256-CTR cipher using plaintext param key
        cipher = Cipher(
            algorithms.AES(crypto_params['aes_keys']['param']),
            modes.CTR(nonce),  # Use 16-byte nonce directly
            backend=default_backend()
        )
        encryptor = cipher.encryptor()

        # Pack start_pos and length as 16-byte plaintext (2 x uint64)
        plaintext = struct.pack('<QQ', param['start_pos'], param['length'])

        # Encrypt
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()

        # Encode ciphertext as Base64 to avoid JSON integer precision issues
        v_position_data = base64.b64encode(ciphertext).decode('ascii')

        v_params_info.append({
            'index': param['index'],
            'op_index': param['op_index'],
            'v_position_data': v_position_data,  # Base64-encoded ciphertext
            'shape': param['shape'],
            'dtype': param['dtype'],
            # Pass through input slot indices to restore the original parameter order in inputs
            # such as SUB or BMM
            'input_slot': param.get('input_slot', None)
        })
    return v_params_info


# Aggregate all original parameters into a single data file
def generate_params_file(all_params, params_info, model_name):
    """
    Build params.bin from the extracted parameter elements.

    Minimal-intrusive deduplication is applied: multiple tensors that share the
    same underlying segment (same start_pos/length in element units) are written
    only once. Subsequent tensors reuse the same start_pos/length and do not
    append duplicate bytes. This keeps file layout consistent with references.
    """
    param_bytes = bytearray()
    # Track first-seen segments by (start_pos, length) to avoid duplicating the
    # same underlying buffer content across different tensor indices.
    seen_segments = set()
    if params_info:
        # Write in ascending start_pos order to preserve the original layout,
        # ensuring that references remain valid without rebasing.
        sorted_params = sorted(params_info, key=lambda p: p['start_pos'])
        for param in sorted_params:
            seg_key = (param['start_pos'], param['length'])
            if seg_key in seen_segments:
                # Already emitted this segment; skip duplicate bytes.
                continue
            start = param['start_pos']
            end = start + param['length']
            slice_data = all_params[start:end]
            dtype = param.get('dtype', 'float32')
            # Keep dtype behavior aligned with previous logic: only int32 stays
            # int32; everything else is serialized as float32.
            if dtype == 'int32':
                array = np.asarray(slice_data, dtype=np.int32)
            else:
                array = np.asarray(slice_data, dtype=np.float32)
            param_bytes.extend(array.tobytes())
            seen_segments.add(seg_key)
    else:
        # Fallback: no structured params_info; dump all unique elements as float32.
        array = np.asarray(all_params, dtype=np.float32)
        param_bytes.extend(array.tobytes())

    # Use model_name for output filename
    output_filename = f'{model_name}_params.bin'
    with open(output_filename, 'wb') as f:
        f.write(param_bytes)
    print(f"Parameter file saved: {output_filename}")
    # Return the total element count, which corresponds to the unique elements
    # emitted (all_params already contains only first-seen segments).
    return len(all_params)

def read_params_file(file_path='params.bin'):
    try:
        params_array = np.fromfile(file_path, dtype=np.float32)
        print(f"Successfully read parameter file: {file_path}")
        print(f"Number of parameters: {len(params_array)}")
        print(f"Parameter range: [{params_array.min():.6f}, {params_array.max():.6f}]")
        return params_array
    except Exception as e:
        print(f"Read parameter file failed: {e}")
        return None

# Parse virtualized parameters
def de_virtualize_param(v_param, modulus):
    return v_param['v_start_pos'] % modulus, v_param['v_length']

# Extract specific parameters based on virtualization information
def extract_parameter_by_index(params_array, v_params_info, param_index):
    for v_param in v_params_info:
        if v_param['index'] == param_index:
            # Parse virtualization information
            start_pos, length = de_virtualize_param(v_param, v_param['modulus'])
            # Extract parameter data
            param_data = params_array[start_pos:start_pos + length]
            # Reshape to the original shape
            if 'shape' in v_param and v_param['shape']:
                param_data = param_data.reshape(v_param['shape'])
            return {
                'data': param_data,
                'shape': v_param.get('shape', []),
                'dtype': v_param.get('dtype', []),
                'quantization': v_param.get('quantization', {})
            }
    return None

# Generate the virtualized parameter info file
def generate_v_params_info_file(virtualized_info):
    # Convert data types
    serializable_info = convert_dtype_for_json(virtualized_info)
    with open('v_params_info.json', 'w', encoding='utf-8') as f:
        json.dump(serializable_info, f, indent=2, ensure_ascii=False)
    print(f"Virtualized parameter info saved: v_params_info.json")


# Parameter virtualization workflow
def params_virtualization_process(interpreter, model_json):
    print("Start the parameter virtualization process...")
    all_params, params_info = extract_all_parameters(interpreter, model_json)
    total_size = generate_params_file(all_params, params_info)
    v_params_info = virtualize_params(params_info, total_size)
    # generate_v_params_info_file(v_params_info)
    print("Parameter virtualization process completed!")
    # print("\nValidate the parameter virtualization results...")
    # params_array = read_params_file()
    # if params_array is not None and v_params_info:
    #     test_param = v_params_info[random.randint(0, len(v_params_info) - 1)]
    #     extracted_param = extract_parameter_by_index(params_array, v_params_info, test_param['index'])
    #     if extracted_param is not None:
    #         print(f"Successfully extracted parameters {test_param['index']}:\nShape{extracted_param['shape']}")
    #     else:
    #         print("Parameter extraction failed")
    return v_params_info
