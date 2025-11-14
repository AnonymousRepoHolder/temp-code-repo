# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 The NeuralVirtualizer Authors
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import fileinput
import json
import random
import secrets
import struct
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

from utils.utils import extract_all_op_types, get_op_type_from_deprecated_builtin_code, convert_dtype_for_json

# Generate v_builtin_options: list, only store values/enums in order, without semantic keys

_ACTIVATION_TO_CODE = {
    'NONE': 0,
    'RELU': 1,
    'RELU6': 2,
}

_PADDING_TO_CODE = {
    'SAME': 0,
    'VALID': 1,
}

_DTYPE_TO_CODE = {
    'FLOAT32': 0,
    'INT32': 1,
    'BOOL': 2,
}


def _bool_to_int(x):
    try:
        return 1 if bool(x) else 0
    except Exception:
        return 0


def _get(options: dict, key: str, default):
    try:
        return options.get(key, default)
    except Exception:
        return default


def _encode_activation(options: dict, default: str = 'NONE') -> int:
    act = str(_get(options, 'fused_activation_function', default))
    return _ACTIVATION_TO_CODE.get(act, _ACTIVATION_TO_CODE['NONE'])


def _encode_padding(options: dict, default: str = 'SAME') -> int:
    pad = str(_get(options, 'padding', default))
    return _PADDING_TO_CODE.get(pad, _PADDING_TO_CODE['SAME'])


def _encode_dtype_code(options: dict, key: str, default: str) -> int:
    # CAST/SHAPE data type encoding (string → code)
    val = str(_get(options, key, default))
    return _DTYPE_TO_CODE.get(val, _DTYPE_TO_CODE[default])


def _encode_dims_list(options: dict, key: str):
    # Encode dims array as [N, d0, d1, ...], N=0 means no
    try:
        dims = options.get(key, [])
        if isinstance(dims, list) and len(dims) > 0:
            return [len(dims)] + [int(v) for v in dims]
    except Exception:
        pass
    return [0]


def _encode_new_shape(options: dict):
    # Encode RESHAPE.new_shape as [N, d0, d1, ...], support -1
    try:
        shape = options.get('new_shape', [])
        if isinstance(shape, list) and len(shape) > 0:
            return [len(shape)] + [int(v) for v in shape]
    except Exception:
        pass
    return [0]


def _encode_builtin_options(op_type: str, options: dict) -> list:
    # Infer the fields covered by the current builder/shape
    # If the operator is not covered or the default value is returned,
    # return an empty list or a sequence of default values
    if not isinstance(options, dict):
        options = {}

    t = op_type
    if t == 'CONV_2D':
        return [
            _encode_padding(options, 'SAME'),
            int(_get(options, 'stride_w', 1)),
            int(_get(options, 'stride_h', 1)),
            _encode_activation(options, 'NONE'),
        ]
    if t == 'DEPTHWISE_CONV_2D':
        return [
            _encode_padding(options, 'SAME'),
            int(_get(options, 'stride_w', 1)),
            int(_get(options, 'stride_h', 1)),
            int(_get(options, 'depth_multiplier', 1)),
            _encode_activation(options, 'NONE'),
        ]
    if t in ('MAX_POOL_2D', 'AVERAGE_POOL_2D'):
        return [
            _encode_padding(options, 'SAME'),
            int(_get(options, 'stride_w', 2)),
            int(_get(options, 'stride_h', 2)),
            int(_get(options, 'filter_width', 2)),
            int(_get(options, 'filter_height', 2)),
            _encode_activation(options, 'NONE'),
        ]
    if t == 'CONCATENATION':
        return [
            int(_get(options, 'axis', 3)),
            _encode_activation(options, 'NONE'),
        ]
    if t == 'FULLY_CONNECTED':
        return [
            _encode_activation(options, 'NONE'),
            _bool_to_int(_get(options, 'keep_num_dims', False)),
            _bool_to_int(_get(options, 'asymmetric_quantize_inputs', False)),
        ]
    if t == 'SOFTMAX':
        return [float(_get(options, 'beta', 1.0))]
    if t in ('ADD', 'MUL', 'SUB'):
        return [_encode_activation(options, 'NONE')]
    if t == 'RESHAPE':
        return _encode_new_shape(options)
    if t == 'RESIZE_BILINEAR':
        return [
            _bool_to_int(_get(options, 'align_corners', False)),
            _bool_to_int(_get(options, 'half_pixel_centers', False)),
        ]
    if t == 'MEAN':
        return [_bool_to_int(_get(options, 'keep_dims', False))]
    if t == 'GELU':
        return [_bool_to_int(_get(options, 'approximate', True))]
    if t == 'BATCH_MATMUL':
        return [
            _bool_to_int(_get(options, 'adj_x', False)),
            _bool_to_int(_get(options, 'adj_y', False)),
            _bool_to_int(_get(options, 'asymmetric_quantize_inputs', False)),
        ]
    if t == 'GATHER':
        return [
            int(_get(options, 'axis', 0)),
            int(_get(options, 'batch_dims', 0)),
        ]
    if t == 'SPLIT':
        return [int(_get(options, 'num_splits', 3))]
    if t == 'SQUEEZE':
        return _encode_dims_list(options, 'squeeze_dims')
    if t == 'STRIDED_SLICE':
        return [
            int(_get(options, 'begin_mask', 0)),
            int(_get(options, 'end_mask', 0)),
            int(_get(options, 'ellipsis_mask', 0)),
            int(_get(options, 'new_axis_mask', 0)),
            int(_get(options, 'shrink_axis_mask', 0)),
        ]
    if t == 'PACK':
        return [
            int(_get(options, 'values_count', 1)),
            int(_get(options, 'axis', 0)),
        ]
    if t == 'RANGE' or t == 'EXPAND_DIMS' or t == 'TRANSPOSE':
        return []
    if t == 'SHAPE':
        # Only out_type
        return [_encode_dtype_code(options, 'out_type', 'INT32')]
    if t == 'CAST':
        # Input/output types
        # Compatible with in_data_type/out_data_type and in_type/out_type two keys
        in_code = _encode_dtype_code(options, 'in_data_type', 'FLOAT32')
        if in_code == _DTYPE_TO_CODE['FLOAT32']:
            in_code = _encode_dtype_code(options, 'in_type', 'FLOAT32')
        out_code = _encode_dtype_code(options, 'out_data_type', 'FLOAT32')
        if out_code == _DTYPE_TO_CODE['FLOAT32']:
            out_code = _encode_dtype_code(options, 'out_type', 'FLOAT32')
        return [in_code, out_code]

    # Other operators not using builtin: empty list
    return []


# Maximum length for builtin_options (fixed for uniformity)
MAX_BUILTIN_OPTIONS_LENGTH = 8

# Fixed padding size for shape dims in Phase 1
# Using a compact K=6 to hide typical 3D-5D tensors while keeping size reasonable
SHAPE_PAD_K = 6


def pad_builtin_options_smart(encoded_options, position_stats):
    """
    Pad builtin_options to fixed length using statistically common values.

    Instead of padding with zeros, this function uses the most common value
    observed at each position across all operators. This makes different
    operator types appear more similar, reducing information leakage.

    Note: If encoded_options is longer than MAX_BUILTIN_OPTIONS_LENGTH,
    it will be preserved as-is to avoid information loss. Such cases are
    rare (e.g., high-dimensional RESHAPE) and do not significantly impact
    the overall uniformity of builtin_options.

    For floating-point positions (e.g., SOFTMAX's beta), safe default values
    are used instead of statistical values to avoid semantic errors.

    Args:
        encoded_options: List of encoded option values (variable length)
        position_stats: Dict mapping position to most common value or 'float'

    Returns:
        List of length MAX_BUILTIN_OPTIONS_LENGTH with smart padding,
        or original list if longer (rare cases preserved)
    """
    # If longer than maximum length, preserve as-is to avoid information loss
    if len(encoded_options) > MAX_BUILTIN_OPTIONS_LENGTH:
        return list(encoded_options)

    # Pad to fixed length using statistical common values
    padded = list(encoded_options)
    for pos in range(len(encoded_options), MAX_BUILTIN_OPTIONS_LENGTH):
        stat_val = position_stats.get(pos, 0)
        if stat_val == 'float':
            # For float positions, use safe default value 1.0
            # This matches common defaults in TFLite (e.g., SOFTMAX beta=1.0)
            padded.append(1.0)
        else:
            # Use statistical most common value for integer positions
            padded.append(stat_val)
    return padded

def encrypt_shape(shape, op_idx, shape_idx, crypto_params, dtype_code: int, param_slot: int):
    """
    Encrypt shape information using AES-256-CTR with an extended plaintext layout.

    Plaintext layout per entry (little-endian):
      [num_dims:uint32][dtype_code:uint32][param_slot:int32][dims:int32 × L]
    - If num_dims <= SHAPE_PAD_K, L = SHAPE_PAD_K (dims padded/truncated to K, parser uses num_dims)
    - If num_dims > SHAPE_PAD_K, L = num_dims (rare, allowed to be variable length)

    Args:
        shape: List[int], tensor dimensions (int32, supports negatives for dynamic dims)
        op_idx: int, operator index for nonce construction
        shape_idx: int, per-operator shape entry index for nonce construction
        crypto_params: dict, contains AES keys and nonce bases
        dtype_code: int, encoded dtype (1=float32,2=int32,3=bool)
        param_slot: int, original input slot index for this parameter (or -1 if absent)

    Returns:
        str: Base64-encoded encrypted shape block
    """
    # Construct 16-byte nonce: 8-byte base + 8-byte counter
    combined_counter = (op_idx << 32) | shape_idx
    nonce_counter = struct.pack('<Q', combined_counter)
    nonce = crypto_params['nonce_bases']['shape'] + nonce_counter

    # Determine padding length L
    # For Phase 3, replace zero-padding with true random padding to enhance obfuscation.
    # When num_dims < SHAPE_PAD_K, the tail slots are filled with random integers in [1, 2048].
    # NOTE: Header still records the true num_dims; parser will only use the first num_dims dims.
    num_dims = int(len(shape))
    if num_dims <= SHAPE_PAD_K:
        L = SHAPE_PAD_K
        dims = list(shape)
        pad_count = SHAPE_PAD_K - num_dims
        if pad_count > 0:
            dims.extend([random.randint(1, 2048) for _ in range(pad_count)])
    else:
        L = num_dims
        dims = list(shape)

    # Build plaintext buffer
    # num_dims (uint32), dtype_code (uint32), param_slot (int32), dims (int32 × L)
    plaintext = struct.pack('<IIi', num_dims, int(dtype_code), int(param_slot))
    if L > 0:
        plaintext += struct.pack(f'<{L}i', *[int(v) for v in dims])

    # Encrypt
    cipher = Cipher(
        algorithms.AES(crypto_params['aes_keys']['shape']),
        modes.CTR(nonce),
        backend=default_backend()
    )
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()

    return base64.b64encode(ciphertext).decode('ascii')


# Encrypt operator type codes using AES-256-CTR
def virtualize_op_types(op_types, input_ops_index, output_ops_index, options_stats, crypto_params):
    total_op_count = len(op_types)
    v_op_types = []

    # Extract op_modulus from crypto_params
    op_modulus = crypto_params['op_modulus']

    # Encrypt each operator's type code
    for op in op_types:
        # Apply modular arithmetic virtualization to deprecated_builtin_code
        real_code = op['deprecated_builtin_code']
        r_op = random.randint(1, 10000)
        v_code = r_op * op_modulus + real_code

        # Construct 16-byte nonce: 8-byte base + 8-byte counter (using semantic op index)
        nonce_counter = struct.pack('<Q', op['index'])
        nonce = crypto_params['nonce_bases']['op'] + nonce_counter  # Total: 16 bytes

        # Create AES-256-CTR cipher using plaintext op key
        cipher = Cipher(
            algorithms.AES(crypto_params['aes_keys']['op']),
            modes.CTR(nonce),  # Use 16-byte nonce directly
            backend=default_backend()
        )
        encryptor = cipher.encryptor()

        # Encrypt virtualized code (4-byte unsigned integer)
        plaintext = struct.pack('<I', v_code)
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()

        # Encode ciphertext as Base64
        v_op_code_data = base64.b64encode(ciphertext[:4]).decode('ascii')

        # Derive op_type for encoding builtin options
        op_type = get_op_type_from_deprecated_builtin_code(real_code)
        encoded_options = _encode_builtin_options(op_type, op.get('builtin_options', {}))

        # Apply smart padding using statistical common values.
        # Original behavior: if length > MAX, preserve as-is; if < MAX, pad to MAX.
        # Do not enforce truncation to keep minimal-intrusive semantics.
        padded_options = pad_builtin_options_smart(encoded_options, options_stats)

        # Quantize float slots and convert all values to int32 (variable length allowed)
        # Rule: positions marked as 'float' in options_stats or actual float values are quantized by x1e3
        int_slots = []
        for pos in range(len(padded_options)):
            val = padded_options[pos]
            if isinstance(val, float):
                q = int(round(float(val) * 1000.0))
            else:
                # Includes integers and booleans already encoded as 0/1
                q = int(val)
            int_slots.append(q)

        # Pack int32 (variable length) to bytes (little-endian)
        packed = struct.pack('<' + 'i' * len(int_slots), *int_slots)

        # Encrypt builtin_options as a single block using PARAM domain (AES-256-CTR)
        # Nonce: param_nonce_base (8B) + counter (8B)
        # Counter: (op_idx << 32) | 0x00000002, reserved to avoid interfering with existing elements
        counter_val = (op['index'] << 32) | 0x00000002
        nonce = crypto_params['nonce_bases']['param'] + struct.pack('<Q', counter_val)

        cipher = Cipher(
            algorithms.AES(crypto_params['aes_keys']['param']),
            modes.CTR(nonce),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ct = encryptor.update(packed) + encryptor.finalize()
        v_builtin_options_enc = base64.b64encode(ct).decode('ascii')

        v_op_types.append({
            'index': op['index'],
            'v_op_code_data': v_op_code_data,  # Base64-encoded ciphertext
            # Store builtin_options as Base64-encoded ciphertext (length may be > MAX when encoded options exceed it)
            'v_builtin_options': v_builtin_options_enc,
            'mutating_variable_inputs': op['mutating_variable_inputs'],
            # Pass through which input slots of this operator come directly from subgraph inputs
            # (slot_index, subgraph_input_pos)
            'graph_input_slots': op.get('graph_input_slots', [])
        })
    
    return v_op_types

# Parse virtualized layer types
def de_virtualize_op_type(v_deprecated_builtin_code, modulus):
    return v_deprecated_builtin_code % modulus

# Generate the virtualized layer types file
def generate_v_op_types_file(v_op_types):
    # Convert data types
    serializable_info = convert_dtype_for_json(v_op_types)
    with open('v_op_types.json', 'w', encoding='utf-8') as f:
        json.dump(serializable_info, f, indent=2, ensure_ascii=False)
    print(f"Virtualized layer types saved: v_op_types.json")

def op_types_virtualization_process(interpreter, model_json):
    print("Start the layer type virtualization process...")
    all_op_types, input_ops_index, output_ops_index = extract_all_op_types(interpreter, model_json)
    v_op_types = virtualize_op_types(all_op_types, input_ops_index, output_ops_index)
    # generate_v_op_types_file(v_op_types)
    print("Layer type virtualization process completed!")
    # print("\nValidate the layer type virtualization results...")
    # test_op_type = v_op_types[random.randint(0, len(v_op_types) - 1)]
    # test_deprecated_builtin_code = de_virtualize_op_type(test_op_type['v_deprecated_builtin_code'], test_op_type['modulus'])
    # real_op_type = get_op_type_from_deprecated_builtin_code(test_deprecated_builtin_code)
    # if real_op_type is not None:
    #     print(f"Successfully extracted layer types {test_op_type['index']}:\Type{real_op_type}")
    # else:
    #     print("Layer type extraction failed")
    return v_op_types
