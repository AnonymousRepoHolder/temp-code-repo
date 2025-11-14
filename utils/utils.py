# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 The NeuralVirtualizer Authors
import os
import gc
import numpy as np
from collections import defaultdict, Counter
# Show only WARNING and above logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.python.ops.ragged.ragged_factory_ops import placeholder

# Extract all network layer type codes
def extract_all_op_types(interpreter, model_json):
    all_op_types = []
    operator_codes = model_json['operator_codes']
    # Allow multiple inputs/outputs
    input_ops_index = [placeholder] * (len(model_json['subgraphs'][0]['inputs']))
    output_ops_index = [placeholder] * (len(model_json['subgraphs'][0]['outputs']))
    # Build a mapping from subgraph input tensors to their positions
    subgraph_inputs = model_json['subgraphs'][0].get('inputs', [])
    input_tensor_to_pos = {tensor_id: pos for pos, tensor_id in enumerate(subgraph_inputs)}
    # Cache ops_details and its reverse mapping once to avoid reverse lookups inside the loop
    try:
        ops_details = interpreter._get_ops_details()
    except Exception:
        ops_details = [{}] * len(model_json['subgraphs'][0]['operators'])
    op_name_to_code = {v: k for k, v in op_type_mapping.items()}

    # Iterate over all operators
    for i, op in enumerate(model_json['subgraphs'][0]['operators']):
        if 'opcode_index' in op:
            deprecated_builtin_code = operator_codes[op['opcode_index']].get('deprecated_builtin_code')
            # operator_codes may contain empty dictionaries
            if deprecated_builtin_code is None:
                op_name = ops_details[i].get('op_name')
                deprecated_builtin_code = op_name_to_code.get(op_name)
        else:
            # Handle special network layer types
            op_name = ops_details[i].get('op_name')
            deprecated_builtin_code = op_name_to_code.get(op_name)
        # Record the total number of inputs for this operator
        num_inputs = len(op.get('inputs', []))
        # Record which input slots come directly from subgraph inputs (and which subgraph input index they use)
        graph_input_slots = []  # List item: (slot_index, subgraph_input_pos)
        for slot_idx, tensor_id in enumerate(op.get('inputs', [])):
            if tensor_id in input_tensor_to_pos:
                graph_input_slots.append([slot_idx, input_tensor_to_pos[tensor_id]])
        # Unify the data structure of builtin_options: use an empty dictionary for missing or unexpected types
        # to avoid errors when using dict APIs during subsequent parsing
        builtin_options = op.get('builtin_options', {})
        if not isinstance(builtin_options, dict):
            builtin_options = {}
        mut_vars = op.get('mutating_variable_inputs', [])
        all_op_types.append({
            'index': i,
            'deprecated_builtin_code': deprecated_builtin_code,
            'builtin_options': builtin_options,
            'mutating_variable_inputs': mut_vars,
            # Record which input slots are taken directly from subgraph inputs (slot_index, subgraph_input_pos).
            'graph_input_slots': graph_input_slots
        })
        # Determine input/output operators; the order must match the JSON produced by flatc
        for input in op['inputs']:
            if input in model_json['subgraphs'][0]['inputs']:
                input_ops_index[model_json['subgraphs'][0]['inputs'].index(input)] = i
                break
        for output in op['outputs']:
            if output in model_json['subgraphs'][0]['outputs']:
                output_ops_index[model_json['subgraphs'][0]['outputs'].index(output)] = i
                break

    # Post-fix unresolved input/output mappings to avoid non-int placeholders.
    # Build a precise mapping from tensor id to its producing operator index.
    try:
        operators_fb = model_json['subgraphs'][0]['operators']
        tensor_id_to_producer = {}
        for i_op, op_fb in enumerate(operators_fb):
            for t_out in op_fb.get('outputs', []):
                tensor_id_to_producer[t_out] = i_op
        # Rebuild output_ops_index strictly in subgraph outputs order, filling all positions.
        subgraph_outputs = model_json['subgraphs'][0].get('outputs', [])
        fixed_output_ops_index = []
        for t_id in subgraph_outputs:
            prod = tensor_id_to_producer.get(t_id, None)
            # Always append an int to avoid type issues downstream; fall back to 0 if unknown.
            fixed_output_ops_index.append(int(prod) if isinstance(prod, int) else 0)
        output_ops_index = fixed_output_ops_index
    except Exception:
        # In the rare case of unexpected JSON shape, keep original behavior.
        pass

    # Filter input_ops_index to integers only; unresolved entries are omitted to avoid type errors.
    # The downstream consumer tolerates empty input list and can infer I/O from the graph.
    input_ops_index = [idx for idx in input_ops_index if isinstance(idx, int)]

    # Collect builtin_options statistics for smart padding
    options_stats = collect_builtin_options_statistics(all_op_types)

    return all_op_types, input_ops_index, output_ops_index, options_stats

# Extract all network layer parameters
def extract_all_parameters(interpreter, model_json):
    interpreter.allocate_tensors()
    all_params = []
    params_info = []
    operator_codes = model_json['operator_codes']
    total_ops = len(model_json['subgraphs'][0]['operators'])
    print(f"Start extracting parameters, total of {total_ops} operators")
    # Precompute the tensor index mapping to avoid repeated calls to get_tensor_details()
    tensor_details_list = interpreter.get_tensor_details()
    tensor_index_map = {detail['index']: detail for detail in tensor_details_list}
    # Precompute operator details to avoid repeated calls to _get_ops_details()
    ops_details = interpreter._get_ops_details()
    # Precompute operator type mapping to reduce repeated lookups
    op_types_cache = {}
    for i, op in enumerate(model_json['subgraphs'][0]['operators']):
        if 'opcode_index' in op:
            deprecated_builtin_code = operator_codes[op['opcode_index']].get('deprecated_builtin_code')
            # operator_codes可能有空字典
            if deprecated_builtin_code is None:
                deprecated_builtin_code = [k for k, v in op_type_mapping.items()
                                            if v == ops_details[i]['op_name']][0]
        else:
            # Handle special network layer types
            deprecated_builtin_code = [k for k, v in op_type_mapping.items()
                                        if v == ops_details[i]['op_name']][0]
        op_types_cache[i] = deprecated_builtin_code
    # Precompute whether an input comes from an upstream operator (non constant) or from a graph input
    produced_outputs = set()
    for node in model_json['subgraphs'][0]['operators']:
        for out_idx in node.get('outputs', []):
            produced_outputs.add(out_idx)
    graph_inputs = set(model_json['subgraphs'][0].get('inputs', []))

    # Record processed parameters by tensor index to avoid re-extracting the same tensor index.
    processed_params = {}
    # Deduplicate by underlying FlatBuffer buffer id: tensors that share the same
    # "buffer" field in the flatc JSON point to the same underlying bytes.
    # We will only append the data of the first-seen buffer id into all_params, and
    # subsequent tensors sharing this buffer will reuse the same start_pos/length.
    buffer_first = {}  # buffer_id -> {'start_pos': int, 'length': int, 'dtype': str}
    # Iterate over all operators
    for i, op in enumerate(model_json['subgraphs'][0]['operators']):
        # Print progress every 50 operators
        if i % 50 == 0:
            print(f"Parameter extraction progress: {i}/{total_ops} ({i/total_ops*100:.1f}%)")
        # Use cached operator types
        deprecated_builtin_code = op_types_cache[i]
        op_type = get_op_type_from_deprecated_builtin_code(deprecated_builtin_code)
        # If there are no weights and biases
        if op_type not in PARAM_OPS:
            continue
        # Optimized tensor handling: check only relevant tensor indices and their slots in inputs
        relevant_pairs = []  # (tensor_index, input_slot)
        if len(op['inputs']) == 2:  # Binary or single parameter
            if op_type == 'SPLIT':
                # For these operators the first input is a parameter
                relevant_pairs.append((op['inputs'][0], 0))
            else:
                # For binary elementwise or other binary operators either position may be a constant
                for slot in (0, 1):
                    cand = op['inputs'][slot]
                    if (cand not in produced_outputs) and (cand not in graph_inputs):
                        relevant_pairs.append((cand, slot))
        elif len(op['inputs']) > 2:  # Multiple inputs: extract only those inputs that appear to be constants including slot 0
            for slot in range(0, len(op['inputs'])):
                cand = op['inputs'][slot]
                if (cand not in produced_outputs) and (cand not in graph_inputs):
                    relevant_pairs.append((cand, slot))
        for tensor_index, input_slot in relevant_pairs:
            if tensor_index in tensor_index_map:
                tensor_details = tensor_index_map[tensor_index]
                try:
                    # Identify underlying buffer id from flatc JSON to detect sharing.
                    tensors_fb = model_json['subgraphs'][0]['tensors']
                    buf_id = None
                    try:
                        buf_id = tensors_fb[tensor_index].get('buffer')
                    except Exception:
                        buf_id = None

                    # Fetch the tensor only when needed (first time we see this buffer),
                    # to avoid unnecessary memory traffic.
                    if buf_id is not None and buf_id in buffer_first:
                        # Reuse the first-seen buffer segment (dedup by buffer id).
                        first = buffer_first[buf_id]
                        new_param_info = {
                            'index': tensor_details['index'],
                            'op_index': i,
                            'start_pos': first['start_pos'],
                            'length': first['length'],
                            'shape': tensor_index_map[tensor_index].get('shape', None),
                            'dtype': first['dtype'],
                            'input_slot': input_slot
                        }
                        params_info.append(new_param_info)
                        processed_params[new_param_info['index']] = new_param_info
                    else:
                        tensor = interpreter.get_tensor(tensor_details['index'])
                        new_param_info = {
                            'index': tensor_details['index'],
                            'op_index': i,
                            'start_pos': len(all_params),
                            'length': tensor.size,
                            'shape': tensor.shape,
                            'dtype': tensor.dtype.name,
                            'input_slot': input_slot
                        }
                        if new_param_info['index'] in processed_params:
                            processed_param = processed_params[new_param_info['index']]
                            new_param_info['start_pos'] = processed_param['start_pos']
                            new_param_info['length'] = processed_param['length']
                            new_param_info['shape'] = processed_param['shape']
                            new_param_info['dtype'] = processed_param['dtype']
                            params_info.append(new_param_info)
                        else:
                            processed_params[new_param_info['index']] = new_param_info
                            params_info.append(new_param_info)
                            # Memory optimization: process large tensors in batches
                            if tensor.size > 1000000:  # >1M elements
                                flat_tensor = tensor.flatten()
                                chunk_size = 100000
                                for chunk_start in range(0, len(flat_tensor), chunk_size):
                                    chunk_end = min(chunk_start + chunk_size, len(flat_tensor))
                                    chunk = flat_tensor[chunk_start:chunk_end].tolist()
                                    all_params.extend(chunk)
                                    del chunk  # Release chunk memory immediately
                                del flat_tensor
                            else:
                                all_params.extend(tensor.flatten().tolist())
                            # Register first-seen info for this buffer id, if available.
                            if buf_id is not None:
                                buffer_first[buf_id] = {
                                    'start_pos': new_param_info['start_pos'],
                                    'length': new_param_info['length'],
                                    'dtype': new_param_info['dtype']
                                }
                        del tensor  # Release tensor memory immediately
                except Exception as e:
                    print(f"Error extracting tensor {tensor_details['index']}: {e}")
                    continue
        # Perform garbage collection every 50 operators
        if i % 50 == 0:
            gc.collect()
    print(f"Parameter extraction completed, extracted {len(all_params)} parameters in total")
    return all_params, params_info

# Extract the computation graph structure
def extract_graph(model_json):
    graph = []
    operators = model_json['subgraphs'][0]['operators']
    # First build the mapping from output tensors to their producers, O(N)
    tensor_id_to_producer = {}
    for i, op in enumerate(operators):
        outs = op.get('outputs', [])
        for b_idx, t_id in enumerate(outs):
            tensor_id_to_producer[t_id] = (i, b_idx)
    # Build the graph and establish connections, O(N)
    for i, op in enumerate(operators):
        node = {
            'index': i,
            'outputs': op.get('outputs', []),
            'forward_connections': [],
            'forward_branches': []
        }
        for input_tensor in op.get('inputs', []):
            prod = tensor_id_to_producer.get(input_tensor)
            if prod is not None:
                p_index, branch = prod
                node['forward_connections'].append(p_index)
                node['forward_branches'].append(branch)
        graph.append(node)
    return graph

# Lookup layer type via deprecated_builtin_code
def get_op_type_from_deprecated_builtin_code(deprecated_builtin_code):
    if deprecated_builtin_code in op_type_mapping:
        return op_type_mapping[deprecated_builtin_code]
    else:
        return f"UNKNOWN_OP_{deprecated_builtin_code}"

# Convert TensorFlow data types to strings for JSON serialization
def convert_dtype_for_json(obj):
    if hasattr(obj, 'name'):
        return str(obj.name)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_dtype_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_dtype_for_json(item) for item in obj]
    else:
        return obj

op_type_mapping = {
    0: "ADD",
    1: "AVERAGE_POOL_2D",
    2: "CONCATENATION",
    3: "CONV_2D",
    4: "DEPTHWISE_CONV_2D",
    5: "DEPTH_TO_SPACE",
    6: "DEQUANTIZE",
    7: "EMBEDDING_LOOKUP",
    8: "FLOOR",
    9: "FULLY_CONNECTED",
    10: "HASHTABLE_LOOKUP",
    11: "L2_NORMALIZATION",
    12: "L2_POOL_2D",
    13: "LOCAL_RESPONSE_NORMALIZATION",
    14: "LOGISTIC",
    15: "LSH_PROJECTION",
    16: "LSTM",
    17: "MAX_POOL_2D",
    18: "MUL",
    19: "RELU",
    20: "RELU_N1_TO_1",
    21: "RELU6",
    22: "RESHAPE",
    23: "RESIZE_BILINEAR",
    24: "RNN",
    25: "SOFTMAX",
    26: "SPACE_TO_DEPTH",
    27: "SVDF",
    28: "TANH",
    29: "CONCAT_EMBEDDINGS",
    30: "SKIP_GRAM",
    31: "CALL",
    32: "CUSTOM",
    33: "EMBEDDING_LOOKUP_SPARSE",
    34: "PAD",
    35: "UNIDIRECTIONAL_SEQUENCE_RNN",
    36: "GATHER",
    37: "BATCH_TO_SPACE_ND",
    38: "SPACE_TO_BATCH_ND",
    39: "TRANSPOSE",
    40: "MEAN",
    41: "SUB",
    42: "DIV",
    43: "SQUEEZE",
    44: "UNIDIRECTIONAL_SEQUENCE_LSTM",
    45: "STRIDED_SLICE",
    46: "BIDIRECTIONAL_SEQUENCE_RNN",
    47: "EXP",
    48: "TOPK_V2",
    49: "SPLIT",
    50: "LOG_SOFTMAX",
    51: "DELEGATE",
    52: "BIDIRECTIONAL_SEQUENCE_LSTM",
    53: "CAST",
    54: "PRELU",
    55: "MAXIMUM",
    56: "ARG_MAX",
    57: "MINIMUM",
    58: "LESS",
    59: "NEG",
    60: "PADV2",
    61: "GREATER",
    62: "GREATER_EQUAL",
    63: "LESS_EQUAL",
    64: "SELECT",
    65: "SLICE",
    66: "SIN",
    67: "TRANSPOSE_CONV",
    68: "SPARSE_TO_DENSE",
    69: "TILE",
    70: "EXPAND_DIMS",
    71: "EQUAL",
    72: "NOT_EQUAL",
    73: "LOG",
    74: "SUM",
    75: "SQRT",
    76: "RSQRT",
    77: "SHAPE",
    78: "POW",
    79: "ARG_MIN",
    80: "FAKE_QUANT",
    81: "REDUCE_PROD",
    82: "REDUCE_MAX",
    83: "PACK",
    84: "LOGICAL_OR",
    85: "ONE_HOT",
    86: "LOGICAL_AND",
    87: "LOGICAL_NOT",
    88: "UNPACK",
    89: "REDUCE_MIN",
    90: "FLOOR_DIV",
    91: "REDUCE_ANY",
    92: "SQUARE",
    93: "ZEROS_LIKE",
    94: "FILL",
    95: "FLOOR_MOD",
    96: "RANGE",
    97: "RESIZE_NEAREST_NEIGHBOR",
    98: "LEAKY_RELU",
    99: "SQUARED_DIFFERENCE",
    100: "MIRROR_PAD",
    101: "ABS",
    102: "SPLIT_V",
    103: "UNIQUE",
    104: "CEIL",
    105: "REVERSE_V2",
    106: "ADD_N",
    107: "GATHER_ND",
    108: "COS",
    109: "WHERE",
    110: "RANK",
    111: "ELU",
    112: "REVERSE_SEQUENCE",
    113: "MATRIX_DIAG",
    114: "QUANTIZE",
    115: "MATRIX_SET_DIAG",
    116: "ROUND",
    117: "HARD_SWISH",
    118: "IF",
    119: "WHILE",
    120: "NON_MAX_SUPPRESSION_V4",
    121: "NON_MAX_SUPPRESSION_V5",
    122: "SCATTER_ND",
    123: "SELECT_V2",
    124: "DENSIFY",
    125: "SEGMENT_SUM",
    126: "BATCH_MATMUL",
    127: "PLACEHOLDER_FOR_GREATER_OP_CODES",
    128: "CUMSUM",
    129: "CALL_ONCE",
    130: "BROADCAST_TO",
    131: "RFFT2D",
    132: "CONV_3D",
    133: "IMAG",
    134: "REAL",
    135: "COMPLEX_ABS",
    136: "HASHTABLE",
    137: "HASHTABLE_FIND",
    138: "HASHTABLE_IMPORT",
    139: "HASHTABLE_SIZE",
    140: "REDUCE_ALL",
    141: "CONV_3D_TRANSPOSE",
    142: "VAR_HANDLE",
    143: "READ_VARIABLE",
    144: "ASSIGN_VARIABLE",
    145: "BROADCAST_ARGS",
    146: "RANDOM_STANDARD_NORMAL",
    147: "BUCKETIZE",
    148: "RANDOM_UNIFORM",
    149: "MULTINOMIAL",
    150: "GELU",
    151: "DYNAMIC_UPDATE_SLICE",
    152: "RELU_0_TO_1",
    153: "UNSORTED_SEGMENT_PROD",
    154: "UNSORTED_SEGMENT_MAX",
    155: "UNSORTED_SEGMENT_SUM",
    156: "ATAN2",
    157: "UNSORTED_SEGMENT_MIN",
    158: "SIGN",
    159: "BITCAST",
    160: "BITWISE_XOR",
    161: "RIGHT_SHIFT",
    162: "STABLEHLO_LOGISTIC",
    163: "STABLEHLO_ADD",
    164: "STABLEHLO_DIVIDE",
    165: "STABLEHLO_MULTIPLY",
    166: "STABLEHLO_MAXIMUM",
    167: "STABLEHLO_RESHAPE",
    168: "STABLEHLO_CLAMP",
    169: "STABLEHLO_CONCATENATE",
    170: "STABLEHLO_BROADCAST_IN_DIM",
    171: "STABLEHLO_CONVOLUTION",
    172: "STABLEHLO_SLICE",
    173: "STABLEHLO_CUSTOM_CALL",
    174: "STABLEHLO_REDUCE",
    175: "STABLEHLO_ABS",
    176: "STABLEHLO_AND",
    177: "STABLEHLO_COSINE",
    178: "STABLEHLO_EXPONENTIAL",
    179: "STABLEHLO_FLOOR",
    180: "STABLEHLO_LOG",
    181: "STABLEHLO_MINIMUM",
    182: "STABLEHLO_NEGATE",
    183: "STABLEHLO_OR",
    184: "STABLEHLO_POWER",
    185: "STABLEHLO_REMAINDER",
    186: "STABLEHLO_RSQRT",
    187: "STABLEHLO_SELECT",
    188: "STABLEHLO_SUBTRACT",
    189: "STABLEHLO_TANH"
}

# Operators with weights or biases
# Extend to include elementwise binary operators to extract their constant inputs such as scale and bias in LayerNorm
PARAM_OPS = {
    'CONV_2D', 'DEPTHWISE_CONV_2D', 'CONV_3D', 'CONV_3D_TRANSPOSE',
    'TRANSPOSE_CONV', 'FULLY_CONNECTED', 'SVDF', 'LSTM',
    'UNIDIRECTIONAL_SEQUENCE_LSTM', 'BIDIRECTIONAL_SEQUENCE_LSTM',
    'RNN', 'UNIDIRECTIONAL_SEQUENCE_RNN', 'BIDIRECTIONAL_SEQUENCE_RNN',
    'MEAN', 'RESHAPE', 'RESIZE_BILINEAR', 'TRANSPOSE', 'GATHER',
    'SPLIT',
    # Elementwise binary operators whose inputs may all come from upstream operators or may include constants
    'ADD', 'MUL', 'SUB', 'SQUARED_DIFFERENCE', 'BATCH_MATMUL',
    # Shape and mask related operators
    # need to extract their constant inputs such as begin end strides axis start limit delta and thresholds
    'STRIDED_SLICE', 'PACK', 'RANGE', 'EXPAND_DIMS', 'CAST', 'GREATER_EQUAL', 'SHAPE'
}

# Apply random index mapping to break sequential index patterns
def apply_random_index_mapping(all_op_types, input_ops_index, output_ops_index,
                               params_info, graph):
    """
    Apply truly random index mapping after extraction and before virtualization.

    Strategy: Sample from range [0, num_ops*10) to ensure:
    1. Non-sequential (impossible to restore by sorting)
    2. Large range (increased randomness)
    3. Uniqueness (no duplicates)

    Args:
        all_op_types: List of operator type information
        input_ops_index: List of input operator indices
        output_ops_index: List of output operator indices
        params_info: List of parameter information
        graph: List of graph node information

    Returns:
        Tuple of (all_op_types, input_ops_index, output_ops_index, params_info, graph)
        with old_to_new mapping applied to all index references
    """
    import random

    num_ops = len(all_op_types)

    # Generate random new index set by sampling from larger range
    available_indices = list(range(num_ops * 10))
    new_indices = random.sample(available_indices, num_ops)

    # Build mapping from old index to new index
    old_to_new = {i: new_indices[i] for i in range(num_ops)}

    # Update all references to old indices

    # 1. Update all_op_types index field
    for op in all_op_types:
        op['index'] = old_to_new[op['index']]

    # 2. Update input/output_ops_index
    new_input_ops = [old_to_new.get(idx, idx) for idx in input_ops_index]
    new_output_ops = [old_to_new.get(idx, idx) for idx in output_ops_index]

    # 3. Update params_info op_index field
    for param in params_info:
        param['op_index'] = old_to_new[param['op_index']]

    # 4. Update graph index and forward_connections
    for node in graph:
        node['index'] = old_to_new[node['index']]
        # Critical: update old indices referenced in forward_connections
        node['forward_connections'] = [
            old_to_new[conn] for conn in node['forward_connections']
        ]
        # forward_branches do not need updating (they are branch numbers, not indices)

    return all_op_types, new_input_ops, new_output_ops, params_info, graph


def collect_builtin_options_statistics(all_op_types):
    """
    Collect statistics of builtin_options values at each position to enable smart padding.

    This function analyzes all operators' builtin_options to find the most common value
    at each position. These statistics are then used for intelligent padding, making
    different operator types appear more similar.

    Note: Floating-point positions (e.g., SOFTMAX's beta) are excluded from statistics
    and will use predefined safe default values during padding to avoid semantic errors.

    Returns:
        dict: {position: most_common_value or 'float'} mapping for positions 0-7
    """
    from virtualizer.ops_virtualizer import _encode_builtin_options, get_op_type_from_deprecated_builtin_code

    position_values = defaultdict(list)
    position_has_float = {}  # Track positions that contain floating-point values

    for op in all_op_types:
        op_type = get_op_type_from_deprecated_builtin_code(
            op['deprecated_builtin_code']
        )

        # Encode this operator's builtin_options (without padding)
        encoded = _encode_builtin_options(op_type, op.get('builtin_options', {}))

        # Record the value at each position
        for pos, val in enumerate(encoded):
            if isinstance(val, float):
                # Mark this position as containing float values
                position_has_float[pos] = True
            else:
                # Only collect statistics for integer positions
                position_values[pos].append(val)

    # Find the most common value at each position
    position_most_common = {}
    for pos in range(8):  # Fixed length of 8
        if pos in position_has_float:
            # For float positions, use special marker 'float'
            # Actual default value will be determined during padding
            position_most_common[pos] = 'float'
        elif pos in position_values and position_values[pos]:
            counter = Counter(position_values[pos])
            most_common_value = counter.most_common(1)[0][0]
            position_most_common[pos] = most_common_value
        else:
            # This position has no real values, use default 0
            position_most_common[pos] = 0

    return position_most_common

