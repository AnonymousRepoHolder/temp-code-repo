# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 The NeuralVirtualizer Authors
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import random
import struct
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

from utils.utils import extract_graph, convert_dtype_for_json


# Encrypt computation graph connections using AES-256-CTR
def virtualize_graph(graph, crypto_params):
    total_ops = len(graph)

    # Extract conn_modulus from crypto_params
    conn_modulus = crypto_params['conn_modulus']

    v_graph = []
    for op in graph:
        v_forward_connections = []

        # Encrypt each forward connection
        for conn_idx, conn_id in enumerate(op['forward_connections']):
            # Apply modular arithmetic virtualization to connection ID
            real_conn_id = conn_id
            r_conn = random.randint(1, 10000)
            v_conn_id = r_conn * conn_modulus + real_conn_id

            # Construct 16-byte nonce: 8-byte base + 8-byte combined counter (using semantic op index)
            # Combine op index and conn_idx into a single 64-bit counter
            combined_counter = (op['index'] << 32) | conn_idx
            nonce_counter = struct.pack('<Q', combined_counter)
            nonce = crypto_params['nonce_bases']['graph'] + nonce_counter  # Total: 16 bytes

            # Create AES-256-CTR cipher using plaintext graph key
            cipher = Cipher(
                algorithms.AES(crypto_params['aes_keys']['graph']),
                modes.CTR(nonce),  # Use 16-byte nonce directly
                backend=default_backend()
            )
            encryptor = cipher.encryptor()

            # Encrypt virtualized connection ID (4-byte unsigned integer)
            plaintext = struct.pack('<I', v_conn_id)
            ciphertext = encryptor.update(plaintext) + encryptor.finalize()

            # Encode ciphertext as Base64
            v_conn_data = base64.b64encode(ciphertext[:4]).decode('ascii')
            v_forward_connections.append(v_conn_data)

        v_graph.append({
            'index': op['index'],
            'v_forward_connections': v_forward_connections,
            'forward_branches': op['forward_branches']
        })

    return v_graph

# Parse the virtualized computation graph
def de_virtualize_graph(v_forward_connections, modulus):
    return [v_conn_id % modulus for v_conn_id in v_forward_connections]

# Generate the virtualized computation graph file
def generate_v_graph_file(v_graph):
    # Convert data types
    serializable_info = convert_dtype_for_json(v_graph)
    with open('v_graph.json', 'w', encoding='utf-8') as f:
        json.dump(serializable_info, f, indent=2, ensure_ascii=False)
    print(f"Virtualized computation graph saved: v_graph.json")

def graph_virtualization_process(model_json):
    print("Start the computation graph virtualization process...")
    graph = extract_graph(model_json)
    v_graph = virtualize_graph(graph)
    # generate_v_graph_file(v_graph)
    print("Computation graph virtualization process completed!")
    # print("\nValidate the computation graph virtualization results...")
    # test_op = v_graph[12]
    # test_op = v_graph[random.randint(0, len(v_graph) - 1)]
    # test_forward_op = de_virtualize_graph(test_op['v_forward_connections'], test_op['modulus'])
    # if test_forward_op is not None:
    #     print(f"Successfully extracted computation graph:\nOperator: {test_op['index']}\nForward connected operators:\n{test_forward_op}")
    # else:
    #     print("Computation graph extraction failed")
    return v_graph
