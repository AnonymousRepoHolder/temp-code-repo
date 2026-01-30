#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 The NeuralVirtualizer Authors

"""
Shared Cryptographic Parameters for Ablation Experiments

This module provides consistent key generation and storage for exp2 and exp3.
Both experiments MUST use the same keys and nonce bases for fair comparison
of collision rates.

Usage:
    from crypto_utils import get_shared_crypto_params, reset_crypto_params

    # Get or generate shared parameters (cached)
    crypto_params = get_shared_crypto_params()

    # Reset for a new experiment batch
    reset_crypto_params()
"""

import json
import os
import secrets
import struct

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CRYPTO_CACHE_FILE = os.path.join(SCRIPT_DIR, 'outputs', '_shared_crypto_params.bin')

# Global cache
_cached_crypto_params = None


def generate_crypto_params() -> dict:
    """
    Generate fresh cryptographic parameters.

    Returns:
        Dictionary with crypto parameters matching the real virtualization format
    """
    domains = ['op', 'param', 'graph', 'shape']
    crypto_params = {
        'op_modulus': 251,
        'conn_modulus': 50021,
        'aes_keys': {},
        'nonce_bases': {}
    }

    for domain in domains:
        crypto_params['aes_keys'][domain] = secrets.token_bytes(32)
        crypto_params['nonce_bases'][domain] = secrets.token_bytes(8)

    return crypto_params


def save_crypto_params(crypto_params: dict, path: str = CRYPTO_CACHE_FILE) -> None:
    """
    Save crypto parameters to binary file.

    Binary format (168 bytes, same as keys_and_offsets.bin):
        Offset   Size   Field
        0        4      OP Modulus (uint32)
        4        4      Connection Modulus (uint32)
        8        32     OP Encryption Key
        40       8      OP Nonce Base
        48       32     Param Encryption Key
        80       8      Param Nonce Base
        88       32     Graph Encryption Key
        120      8      Graph Nonce Base
        128      32     Shape Encryption Key
        160      8      Shape Nonce Base
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    domains = ['op', 'param', 'graph', 'shape']

    with open(path, 'wb') as f:
        # Write modulus values (8 bytes)
        f.write(struct.pack('<II', crypto_params['op_modulus'],
                                   crypto_params['conn_modulus']))

        # Write 4 domains x (32-byte key + 8-byte nonce)
        for domain in domains:
            f.write(crypto_params['aes_keys'][domain])
            f.write(crypto_params['nonce_bases'][domain])


def load_crypto_params(path: str = CRYPTO_CACHE_FILE) -> dict:
    """
    Load crypto parameters from binary file.

    Returns:
        Dictionary with crypto parameters
    """
    if not os.path.exists(path):
        return None

    domains = ['op', 'param', 'graph', 'shape']
    crypto_params = {
        'aes_keys': {},
        'nonce_bases': {}
    }

    with open(path, 'rb') as f:
        # Read modulus values
        modulus_data = f.read(8)
        crypto_params['op_modulus'], crypto_params['conn_modulus'] = struct.unpack('<II', modulus_data)

        # Read 4 domains
        for domain in domains:
            crypto_params['aes_keys'][domain] = f.read(32)
            crypto_params['nonce_bases'][domain] = f.read(8)

    return crypto_params


def get_shared_crypto_params(force_regenerate: bool = False) -> dict:
    """
    Get shared crypto parameters for ablation experiments.

    This function ensures that exp2 and exp3 use the SAME keys and nonce bases.
    Parameters are cached in memory and persisted to disk.

    Args:
        force_regenerate: If True, generate new parameters even if cached

    Returns:
        Dictionary with crypto parameters
    """
    global _cached_crypto_params

    # Return cached if available and not forcing regeneration
    if _cached_crypto_params is not None and not force_regenerate:
        return _cached_crypto_params

    # Try to load from disk
    if not force_regenerate and os.path.exists(CRYPTO_CACHE_FILE):
        _cached_crypto_params = load_crypto_params(CRYPTO_CACHE_FILE)
        if _cached_crypto_params is not None:
            return _cached_crypto_params

    # Generate new parameters
    _cached_crypto_params = generate_crypto_params()
    save_crypto_params(_cached_crypto_params, CRYPTO_CACHE_FILE)

    return _cached_crypto_params


def reset_crypto_params() -> None:
    """
    Reset cached crypto parameters and delete the cache file.

    Call this at the start of a new experiment batch to ensure fresh keys.
    """
    global _cached_crypto_params
    _cached_crypto_params = None

    if os.path.exists(CRYPTO_CACHE_FILE):
        os.remove(CRYPTO_CACHE_FILE)
