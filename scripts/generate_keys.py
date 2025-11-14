#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 The NeuralVirtualizer Authors

"""
Key Generation Script for Multi-Model Virtualization

This script generates cryptographic parameters for batch model virtualization:
- Modulus values for modular arithmetic virtualization (OP and Connection domains)
- AES-256 plaintext keys for 4 encryption domains (OP, Param, Graph, Shape)
- Nonce base values (8 bytes each) for AES-CTR mode

Output: keys_and_offsets.bin (168 bytes, plaintext storage)
Note: XOR obfuscation parameters are NOT generated here - they will be created
      during injection to C++ backend by inject_keys_to_backend.py
"""

import os
import sys
import secrets
import struct
import glob


def generate_crypto_params():
    """
    Generate all cryptographic parameters for virtualization.

    Returns:
        dict: Contains modulus values, AES keys, and nonce bases
    """
    # Fixed modulus values
    op_modulus = 251  # Covers all TFLite operator types (0-189)
    conn_modulus = 50021  # Covers models with <5000 layers

    # Generate plaintext AES-256 keys (32 bytes each) for 4 domains
    domains = ['op', 'param', 'graph', 'shape']
    aes_keys = {}
    nonce_bases = {}

    for domain in domains:
        aes_keys[domain] = secrets.token_bytes(32)  # 32-byte plaintext key
        nonce_bases[domain] = secrets.token_bytes(8)  # 8-byte nonce base

    return {
        'op_modulus': op_modulus,
        'conn_modulus': conn_modulus,
        'aes_keys': aes_keys,
        'nonce_bases': nonce_bases
    }


def write_binary_file(crypto_params, output_path='keys_and_offsets.bin'):
    """
    Write cryptographic parameters to binary file (168 bytes).

    Binary format (168 bytes total):
        Offset   Size   Field
        0        4      OP Modulus (uint32)
        4        4      Connection Modulus (uint32)
        8        32     OP Encryption Key (plaintext)
        40       8      OP Nonce Base
        48       32     Param Encryption Key (plaintext)
        80       8      Param Nonce Base
        88       32     Graph Encryption Key (plaintext)
        120      8      Graph Nonce Base
        128      32     Shape Encryption Key (plaintext)
        160      8      Shape Nonce Base

    Args:
        crypto_params: Dictionary containing all cryptographic parameters
        output_path: Output file path (default: keys_and_offsets.bin)
    """
    domains = ['op', 'param', 'graph', 'shape']

    with open(output_path, 'wb') as f:
        # Write modulus values (8 bytes total)
        f.write(struct.pack('<II', crypto_params['op_modulus'],
                                   crypto_params['conn_modulus']))

        # Write 4 domains × (32-byte key + 8-byte nonce) = 160 bytes
        for domain in domains:
            f.write(crypto_params['aes_keys'][domain])  # 32 bytes plaintext key
            f.write(crypto_params['nonce_bases'][domain])  # 8 bytes nonce base

    file_size = os.path.getsize(output_path)
    if file_size != 168:
        raise ValueError(f"ERROR: Binary file size is {file_size} bytes, expected 168 bytes")

    print(f"✓ Generated {output_path} ({file_size} bytes)")
    print(f"  - Modulus values: op_modulus={crypto_params['op_modulus']}, "
          f"conn_modulus={crypto_params['conn_modulus']}")
    print(f"  - 4 plaintext AES-256 keys (32 bytes each)")
    print(f"  - 4 nonce bases (8 bytes each)")
    print(f"  - NOTE: Keys are stored in plaintext for Python virtualization use")


def cleanup_old_files():
    """
    Clean up old virtualization files to prevent key mismatch.
    Removes all *_v_infos.json and *_params.bin files in current directory.
    """
    current_dir = os.getcwd()

    # Find all old virtualization files
    v_infos_files = glob.glob(os.path.join(current_dir, '*_v_infos.json'))
    params_files = glob.glob(os.path.join(current_dir, '*_params.bin'))
    old_files = v_infos_files + params_files

    if not old_files:
        print("✓ No old virtualization files to clean up")
        return

    # Show files to be deleted
    print("\nWARNING: Found old virtualization files:")
    for f in old_files:
        print(f"  - {os.path.basename(f)}")

    # Ask for confirmation
    response = input("\nDelete these files? (yes/no): ").strip().lower()

    if response == 'yes':
        for f in old_files:
            try:
                os.remove(f)
                print(f"  ✓ Deleted {os.path.basename(f)}")
            except Exception as e:
                print(f"  ✗ Failed to delete {os.path.basename(f)}: {e}")
        print("✓ Cleanup completed")
    else:
        print("✗ Cleanup cancelled by user")
        sys.exit(1)


def main():
    """Main entry point for key generation."""
    print("=" * 70)
    print("NeuralVirtualizer - Key Generation for Multi-Model Virtualization")
    print("=" * 70)

    # Step 1: Clean up old files
    print("\n[Step 1/3] Cleaning up old virtualization files...")
    cleanup_old_files()

    # Step 2: Generate cryptographic parameters
    print("\n[Step 2/3] Generating cryptographic parameters...")
    crypto_params = generate_crypto_params()
    print("✓ Generated modulus values and AES-256 plaintext keys")

    # Step 3: Write to binary file
    print("\n[Step 3/3] Writing to binary file...")
    write_binary_file(crypto_params, 'keys_and_offsets.bin')

    print("\n" + "=" * 70)
    print("SUCCESS: Key generation completed!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Run virtualization.py for each model:")
    print("     python virtualizer/virtualization.py --model_name=<model_name>")
    print("  2. After virtualizing all models, inject keys to C++ backend:")
    print("     python scripts/inject_keys_to_backend.py")
    print("  3. Compile the TFLite backend and run tests")
    print("\nIMPORTANT:")
    print("  - keys_and_offsets.bin contains PLAINTEXT keys")
    print("  - These keys will be used directly by Python for encryption")
    print("  - inject_keys_to_backend.py will obfuscate keys before injecting to C++")
    print("  - Keep keys_and_offsets.bin secure until injection is complete")


if __name__ == '__main__':
    main()
