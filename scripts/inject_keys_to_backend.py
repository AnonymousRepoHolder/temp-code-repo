#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 The NeuralVirtualizer Authors

"""
Key Injection Script for C++ Backend

This script reads plaintext cryptographic parameters from keys_and_offsets.bin,
generates XOR obfuscation parameters, obfuscates the keys, and injects them into
the C++ virtualized model parser.

Workflow:
1. Read plaintext keys from keys_and_offsets.bin (168 bytes)
2. Generate XOR master mask and domain offsets (in-memory only)
3. Obfuscate plaintext keys using XOR parameters
4. Inject obfuscated keys and XOR parameters to virtualized_model_parser.cc
5. Delete keys_and_offsets.bin (burn after use)

Note: XOR obfuscation is solely for protecting C++ binary from static analysis.
      It does NOT affect the encryption/decryption logic.
"""

import os
import sys
import secrets
import struct
import fileinput


def read_crypto_params(binary_path='keys_and_offsets.bin'):
    """
    Read plaintext cryptographic parameters from binary file.

    Args:
        binary_path: Path to the 168-byte binary file

    Returns:
        dict: Contains modulus values, plaintext AES keys, and nonce bases
    """
    if not os.path.exists(binary_path):
        print(f"ERROR: {binary_path} not found!")
        print("Please run scripts/generate_keys.py first to generate keys.")
        sys.exit(1)

    file_size = os.path.getsize(binary_path)
    if file_size != 168:
        print(f"ERROR: {binary_path} has invalid size: {file_size} bytes (expected 168)")
        sys.exit(1)

    domains = ['op', 'param', 'graph', 'shape']
    crypto_params = {
        'aes_keys': {},
        'nonce_bases': {}
    }

    with open(binary_path, 'rb') as f:
        # Read modulus values (8 bytes)
        modulus_data = f.read(8)
        crypto_params['op_modulus'], crypto_params['conn_modulus'] = struct.unpack('<II', modulus_data)

        # Read 4 domains × (32-byte plaintext key + 8-byte nonce)
        for domain in domains:
            crypto_params['aes_keys'][domain] = f.read(32)  # Plaintext key
            crypto_params['nonce_bases'][domain] = f.read(8)  # Nonce base

    print(f"✓ Read {binary_path} ({file_size} bytes)")
    print(f"  - Modulus: op_modulus={crypto_params['op_modulus']}, "
          f"conn_modulus={crypto_params['conn_modulus']}")
    print(f"  - 4 plaintext AES-256 keys loaded")
    print(f"  - 4 nonce bases loaded")

    return crypto_params


def generate_xor_params():
    """
    Generate XOR obfuscation parameters (in-memory only, not persisted).

    Returns:
        dict: Contains master mask and domain offsets
    """
    xor_params = {
        'master_mask': secrets.token_bytes(32),  # 32-byte master XOR mask
        'domain_offsets': {
            'op': secrets.randbits(8),
            'param': secrets.randbits(8),
            'graph': secrets.randbits(8),
            'shape': secrets.randbits(8)
        }
    }

    print("✓ Generated XOR obfuscation parameters (in-memory)")
    print(f"  - Master mask: 32 bytes")
    print(f"  - Domain offsets: 4 × 1 byte")

    return xor_params


def obfuscate_key(plaintext_key, master_mask, domain_offset):
    """
    Obfuscate a plaintext key using XOR with master mask and domain offset.

    Formula: obfuscated[i] = plaintext[i] XOR master_mask[i] XOR domain_offset

    Args:
        plaintext_key: 32-byte plaintext AES key
        master_mask: 32-byte master XOR mask
        domain_offset: 1-byte domain-specific offset

    Returns:
        bytes: 32-byte obfuscated key
    """
    obfuscated_key = bytearray(32)
    for i in range(32):
        obfuscated_key[i] = plaintext_key[i] ^ master_mask[i] ^ domain_offset
    return bytes(obfuscated_key)


def format_cpp_byte_array(data):
    """
    Format binary data as C++ std::array initialization.

    Args:
        data: bytes object

    Returns:
        str: C++ array initialization string
    """
    hex_values = ', '.join(f'0x{b:02x}' for b in data)
    return f'{{{hex_values}}}'


def inject_to_cpp(crypto_params, xor_params, cpp_path):
    """
    Inject obfuscated keys and XOR parameters to C++ parser file.

    Args:
        crypto_params: Dictionary with plaintext keys and nonce bases
        xor_params: Dictionary with XOR master mask and domain offsets
        cpp_path: Path to virtualized_model_parser.cc
    """
    if not os.path.exists(cpp_path):
        print(f"ERROR: {cpp_path} not found!")
        sys.exit(1)

    domains = ['op', 'param', 'graph', 'shape']

    # Obfuscate all plaintext keys
    obfuscated_keys = {}
    for domain in domains:
        obfuscated_keys[domain] = obfuscate_key(
            crypto_params['aes_keys'][domain],
            xor_params['master_mask'],
            xor_params['domain_offsets'][domain]
        )

    print(f"\n✓ Obfuscated all plaintext keys using XOR parameters")

    # Prepare injection markers and values
    injections = {
        # Modulus values
        'op_modulus_ = 251;': f'op_modulus_ = {crypto_params["op_modulus"]};',
        'conn_modulus_ = 50021;': f'conn_modulus_ = {crypto_params["conn_modulus"]};',

        # XOR parameters
        'xor_master_mask_ = {};': f'xor_master_mask_ = {format_cpp_byte_array(xor_params["master_mask"])};',
        'xor_op_offset_ = 0x00;': f'xor_op_offset_ = 0x{xor_params["domain_offsets"]["op"]:02x};',
        'xor_param_offset_ = 0x00;': f'xor_param_offset_ = 0x{xor_params["domain_offsets"]["param"]:02x};',
        'xor_graph_offset_ = 0x00;': f'xor_graph_offset_ = 0x{xor_params["domain_offsets"]["graph"]:02x};',
        'xor_shape_offset_ = 0x00;': f'xor_shape_offset_ = 0x{xor_params["domain_offsets"]["shape"]:02x};',

        # Obfuscated encryption keys
        'op_encryption_key_ = {};': f'op_encryption_key_ = {format_cpp_byte_array(obfuscated_keys["op"])};',
        'param_encryption_key_ = {};': f'param_encryption_key_ = {format_cpp_byte_array(obfuscated_keys["param"])};',
        'graph_encryption_key_ = {};': f'graph_encryption_key_ = {format_cpp_byte_array(obfuscated_keys["graph"])};',
        'shape_encryption_key_ = {};': f'shape_encryption_key_ = {format_cpp_byte_array(obfuscated_keys["shape"])};',

        # Nonce bases (not obfuscated, copied directly)
        'op_nonce_base_ = {};': f'op_nonce_base_ = {format_cpp_byte_array(crypto_params["nonce_bases"]["op"])};',
        'param_nonce_base_ = {};': f'param_nonce_base_ = {format_cpp_byte_array(crypto_params["nonce_bases"]["param"])};',
        'graph_nonce_base_ = {};': f'graph_nonce_base_ = {format_cpp_byte_array(crypto_params["nonce_bases"]["graph"])};',
        'shape_nonce_base_ = {};': f'shape_nonce_base_ = {format_cpp_byte_array(crypto_params["nonce_bases"]["shape"])};',
    }

    # Perform in-place injection
    injection_count = 0
    for line in fileinput.input(cpp_path, inplace=True):
        modified = False
        for marker, replacement in injections.items():
            if marker in line:
                # Preserve leading whitespace
                indent = line[:len(line) - len(line.lstrip())]
                print(f'{indent}{replacement}')
                modified = True
                injection_count += 1
                break
        if not modified:
            print(line, end='')

    # Verify injection count
    expected_count = 15  # 2 modulus + 1 master_mask + 4 offsets + 4 keys + 4 nonces
    if injection_count != expected_count:
        print(f"\n✗ ERROR: Injection incomplete!")
        print(f"  - Expected {expected_count} parameters, but only injected {injection_count}")
        print(f"  - This indicates the C++ file structure may have changed")
        print(f"  - Please check {os.path.basename(cpp_path)} for marker placeholders")
        sys.exit(1)

    print(f"\n✓ Injected {injection_count} parameters to {os.path.basename(cpp_path)}")
    print(f"  - 2 modulus values (op_modulus, conn_modulus)")
    print(f"  - 1 XOR master mask (32 bytes)")
    print(f"  - 4 XOR domain offsets (1 byte each)")
    print(f"  - 4 obfuscated encryption keys (32 bytes each)")
    print(f"  - 4 nonce bases (8 bytes each)")


def delete_binary_file(binary_path='keys_and_offsets.bin'):
    """
    Delete the plaintext key file after successful injection.

    Args:
        binary_path: Path to the binary file to delete
    """
    if os.path.exists(binary_path):
        os.remove(binary_path)
        print(f"\n✓ Deleted {binary_path} (burn after use)")
    else:
        print(f"\n⚠ {binary_path} not found, skipping deletion")


def main():
    """Main entry point for key injection."""
    print("=" * 70)
    print("NeuralVirtualizer - Key Injection to C++ Backend")
    print("=" * 70)

    # Step 1: Read plaintext keys
    print("\n[Step 1/5] Reading plaintext cryptographic parameters...")
    crypto_params = read_crypto_params('keys_and_offsets.bin')

    # Step 2: Generate XOR parameters
    print("\n[Step 2/5] Generating XOR obfuscation parameters...")
    xor_params = generate_xor_params()

    # Step 3: Locate C++ parser file
    print("\n[Step 3/5] Locating C++ parser file...")
    cpp_path = os.path.join(
        'tensorflow_src',
        'tensorflow',
        'lite',
        'parser',
        'virtualized_model_parser.cc'
    )
    if not os.path.exists(cpp_path):
        print(f"ERROR: {cpp_path} not found!")
        print("Please ensure you are running this script from the NeuralVirtualizer root directory.")
        sys.exit(1)
    print(f"✓ Found {cpp_path}")

    # Step 4: Inject to C++ file
    print("\n[Step 4/5] Injecting obfuscated keys to C++ parser...")
    inject_to_cpp(crypto_params, xor_params, cpp_path)

    # Step 5: Delete binary file
    print("\n[Step 5/5] Cleaning up temporary files...")
    delete_binary_file('keys_and_offsets.bin')

    print("\n" + "=" * 70)
    print("SUCCESS: Key injection completed!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Compile TFLite backend:")
    print("     cd ${TF_UPSTREAM_ROOT}")
    print("     bazel build -c opt //tensorflow/lite:libtensorflowlite.so")
    print("  2. Build test program:")
    print("     cd test_comparison/build && cmake .. && make")
    print("  3. Run tests:")
    print("     ./test_virtualized_interpreter --model_name=<model_name>")
    print("\nIMPORTANT:")
    print("  - Plaintext keys have been obfuscated before injection")
    print("  - C++ runtime will deobfuscate keys back to plaintext for decryption")
    print("  - Python encryption key == C++ decryption key (both plaintext)")


if __name__ == '__main__':
    main()
