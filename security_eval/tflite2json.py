# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 The NeuralVirtualizer Authors
import os
import sys
# Show only WARNING and above logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import tensorflow as tf
import shutil

# Simplify the model JSON file by removing the buffers field
def reduce_size_json(json_file):
    # Use a temporary file for streaming to avoid reading the entire file at once
    temp_file = json_file + '.tmp'
    try:
        with open(json_file, 'r', encoding='utf-8') as input_file, \
            open(temp_file, 'w', encoding='utf-8') as output_file:
            keep_sign = True
            line_count = 0
            # Process line by line to avoid loading the entire file into memory
            for line in input_file:
                line_count += 1
                # Print progress every 10000 lines
                if line_count % 10000 == 0:
                    print(f"Processing JSON file progress: {line_count} lines")
                if 'buffers:' in line:
                    keep_sign = False
                    output_file.write('}\n')
                    break  # Stop immediately after finding buffers without further processing
                if keep_sign:
                    output_file.write(line)
        # Atomically replace the original file
        shutil.move(temp_file, json_file)
        print(f"JSON file simplification completed, processed {line_count} lines")
    except Exception as e:
        # Clean up temporary files
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise e

# Command line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='1', help='name of the model')
opt = parser.parse_args()

# Resolve important paths relative to this script so execution cwd doesn't matter
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# security_eval is directly under repo root, so go up one level
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
SCHEMA_FBS = os.path.join(REPO_ROOT, 'schema.fbs')

# Input: repo_root/tflite_model/<model>.tflite
TFLITE_DIR = os.path.join(REPO_ROOT, 'tflite_model')
# Output: security_eval/model_json/<model>.json
OUT_DIR = os.path.join(SCRIPT_DIR, 'model_json')
os.makedirs(OUT_DIR, exist_ok=True)

model_name = opt.model_name + '.tflite'
model_tflite_path = os.path.join(TFLITE_DIR, model_name)

# Parse the TFLite model into a JSON object
print("Start converting the TFLite model to JSON...")
# Use absolute paths and direct flatc output to the OUT_DIR directory
flatc_cmd = 'flatc -t -o "{out_dir}" "{schema}" -- "{input_file}"'.format(
    out_dir=OUT_DIR,
    schema=SCHEMA_FBS,
    input_file=model_tflite_path,
)
ret = os.system(flatc_cmd)
if ret != 0:
    raise RuntimeError(f"flatc failed with exit code {ret} for command: {flatc_cmd}")

print("Start simplifying the JSON file...")
out_json_path = os.path.join(OUT_DIR, os.path.splitext(model_name)[0] + '.json')
reduce_size_json(out_json_path)

print("Fix JSON formatting...")
jsonrepair_cmd = 'jsonrepair "{json_path}" --overwrite'.format(json_path=out_json_path)
ret = os.system(jsonrepair_cmd)
if ret != 0:
    raise RuntimeError(f"jsonrepair failed with exit code {ret} for command: {jsonrepair_cmd}")

# Memory-optimized JSON loading
print("Start loading the JSON file into memory...")
json_file_path = out_json_path

# Check file size
file_size = os.path.getsize(json_file_path)
print(f"JSON file size: {file_size / (1024*1024):.2f} MB")