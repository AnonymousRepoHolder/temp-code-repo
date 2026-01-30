# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 The NeuralVirtualizer Authors

"""
Generate a single-subgraph TFLite model that covers ops supported by DynaMO
for model obfuscation testing. The model outputs multiple tensors (one per branch)
to avoid graph pruning and ensure each op is preserved in the FlatBuffer.

Outputs: ../tflite_model/security_test.tflite (relative to this script)

Covered ops (DynaMO-supported only):
- CONV_2D, DEPTHWISE_CONV_2D, MAX_POOL_2D, AVERAGE_POOL_2D
- FULLY_CONNECTED, SOFTMAX
- CONCATENATION, ADD
- RESHAPE, MEAN, SQUEEZE, RESIZE_BILINEAR

Notes:
- Use float32 everywhere, keep simple, valid, and runnable.
- Each branch end is returned as a model output to prevent pruning.
- Removed unsupported ops: MUL, SUB, GELU, BATCH_MATMUL, TRANSPOSE, GATHER,
  SPLIT, SHAPE, STRIDED_SLICE, PACK, RANGE, EXPAND_DIMS, CAST
"""

import os
import pathlib
import tensorflow as tf


# Fixed shapes for simplicity and converter friendliness
SHAPE_4D = (1, 8, 8, 3)
SHAPE_4D_B = (1, 8, 8, 3)
SHAPE_FC_IN = (1, 16)


# Layers created outside tf.function to ensure variables/weights are captured
conv2d_layer = tf.keras.layers.Conv2D(
    filters=4, kernel_size=3, padding='same', use_bias=True, activation=None,
    name='conv2d')
dwconv2d_layer = tf.keras.layers.DepthwiseConv2D(
    kernel_size=3, padding='same', depth_multiplier=1, use_bias=True,
    name='depthwise_conv2d')
dense_layer = tf.keras.layers.Dense(8, use_bias=True, activation=None, name='fc')


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=SHAPE_4D, dtype=tf.float32, name='input_4d'),
        tf.TensorSpec(shape=SHAPE_4D_B, dtype=tf.float32, name='input_4d_b'),
        tf.TensorSpec(shape=SHAPE_FC_IN, dtype=tf.float32, name='fc_in'),
    ]
)
def balanced_ops_model(input_4d, input_4d_b, fc_in):
    """Model using only DynaMO-supported operators."""
    # Convolutional / pooling family
    conv_out = conv2d_layer(input_4d)  # CONV_2D
    dw_out = dwconv2d_layer(input_4d)  # DEPTHWISE_CONV_2D
    maxp_out = tf.nn.max_pool2d(input_4d, ksize=2, strides=2, padding='SAME', name='max_pool_2d')
    avgp_out = tf.nn.avg_pool2d(input_4d, ksize=2, strides=2, padding='SAME', name='avg_pool_2d')

    # Elementwise family (only ADD is supported by DynaMO)
    add_out = tf.add(input_4d, input_4d_b, name='add')  # ADD

    # Concatenation along channels
    concat_out = tf.concat([conv_out, dw_out], axis=3, name='concat')  # CONCATENATION

    # Shape/transform family (DynaMO-supported only)
    reshape_out = tf.reshape(add_out, [1, 8 * 8 * 3], name='reshape')  # RESHAPE to [1,192]
    mean_out = tf.reduce_mean(input_4d, axis=[1, 2], keepdims=True, name='mean')  # MEAN [1,1,1,3]
    resize_out = tf.image.resize(input_4d, size=[12, 12], method='bilinear', name='resize_bilinear')
    # SQUEEZE: need a dimension of size 1 to squeeze
    squeeze_out = tf.squeeze(mean_out, axis=[1, 2], name='squeeze')  # [1,3]

    # Linear / activation
    fc_out = dense_layer(fc_in)  # FULLY_CONNECTED
    softmax_out = tf.nn.softmax(fc_out, name='softmax')  # SOFTMAX

    # Return outputs to force retention of all branches
    return (
        conv_out, dw_out, maxp_out, avgp_out,
        add_out, concat_out,
        reshape_out, mean_out, squeeze_out, resize_out,
        fc_out, softmax_out,
    )


def convert_and_save(output_path: str) -> None:
    # Trace concrete function
    concrete_func = balanced_ops_model.get_concrete_function(
        tf.TensorSpec(shape=SHAPE_4D, dtype=tf.float32, name='input_4d'),
        tf.TensorSpec(shape=SHAPE_4D_B, dtype=tf.float32, name='input_4d_b'),
        tf.TensorSpec(shape=SHAPE_FC_IN, dtype=tf.float32, name='fc_in'),
    )

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    # Keep converter simple and avoid aggressive graph transforms
    converter.optimizations = []
    converter.allow_custom_ops = False
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    # Convert
    tflite_model = converter.convert()

    # Ensure directory exists and write file
    out_path = pathlib.Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Wrote TFLite model: {out_path} ({len(tflite_model)} bytes)")


def main():
    # Resolve output path relative to this script
    script_dir = pathlib.Path(__file__).resolve().parent
    repo_root = script_dir.parent  # ../
    out_path = repo_root / 'tflite_model' / 'security_test.tflite'
    convert_and_save(str(out_path))


if __name__ == '__main__':
    # Limit TF logging verbosity for cleaner output
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    main()
