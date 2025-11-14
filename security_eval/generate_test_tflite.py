# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 The NeuralVirtualizer Authors

"""
Generate a single-subgraph TFLite model that covers a balanced set of ops
for security_eval. The model outputs multiple tensors (one per branch) to
avoid graph pruning and ensure each op is preserved in the FlatBuffer.

Outputs: ../tflite_model/security_test.tflite (relative to this script)

Covered ops (one instance each, unless naturally multi-output):
- CONV_2D, DEPTHWISE_CONV_2D, MAX_POOL_2D, AVERAGE_POOL_2D
- FULLY_CONNECTED, SOFTMAX, GELU, BATCH_MATMUL
- CONCATENATION, ADD, MUL, SUB
- RESHAPE, MEAN, SQUEEZE, RESIZE_BILINEAR
- TRANSPOSE, GATHER, SPLIT, SHAPE, STRIDED_SLICE, PACK, RANGE, EXPAND_DIMS, CAST

Notes:
- Use float32 everywhere (except CAST output), keep simple, valid, and runnable.
- Tie some parameters to input shapes (RANGE, etc.) to avoid constant folding.
- Each branch end is returned as a model output to prevent pruning.
"""

import os
import pathlib
import tensorflow as tf


# Fixed shapes for simplicity and converter friendliness
SHAPE_4D = (1, 8, 8, 3)
SHAPE_4D_B = (1, 8, 8, 3)
SHAPE_BMM_X = (1, 4, 8)
SHAPE_BMM_Y = (1, 8, 6)
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
        tf.TensorSpec(shape=SHAPE_BMM_X, dtype=tf.float32, name='bmm_x'),
        tf.TensorSpec(shape=SHAPE_BMM_Y, dtype=tf.float32, name='bmm_y'),
        tf.TensorSpec(shape=SHAPE_FC_IN, dtype=tf.float32, name='fc_in'),
    ]
)
def balanced_ops_model(input_4d, input_4d_b, bmm_x, bmm_y, fc_in):
    # Convolutional / pooling family
    conv_out = conv2d_layer(input_4d)  # CONV_2D
    dw_out = dwconv2d_layer(input_4d)  # DEPTHWISE_CONV_2D
    maxp_out = tf.nn.max_pool2d(input_4d, ksize=2, strides=2, padding='SAME', name='max_pool_2d')
    avgp_out = tf.nn.avg_pool2d(input_4d, ksize=2, strides=2, padding='SAME', name='avg_pool_2d')

    # Elementwise family
    add_out = tf.add(input_4d, input_4d_b, name='add')  # ADD
    mul_out = tf.multiply(input_4d, input_4d_b, name='mul')  # MUL
    sub_out = tf.subtract(input_4d, input_4d_b, name='sub')  # SUB

    # Concatenation along channels
    concat_out = tf.concat([add_out, sub_out], axis=3, name='concat')  # CONCATENATION

    # Shape/transform family
    reshape_out = tf.reshape(add_out, [1, 8 * 8 * 3], name='reshape')  # RESHAPE to [1,192]
    mean_out = tf.reduce_mean(input_4d, axis=[1, 2], keepdims=False, name='mean')  # [1,3]
    exd_out = tf.expand_dims(mean_out, axis=1, name='expand_dims')  # EXPAND_DIMS -> [1,1,3]
    resize_out = tf.image.resize(input_4d, size=[12, 12], method='bilinear', name='resize_bilinear')
    transpose_out = tf.transpose(input_4d, perm=[0, 3, 1, 2], name='transpose')  # [1,3,8,8]
    gather_out = tf.gather(input_4d, indices=[0, 2], axis=3, name='gather')  # [1,8,8,2]
    # strided_slice to downsample spatial dims by stride 2
    slice_out = tf.strided_slice(input_4d,
                                 begin=[0, 0, 0, 0], end=[1, 8, 8, 3], strides=[1, 2, 2, 1],
                                 name='strided_slice')  # [1,4,4,3]
    split_out = tf.split(input_4d, num_or_size_splits=3, axis=3, name='split')  # 3 outputs [1,8,8,1]
    # SQUEEZE on a split branch to produce a unique shape not used elsewhere (avoid aliasing with mean)
    squeeze_out = tf.squeeze(split_out[0], axis=[3], name='squeeze_from_split')  # [1,8,8]
    pack_out = tf.stack([add_out, mul_out], axis=0, name='pack')  # [2,8,8,3]
    shape_out = tf.shape(input_4d, out_type=tf.int32, name='shape')  # [4]

    # RANGE with dependency on input shape to avoid (most) constant folding in practice
    w_dim = shape_out[2]  # 8
    range_out = tf.range(start=0, limit=w_dim, delta=2, name='range')  # [4]

    # Linear / activation / matmul
    fc_out = dense_layer(fc_in)  # FULLY_CONNECTED
    softmax_out = tf.nn.softmax(fc_out, name='softmax')  # SOFTMAX
    gelu_out = tf.nn.gelu(fc_out, approximate=True, name='gelu')  # GELU
    bmm_out = tf.matmul(bmm_x, bmm_y, name='batch_matmul')  # BATCH_MATMUL, shape [1,4,6]

    # CAST branch to int32 (kept separate)
    cast_out = tf.cast(input_4d, tf.int32, name='cast')  # CAST

    # Return many outputs to force retention of all branches
    # Note: `split_out` yields a tuple of 3 tensors; include them explicitly.
    return (
        conv_out, dw_out, maxp_out, avgp_out,
        add_out, mul_out, sub_out, concat_out,
        reshape_out, mean_out, exd_out, squeeze_out, resize_out,
        transpose_out, gather_out, slice_out,
        split_out[0], split_out[1], split_out[2],
        pack_out, shape_out, range_out,
        fc_out, softmax_out, gelu_out, bmm_out,
        cast_out,
    )


def convert_and_save(output_path: str) -> None:
    # Trace concrete function
    concrete_func = balanced_ops_model.get_concrete_function(
        tf.TensorSpec(shape=SHAPE_4D, dtype=tf.float32, name='input_4d'),
        tf.TensorSpec(shape=SHAPE_4D_B, dtype=tf.float32, name='input_4d_b'),
        tf.TensorSpec(shape=SHAPE_BMM_X, dtype=tf.float32, name='bmm_x'),
        tf.TensorSpec(shape=SHAPE_BMM_Y, dtype=tf.float32, name='bmm_y'),
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
