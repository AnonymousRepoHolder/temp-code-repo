#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
频率分析攻击脚本
基于密文碰撞进行算子类型推断
"""

import json
import os
from collections import defaultdict
from pathlib import Path


def build_cipher_collision_map(data):
    """构建密文碰撞映射"""
    cipher_occurrences = defaultdict(list)

    for model in data['models']:
        model_id = model['model_id']
        for op in model['operators']:
            cipher = op['v_op_code_data']
            op_index = op['index']
            cipher_occurrences[cipher].append((model_id, op_index))

    return cipher_occurrences


def identify_colliding_ciphers(cipher_occurrences):
    """识别碰撞密文和唯一密文"""
    colliding_ciphers = {}
    unique_ciphers = {}

    for cipher, occurrences in cipher_occurrences.items():
        if len(occurrences) >= 2:
            colliding_ciphers[cipher] = occurrences
        else:
            unique_ciphers[cipher] = occurrences

    return colliding_ciphers, unique_ciphers


def generate_predictions(data, colliding_ciphers, unique_ciphers):
    """为所有模型生成预测"""
    predictions = {}

    for model in data['models']:
        model_id = model['model_id']
        model_predictions = []

        for op in model['operators']:
            cipher = op['v_op_code_data']
            op_index = op['index']

            if cipher in colliding_ciphers:
                # 碰撞密文：预测 Top-2（CONV_2D 和 DEPTHWISE_CONV_2D）
                topk = [
                    {"type": "CONV_2D", "p": 0.5},
                    {"type": "DEPTHWISE_CONV_2D", "p": 0.5}
                ]
            else:
                # 唯一密文：输出 UNKNOWN
                topk = [{"type": "UNKNOWN", "p": 1.0}]

            model_predictions.append({
                "idx": op_index,
                "topk": topk
            })

        predictions[model_id] = model_predictions

    return predictions


def write_prediction_files(predictions, exp_name, output_dir):
    """写入预测文件"""
    exp_output_dir = Path(output_dir) / exp_name
    exp_output_dir.mkdir(parents=True, exist_ok=True)

    for model_id, model_predictions in predictions.items():
        output_file = exp_output_dir / f"{model_id}_predict.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(model_predictions, f, indent=2, ensure_ascii=False)
        print(f"已写入: {output_file}")


def main(input_file, output_dir):
    """主函数"""
    print(f"正在读取输入文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 确定实验名称
    experiment = data['experiment']
    if experiment == 'exp2_enc_only':
        exp_name = 'exp2'
    elif experiment == 'exp3_full_protection':
        exp_name = 'exp3'
    else:
        raise ValueError(f"未知的实验类型: {experiment}")

    print(f"\n实验类型: {experiment} -> 输出目录: {exp_name}")

    # 步骤 1: 构建密文碰撞映射
    print("\n步骤 1: 构建密文碰撞映射...")
    cipher_occurrences = build_cipher_collision_map(data)
    print(f"总密文数: {len(cipher_occurrences)}")

    # 步骤 2: 识别碰撞密文和唯一密文
    print("\n步骤 2: 识别碰撞密文和唯一密文...")
    colliding_ciphers, unique_ciphers = identify_colliding_ciphers(cipher_occurrences)
    print(f"碰撞密文数 (出现次数 >= 2): {len(colliding_ciphers)}")
    print(f"唯一密文数 (出现次数 = 1): {len(unique_ciphers)}")

    # 计算碰撞率
    total_ops = sum(len(model['operators']) for model in data['models'])
    colliding_ops = sum(len(occs) for occs in colliding_ciphers.values())
    collision_rate = colliding_ops / total_ops
    print(f"碰撞率: {collision_rate:.4f} ({colliding_ops}/{total_ops})")

    # 步骤 3: 生成预测
    print("\n步骤 3: 生成预测...")
    predictions = generate_predictions(data, colliding_ciphers, unique_ciphers)

    # 步骤 4: 写入预测文件
    print("\n步骤 4: 写入预测文件...")
    write_prediction_files(predictions, exp_name, output_dir)

    print(f"\n完成！已为 {len(predictions)} 个模型生成预测文件。")

    # 统计信息
    print("\n统计信息:")
    for model_id, model_predictions in predictions.items():
        colliding_count = sum(1 for pred in model_predictions if pred['topk'][0]['type'] != 'UNKNOWN')
        unknown_count = sum(1 for pred in model_predictions if pred['topk'][0]['type'] == 'UNKNOWN')
        print(f"  {model_id}: {len(model_predictions)} 个算子 (碰撞: {colliding_count}, 未知: {unknown_count})")


if __name__ == '__main__':
    # 处理 exp2
    print("=" * 80)
    print("处理 Exp2: 仅加密（高碰撞率）")
    print("=" * 80)
    input_file_exp2 = '/Users/zhujunhua/School/虚拟化/论文/NeuralVirtualizer_Anonymous/ablation/attack_inputs/exp2_attack_input.json'
    output_dir = '/Users/zhujunhua/School/虚拟化/论文/NeuralVirtualizer_Anonymous/ablation/attack_outputs'
    main(input_file_exp2, output_dir)

    # 处理 exp3
    print("\n" + "=" * 80)
    print("处理 Exp3: 完整保护（低碰撞率）")
    print("=" * 80)
    input_file_exp3 = '/Users/zhujunhua/School/虚拟化/论文/NeuralVirtualizer_Anonymous/ablation/attack_inputs/exp3_attack_input.json'
    main(input_file_exp3, output_dir)
