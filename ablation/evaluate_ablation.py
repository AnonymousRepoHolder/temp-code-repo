#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 The NeuralVirtualizer Authors

"""
Evaluate LLM predictions for ablation experiments.

This script evaluates LLM predictions against ground truth using the same
evaluation logic as the main security evaluation (ModelObfuscator).
"""

import json
import os
import sys
from collections import defaultdict

from common_utils import SCRIPT_DIR, PROJECT_ROOT, MODEL_IDS

# Add ModelObfuscator path for evaluation logic
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'security_eval', 'ModelObfuscator', 'opTypes'))
from compare_eval import compute_metrics

# Top-2 common operators for collision-based inference
TOP2_OPERATORS = ['CONV_2D', 'DEPTHWISE_CONV_2D']


def compute_collision_precision(exp_name):
    """
    Compute precision metrics for collision-based inference.

    This function calculates:
    1. Collision precision: accuracy of Top-2 prediction on colliding ciphertexts
    2. Non-collision precision: accuracy of Top-2 prediction on non-colliding ciphertexts (baseline)
    3. Information gain: collision precision - non-collision precision

    Args:
        exp_name: Experiment name ('exp2' or 'exp3')

    Returns:
        dict: Precision metrics
    """
    # Load attack input to get collision information
    input_path = os.path.join(SCRIPT_DIR, 'attack_inputs', f'{exp_name}_attack_input.json')
    if not os.path.exists(input_path):
        return None

    with open(input_path, 'r', encoding='utf-8') as f:
        attack_input = json.load(f)

    # Build cipher collision map
    cipher_to_positions = defaultdict(list)
    for model in attack_input['models']:
        model_id = model['model_id']
        for op in model['operators']:
            cipher = op['v_op_code_data']
            idx = op['index']
            cipher_to_positions[cipher].append((model_id, idx))

    # Load ground truth
    ground_truths = {}
    for model_id in MODEL_IDS:
        real_path = os.path.join(SCRIPT_DIR, 'ground_truth', f'{model_id}_real.json')
        if os.path.exists(real_path):
            with open(real_path, 'r', encoding='utf-8') as f:
                ground_truths[model_id] = {item['idx']: item['type'] for item in json.load(f)}

    # Calculate precision for collision and non-collision ciphertexts
    collision_top2_correct = 0
    collision_total = 0
    non_collision_top2_correct = 0
    non_collision_total = 0

    for cipher, positions in cipher_to_positions.items():
        is_collision = len(positions) >= 2

        for model_id, idx in positions:
            if model_id not in ground_truths:
                continue

            actual_type = ground_truths[model_id].get(idx, 'UNKNOWN')
            is_top2 = actual_type in TOP2_OPERATORS

            if is_collision:
                collision_total += 1
                if is_top2:
                    collision_top2_correct += 1
            else:
                non_collision_total += 1
                if is_top2:
                    non_collision_top2_correct += 1

    # Calculate metrics
    collision_precision = collision_top2_correct / collision_total if collision_total > 0 else None
    non_collision_precision = non_collision_top2_correct / non_collision_total if non_collision_total > 0 else 0
    collision_rate = collision_total / (collision_total + non_collision_total) if (collision_total + non_collision_total) > 0 else 0

    # Information gain is only meaningful when there are collisions
    if collision_precision is not None:
        information_gain = collision_precision - non_collision_precision
    else:
        information_gain = None

    return {
        'collision_total': collision_total,
        'collision_top2_correct': collision_top2_correct,
        'collision_precision': collision_precision,
        'non_collision_total': non_collision_total,
        'non_collision_top2_correct': non_collision_top2_correct,
        'non_collision_precision': non_collision_precision,
        'collision_rate': collision_rate,
        'information_gain': information_gain
    }


def evaluate_experiment(exp_name):
    """
    Evaluate one ablation experiment.

    Args:
        exp_name: Experiment name ('exp1', 'exp2', or 'exp3')

    Returns:
        dict: Evaluation results
    """
    print(f'Evaluating {exp_name}...')

    results_dir = os.path.join(SCRIPT_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    per_model_results = []

    for model_id in MODEL_IDS:
        pred_path = os.path.join(SCRIPT_DIR, 'attack_outputs', exp_name, f'{model_id}_predict.json')
        real_path = os.path.join(SCRIPT_DIR, 'ground_truth', f'{model_id}_real.json')
        eval_path = os.path.join(results_dir, f'{exp_name}_{model_id}_eval.json')

        if not os.path.exists(pred_path):
            print(f'  Warning: {pred_path} not found, skipping {model_id}')
            continue

        if not os.path.exists(real_path):
            print(f'  Error: {real_path} not found, skipping {model_id}')
            continue

        # Use the same evaluation logic as ModelObfuscator
        result = compute_metrics(model_id, pred_path, real_path, eval_path)

        per_model_results.append({
            'model_id': model_id,
            'top1_acc': result['metrics']['top1_acc'],
            'top3_acc': result['metrics']['top3_acc'],
            'top5_acc': result['metrics']['top5_acc']
        })

        print(f'  {model_id}: Top-1={result["metrics"]["top1_acc"]:.3f}, '
              f'Top-3={result["metrics"]["top3_acc"]:.3f}, '
              f'Top-5={result["metrics"]["top5_acc"]:.3f}')

    if not per_model_results:
        print(f'  No results to evaluate for {exp_name}')
        return None

    # Calculate average
    avg_result = {
        'experiment': exp_name,
        'models': per_model_results,
        'average': {
            'top1_acc': sum(r['top1_acc'] for r in per_model_results) / len(per_model_results),
            'top3_acc': sum(r['top3_acc'] for r in per_model_results) / len(per_model_results),
            'top5_acc': sum(r['top5_acc'] for r in per_model_results) / len(per_model_results)
        }
    }

    # Save summary
    summary_path = os.path.join(results_dir, f'{exp_name}_results.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(avg_result, f, indent=2)

    print(f'  Average: Top-1={avg_result["average"]["top1_acc"]:.3f}, '
          f'Top-3={avg_result["average"]["top3_acc"]:.3f}, '
          f'Top-5={avg_result["average"]["top5_acc"]:.3f}')
    print(f'  Saved to: {summary_path}')

    return avg_result


def generate_summary(exp1_result, exp2_result, exp3_result, exp2_precision, exp3_precision):
    """
    Generate comprehensive ablation study summary.

    Args:
        exp1_result: Exp1 evaluation results
        exp2_result: Exp2 evaluation results
        exp3_result: Exp3 evaluation results
        exp2_precision: Exp2 collision precision metrics
        exp3_precision: Exp3 collision precision metrics

    Returns:
        dict: Summary dictionary
    """
    summary = {
        'ablation_study': 'Security Contribution of Protection Mechanisms',
        'experiments': {
            'exp1_mod_only': {
                'description': 'Modular arithmetic only (no encryption)',
                'attack_method': 'Brute-force modulus search + LLM inference',
                'results': exp1_result['average'] if exp1_result else None
            },
            'exp2_enc_only': {
                'description': 'Encryption only (no modular arithmetic)',
                'attack_method': 'Cross-model frequency analysis + LLM inference',
                'results': exp2_result['average'] if exp2_result else None,
                'collision_analysis': exp2_precision
            },
            'exp3_full_protection': {
                'description': 'Full protection (modular arithmetic + encryption)',
                'attack_method': 'LLM inference with local signals only',
                'results': exp3_result['average'] if exp3_result else None,
                'collision_analysis': exp3_precision
            }
        }
    }

    # Calculate improvements
    if all([exp1_result, exp2_result, exp3_result]):
        summary['improvements'] = {
            'exp1_to_exp3': {
                'top1_reduction': round((exp1_result['average']['top1_acc'] - exp3_result['average']['top1_acc']) * 100, 2),
                'top3_reduction': round((exp1_result['average']['top3_acc'] - exp3_result['average']['top3_acc']) * 100, 2),
                'top5_reduction': round((exp1_result['average']['top5_acc'] - exp3_result['average']['top5_acc']) * 100, 2)
            },
            'exp2_to_exp3': {
                'top1_reduction': round((exp2_result['average']['top1_acc'] - exp3_result['average']['top1_acc']) * 100, 2),
                'top3_reduction': round((exp2_result['average']['top3_acc'] - exp3_result['average']['top3_acc']) * 100, 2),
                'top5_reduction': round((exp2_result['average']['top5_acc'] - exp3_result['average']['top5_acc']) * 100, 2)
            }
        }

        summary['conclusion'] = {
            'modular_arithmetic_contribution': f'{summary["improvements"]["exp2_to_exp3"]["top1_reduction"]}% Top-1 accuracy reduction',
            'encryption_contribution': f'{summary["improvements"]["exp1_to_exp3"]["top1_reduction"] - summary["improvements"]["exp2_to_exp3"]["top1_reduction"]:.2f}% Top-1 accuracy reduction',
            'synergy': 'Both mechanisms are necessary for robust protection'
        }

    return summary


def main():
    print('=' * 70)
    print('Evaluating Ablation Experiments')
    print('=' * 70)
    print()

    # Evaluate each experiment
    exp1_result = evaluate_experiment('exp1')
    print()

    exp2_result = evaluate_experiment('exp2')
    print()

    exp3_result = evaluate_experiment('exp3')
    print()

    # Compute collision precision metrics
    print('=' * 70)
    print('Computing Collision Precision Metrics')
    print('=' * 70)
    print()

    exp2_precision = compute_collision_precision('exp2')
    exp3_precision = compute_collision_precision('exp3')

    if exp2_precision:
        print('Exp2 (Encryption Only):')
        print(f'  Collision rate: {exp2_precision["collision_rate"]*100:.1f}%')
        print(f'  Collision instances: {exp2_precision["collision_total"]}')
        if exp2_precision["collision_precision"] is not None:
            print(f'  Collision Top-2 precision: {exp2_precision["collision_precision"]*100:.1f}%')
        else:
            print(f'  Collision Top-2 precision: N/A (no collisions)')
        print(f'  Non-collision instances: {exp2_precision["non_collision_total"]}')
        print(f'  Non-collision Top-2 precision: {exp2_precision["non_collision_precision"]*100:.1f}%')
        if exp2_precision["information_gain"] is not None:
            print(f'  Information gain: {exp2_precision["information_gain"]*100:.1f} percentage points')
        else:
            print(f'  Information gain: N/A (no collisions)')
        print()

    if exp3_precision:
        print('Exp3 (Full Protection):')
        print(f'  Collision rate: {exp3_precision["collision_rate"]*100:.1f}%')
        print(f'  Collision instances: {exp3_precision["collision_total"]}')
        if exp3_precision["collision_precision"] is not None:
            print(f'  Collision Top-2 precision: {exp3_precision["collision_precision"]*100:.1f}%')
        else:
            print(f'  Collision Top-2 precision: N/A (no collisions)')
        print(f'  Non-collision instances: {exp3_precision["non_collision_total"]}')
        print(f'  Non-collision Top-2 precision: {exp3_precision["non_collision_precision"]*100:.1f}%')
        if exp3_precision["information_gain"] is not None:
            print(f'  Information gain: {exp3_precision["information_gain"]*100:.1f} percentage points')
        else:
            print(f'  Information gain: N/A (no collisions)')
        print()

    # Generate summary
    if any([exp1_result, exp2_result, exp3_result]):
        summary = generate_summary(exp1_result, exp2_result, exp3_result, exp2_precision, exp3_precision)

        summary_path = os.path.join(SCRIPT_DIR, 'results', 'ablation_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        print('=' * 70)
        print('ABLATION STUDY SUMMARY')
        print('=' * 70)
        print()

        if all([exp1_result, exp2_result, exp3_result]):
            print(f'{"Experiment":<30} {"Top-1 Acc":<12} {"Top-3 Acc":<12} {"Top-5 Acc":<12}')
            print('-' * 70)
            print(f'{"Exp1 (Mod Only)":<30} {exp1_result["average"]["top1_acc"]:.3f}{"":<8} '
                  f'{exp1_result["average"]["top3_acc"]:.3f}{"":<8} '
                  f'{exp1_result["average"]["top5_acc"]:.3f}')
            print(f'{"Exp2 (Enc Only)":<30} {exp2_result["average"]["top1_acc"]:.3f}{"":<8} '
                  f'{exp2_result["average"]["top3_acc"]:.3f}{"":<8} '
                  f'{exp2_result["average"]["top5_acc"]:.3f}')
            print(f'{"Exp3 (Full Protection)":<30} {exp3_result["average"]["top1_acc"]:.3f}{"":<8} '
                  f'{exp3_result["average"]["top3_acc"]:.3f}{"":<8} '
                  f'{exp3_result["average"]["top5_acc"]:.3f}')
            print()

            if exp2_precision and exp2_precision["information_gain"] is not None:
                print('Collision-Based Information Gain (Exp2):')
                print(f'  Collision Top-2 precision: {exp2_precision["collision_precision"]*100:.1f}%')
                print(f'  Non-collision Top-2 precision: {exp2_precision["non_collision_precision"]*100:.1f}%')
                print(f'  Information gain: {exp2_precision["information_gain"]*100:.1f} percentage points')
                print()

            print('Accuracy Reduction (vs Full Protection):')
            print(f'  Exp1 -> Exp3: {summary["improvements"]["exp1_to_exp3"]["top1_reduction"]:.2f}% Top-1 reduction')
            print(f'  Exp2 -> Exp3: {summary["improvements"]["exp2_to_exp3"]["top1_reduction"]:.2f}% Top-1 reduction')
            print()
            print(f'Saved summary to: {summary_path}')
        else:
            print('Incomplete results. Please ensure all experiments have predictions.')
    else:
        print('No results to evaluate. Please run LLM attacks first.')


if __name__ == '__main__':
    main()
