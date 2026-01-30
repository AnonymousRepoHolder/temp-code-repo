# Ablation Study: Security Contribution of Protection Mechanisms

This directory contains ablation experiments that demonstrate the security contribution of each protection mechanism in NeuralVirtualizer's opcode virtualization.

## Overview

NeuralVirtualizer protects operator type codes using two mechanisms:
1. **Modular Arithmetic**: `v_code = r_op * modulus + real_code` (random r_op per encryption)
2. **AES-256-CTR Encryption**: `ciphertext = AES_CTR(v_code, key, nonce)`

This ablation study shows that **both mechanisms are necessary** for robust protection against LLM-based attacks.

## Experiments

### Experiment 1: Modular Arithmetic Only

**Configuration**: Modular arithmetic without encryption (plaintext v_code output)

**Attack Method**: Brute-force modulus search + LLM inference
- Attacker brute-forces candidate modulus values
- Computes potential real_code = v_code % modulus
- LLM combines recovered codes with other signals (shapes, builtin_options) to infer operator types

**Expected Result**: High attack accuracy (~75-85% Top-1)

**Conclusion**: Modular arithmetic alone is insufficient; the modulus can be recovered and combined with other signals.

### Experiment 2: Encryption Only

**Configuration**: AES encryption without modular arithmetic (`v_code = real_code`)

**Attack Method**: Cross-model frequency analysis + LLM inference
- Collect encrypted opcodes from multiple models
- Analyze ciphertext collision patterns (same opcode at same position → same ciphertext)
- LLM infers operator types from frequency distribution and context patterns

**Vulnerability**: Without the random factor `r_op`, same `(op_index, real_code)` produces the same ciphertext across different models.

**Expected Result**: Moderate attack accuracy (~50-65% Top-1)

**Conclusion**: Encryption alone is vulnerable to frequency analysis when the plaintext is deterministic.

### Experiment 3: Full Protection (Baseline)

**Configuration**: Modular arithmetic + AES encryption (complete protection)

**Attack Method**: Same as Exp2 (cross-model frequency analysis + LLM inference)
- Uses the SAME prompt as Exp2 for fair comparison
- LLM attempts frequency analysis but finds very low collision rate (<1%)
- LLM must rely on local signals (shapes, patterns) instead

**Protection**: Random `r_op` factor ensures that even the same opcode at the same position produces different ciphertexts.

**Expected Result**: Low attack accuracy (~20-30% Top-1) due to ineffective frequency analysis

**Conclusion**: The combination of both mechanisms provides robust protection. The modular arithmetic effectively eliminates ciphertext collisions, making frequency analysis ineffective.

## Quick Start

### Step 1: Generate Virtualized Opcodes

```bash
# Run all experiments
python ablation/exp1_mod_only.py
python ablation/exp2_enc_only.py
python ablation/exp3_full_protection.py
```

### Step 2: Prepare Attack Inputs

```bash
python ablation/prepare_attack_inputs.py
```

This generates:
- `attack_inputs/exp1_attack_input.json`
- `attack_inputs/exp2_attack_input.json`
- `attack_inputs/exp3_attack_input.json`
- `ground_truth/model_*_real.json`

### Step 3: LLM Attack (Manual)

For Exp1:
1. Provide the prompt to your LLM: `EXP1-ATTACK-Prompt.md`
2. LLM reads: `attack_inputs/exp1_attack_input.json`
3. LLM writes: `attack_outputs/exp1/model_*_predict.json`

For Exp2 and Exp3 (using the same prompt for fair comparison):
1. Provide the prompt to your LLM: `EXP2-EXP3-ATTACK-Prompt.md`
2. For Exp2: LLM reads `attack_inputs/exp2_attack_input.json`
3. For Exp3: LLM reads `attack_inputs/exp3_attack_input.json`
4. LLM writes to the appropriate output directory based on experiment name

### Step 4: Evaluate Results

```bash
python ablation/evaluate_ablation.py
```

This generates:
- `results/exp1_results.json`
- `results/exp2_results.json`
- `results/exp3_results.json`
- `results/ablation_summary.json`

## Key Design Principles

### Model Anonymization

Models are referred to as `model_0`, `model_1`, `model_2` (not by actual names) to prevent LLM from using external prior knowledge about specific models. This ensures fair evaluation based solely on the provided signals.

### Minimal Prior Knowledge

Only common operator frequency distribution is provided as prior knowledge:
- CONV_2D: ~22%
- DEPTHWISE_CONV_2D: ~18%
- RELU: ~15%
- RELU6: ~12%
- ADD: ~8%
- Others: ~25%

This represents knowledge an attacker could reasonably obtain from public model repositories.

### Unified Evaluation

All experiments use the same evaluation metrics:
- **Top-1 Accuracy**: Fraction of operators where the first predicted type equals ground truth
- **Top-3 Accuracy**: Ground truth appears within the first 3 predicted types
- **Top-5 Accuracy**: Ground truth appears within the first 5 predicted types

## Expected Results

| Experiment | Collision Rate | Top-1 Acc | Security Level |
|-----------|----------------|-----------|----------------|
| Exp1 (Mod Only) | N/A | ~100% | Weak |
| Exp2 (Enc Only) | ~60% | 50-70% | Moderate |
| Exp3 (Full) | <1% | <5% | Strong |

Note: Exp2 and Exp3 use strict collision-based inference only. For non-colliding ciphertexts, the prediction is UNKNOWN. This ensures fair comparison and demonstrates the value of modular arithmetic in eliminating ciphertext collisions.

## Security Mechanism Roles

| Mechanism | Primary Role | Weakness Alone |
|-----------|--------------|----------------|
| Modular Arithmetic | Randomization (salt effect) | Modulus can be brute-forced |
| AES Encryption | Confidentiality | Deterministic encryption enables frequency analysis |
| **Combined** | **Synergistic protection** | - |

## Dependencies

- Python 3.8+
- TensorFlow 2.x
- cryptography
- flatc (FlatBuffers compiler)
- jsonrepair (npm package)

## Test Models

The experiments use 7 anonymized models selected from the main evaluation:
- model_0: Mobile architecture with depthwise convolutions
- model_1: Lightweight architecture with fire modules
- model_2: NAS-designed architecture
- model_3: Efficient scaling architecture
- model_4: Pose estimation model
- model_5: Classification model (fruit)
- model_6: Classification model (skin)

Using 7 models ensures sufficient cross-model collision samples for effective frequency analysis in Exp2.

## File Structure

See [STRUCTURE.md](STRUCTURE.md) for detailed file structure and format specifications.

## Notes

- All experiments use the same cryptographic parameters (shared key) to enable fair comparison
- Exp2 and Exp3 use simplified virtualization (opcode only) compared to the full system
- The ablation study focuses on opcode protection; other protections (structure, connections) are not included
