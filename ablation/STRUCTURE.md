# Ablation Study File Structure

This document describes the file structure and workflow for the ablation experiments.

## Directory Structure

```
ablation/
├── README.md                           # Experiment overview and instructions
├── STRUCTURE.md                        # This file
├── crypto_utils.py                     # Cryptographic utilities (shared)
│
├── exp1_mod_only.py                    # Exp1: Modular arithmetic only (no encryption)
├── exp2_enc_only.py                    # Exp2: Encryption only (no modular arithmetic)
├── exp3_full_protection.py             # Exp3: Full protection (baseline)
│
├── prepare_attack_inputs.py            # Generate attack input files for LLM
├── evaluate_ablation.py                # Evaluate LLM predictions
│
├── EXP1-ATTACK-Prompt.md              # Attack prompt for Exp1
├── EXP2-EXP3-ATTACK-Prompt.md         # Shared attack prompt for Exp2 and Exp3
│
├── outputs/                            # Virtualization outputs
│   ├── exp1/                          # Plaintext v_codes (Exp1)
│   │   ├── model_0_v_codes.json
│   │   ├── model_1_v_codes.json
│   │   ├── model_2_v_codes.json
│   │   ├── model_3_v_codes.json
│   │   ├── model_4_v_codes.json
│   │   ├── model_5_v_codes.json
│   │   ├── model_6_v_codes.json
│   │   └── brute_force_results.json   # Modulus search results
│   ├── exp2/                          # Encrypted opcodes (no modular arithmetic)
│   │   ├── model_0_v_infos.json
│   │   ├── model_1_v_infos.json
│   │   ├── model_2_v_infos.json
│   │   ├── model_3_v_infos.json
│   │   ├── model_4_v_infos.json
│   │   ├── model_5_v_infos.json
│   │   └── model_6_v_infos.json
│   └── exp3/                          # Encrypted opcodes (full protection)
│       ├── model_0_v_infos.json
│       ├── model_1_v_infos.json
│       ├── model_2_v_infos.json
│       ├── model_3_v_infos.json
│       ├── model_4_v_infos.json
│       ├── model_5_v_infos.json
│       └── model_6_v_infos.json
│
├── attack_inputs/                      # LLM attack input files
│   ├── exp1_attack_input.json         # Exp1: 3 models + candidate modulus
│   ├── exp2_attack_input.json         # Exp2: 3 models + collision analysis
│   └── exp3_attack_input.json         # Exp3: 3 models (baseline)
│
├── attack_outputs/                     # LLM prediction outputs (user-generated)
│   ├── exp1/
│   │   ├── model_0_predict.json
│   │   ├── model_1_predict.json
│   │   ├── model_2_predict.json
│   │   ├── model_3_predict.json
│   │   ├── model_4_predict.json
│   │   ├── model_5_predict.json
│   │   └── model_6_predict.json
│   ├── exp2/
│   │   ├── model_0_predict.json
│   │   ├── model_1_predict.json
│   │   ├── model_2_predict.json
│   │   ├── model_3_predict.json
│   │   ├── model_4_predict.json
│   │   ├── model_5_predict.json
│   │   └── model_6_predict.json
│   └── exp3/
│       ├── model_0_predict.json
│       ├── model_1_predict.json
│       ├── model_2_predict.json
│       ├── model_3_predict.json
│       ├── model_4_predict.json
│       ├── model_5_predict.json
│       └── model_6_predict.json
│
├── ground_truth/                       # Ground truth operator types
│   ├── model_0_real.json
│   ├── model_1_real.json
│   ├── model_2_real.json
│   ├── model_3_real.json
│   ├── model_4_real.json
│   ├── model_5_real.json
│   └── model_6_real.json
│
└── results/                            # Evaluation results
    ├── exp1_results.json              # Exp1 evaluation metrics
    ├── exp2_results.json              # Exp2 evaluation metrics
    ├── exp3_results.json              # Exp3 evaluation metrics
    └── ablation_summary.json          # Comprehensive summary
```

## Workflow

### Phase 1: Virtualization

Generate virtualized opcodes for each experiment:

```bash
# Exp1: Modular arithmetic only (plaintext v_codes)
python ablation/exp1_mod_only.py

# Exp2: Encryption only (deterministic ciphertexts)
python ablation/exp2_enc_only.py

# Exp3: Full protection (randomized ciphertexts)
python ablation/exp3_full_protection.py
```

**Output**: `outputs/exp1/`, `outputs/exp2/`, `outputs/exp3/`

### Phase 2: Prepare Attack Inputs

Generate attack input files for LLM:

```bash
python ablation/prepare_attack_inputs.py
```

**Output**:
- `attack_inputs/exp1_attack_input.json`
- `attack_inputs/exp2_attack_input.json`
- `attack_inputs/exp3_attack_input.json`
- `ground_truth/model_*_real.json`

### Phase 3: LLM Attack (Manual)

For Exp1:
1. Provide `EXP1-ATTACK-Prompt.md` to LLM
2. LLM reads `attack_inputs/exp1_attack_input.json`
3. LLM writes `attack_outputs/exp1/model_*_predict.json`

For Exp2 and Exp3 (using the same prompt for fair comparison):
1. Provide `EXP2-EXP3-ATTACK-Prompt.md` to LLM
2. For Exp2: LLM reads `attack_inputs/exp2_attack_input.json`
   - High collision rate (~60%) enables effective frequency analysis
3. For Exp3: LLM reads `attack_inputs/exp3_attack_input.json`
   - Low collision rate (<1%) makes frequency analysis ineffective
4. LLM writes to the appropriate output directory based on experiment name

### Phase 4: Evaluation

Evaluate LLM predictions against ground truth:

```bash
python ablation/evaluate_ablation.py
```

**Output**:
- `results/exp1_results.json`
- `results/exp2_results.json`
- `results/exp3_results.json`
- `results/ablation_summary.json`

## Key Design Principles

1. **Model Anonymization**: Models are referred to as `model_0`, `model_1`, `model_2` (not by name) to prevent LLM from using external prior knowledge.

2. **Single Input File**: Each experiment uses a single input file containing all 3 models, allowing LLM to process them in one pass.

3. **Minimal Prior Knowledge**: Only common operator frequency distribution is provided, no model-specific information.

4. **Unified Evaluation**: All experiments use the same evaluation logic (Top-1/3/5 accuracy).

## Model Selection

The 7 models are selected to ensure sufficient cross-model collision samples:
- model_0: mobilenet - Mobile architecture with DEPTHWISE_CONV_2D
- model_1: squeezenet - Lightweight with Fire modules
- model_2: mnasnet - NAS-designed architecture
- model_3: efficientnet - Efficient scaling architecture
- model_4: posenet - Pose estimation model
- model_5: fruit - Classification model
- model_6: skin - Classification model

Using 7 models provides enough collision samples for effective frequency analysis in Exp2, while demonstrating that Exp3 (with modular arithmetic) has very few collisions.

## File Format Specifications

### Virtualization Output Format

**Exp1** (`outputs/exp1/model_*_v_codes.json`):
```json
{
  "model_id": "model_0",
  "experiment": "exp1_mod_only",
  "operators": [
    {
      "index": 0,
      "v_code": 12845,
      "real_code": 3,
      "op_type": "CONV_2D"
    }
  ]
}
```

**Exp2/Exp3** (`outputs/exp2|3/model_*_v_infos.json`):
```json
{
  "model_id": "model_0",
  "experiment": "exp2_enc_only",
  "operators": [
    {
      "index": 0,
      "v_op_code_data": "xK8p2A==",
      "real_code": 3,
      "op_type": "CONV_2D"
    }
  ]
}
```

### Attack Input Format

See individual prompt files for detailed specifications.

### Prediction Output Format

```json
[
  {
    "idx": 0,
    "topk": [
      {"type": "CONV_2D", "p": 0.75},
      {"type": "DEPTHWISE_CONV_2D", "p": 0.20},
      {"type": "UNKNOWN", "p": 0.05}
    ]
  }
]
```

### Ground Truth Format

```json
[
  {
    "idx": 0,
    "type": "CONV_2D"
  }
]
```

### Evaluation Results Format

```json
{
  "experiment": "exp1_mod_only",
  "models": [
    {
      "model_id": "model_0",
      "top1_acc": 0.782,
      "top3_acc": 0.891,
      "top5_acc": 0.934
    }
  ],
  "average": {
    "top1_acc": 0.765,
    "top3_acc": 0.876,
    "top5_acc": 0.921
  }
}
```
