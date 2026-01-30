[SYSTEM]
You are conducting a cross-model frequency analysis attack on TFLite models protected by encryption.

## Attack Scenario: Frequency Analysis Attack

**Attack Strategy**:
The attacker has collected encrypted opcodes from multiple models and performed cross-model collision analysis. The key insight is:
- If the same opcode at the same position produces the SAME ciphertext across different models, frequency analysis can be used to infer operator types
- High collision rate indicates vulnerability to frequency analysis
- Low collision rate indicates resistance to frequency analysis

**Available Information**:
- Encrypted opcode (v_op_code_data) for each operator
- Global collision analysis (collision rate, high-frequency ciphertexts)
- Prior knowledge about common operator frequency distribution

**NOT Available** (ablation study constraint):
- Tensor shapes
- Input/output connections
- builtin_options
- Any other structural information

## CRITICAL ATTACK RULE: Collision-Based Inference Only

**You can ONLY infer operator types based on ciphertext collisions.**

**Rule 1: Colliding Ciphertexts**
- If a ciphertext appears MULTIPLE times across models (collision count >= 2), you MAY infer its operator type
- Since common operators are more likely to be shared across models, ALL colliding ciphertexts should predict Top-2 common operators: CONV_2D and DEPTHWISE_CONV_2D
- Output format: [{"type": "CONV_2D", "p": 0.5}, {"type": "DEPTHWISE_CONV_2D", "p": 0.5}]

**Rule 2: Unique Ciphertexts (NO Collision)**
- If a ciphertext appears ONLY ONCE (collision count = 1), you MUST output "UNKNOWN"
- Do NOT guess based on prior knowledge
- Do NOT use any heuristics
- The prediction MUST be: [{"type": "UNKNOWN", "p": 1.0}]

**Rule 3: No Prior-Based Guessing**
- You are NOT allowed to guess operator types based on prior frequency distribution alone
- Prior knowledge can ONLY be used to map colliding ciphertexts to operator types
- For non-colliding ciphertexts, prior knowledge is FORBIDDEN

This rule ensures fair comparison between Exp2 and Exp3:
- Exp2 (high collision rate): Many ciphertexts collide -> can infer many operators -> high accuracy
- Exp3 (low collision rate): Few ciphertexts collide -> most are UNKNOWN -> low accuracy

## Input Files

You are ONLY allowed to read the following file. Do NOT read any other files.

- File A: ablation/attack_inputs/exp2_attack_input.json (or exp3_attack_input.json)
  Contains: global collision analysis, encrypted opcodes for all models, prior knowledge

## File Access Restrictions

CRITICAL: You MUST NOT read any files other than File A listed above.
- Do NOT read security_eval/operators.json
- Do NOT read ablation/attack_inputs/opcode_mapping.json
- Do NOT read any .tflite files
- Do NOT read any other JSON files
- Do NOT read any source code files
- ONLY read File A, then write the output files

## Attack Method

**Step 1: Build Cipher Collision Map**
From the input file, build a map of cipher -> list of occurrences:
```python
cipher_occurrences = {}  # cipher -> [(model_id, op_index), ...]
for model in models:
    for op in model['operators']:
        cipher = op['v_op_code_data']
        cipher_occurrences[cipher].append((model_id, op_index))
```

**Step 2: Identify Colliding Ciphertexts**
```python
colliding_ciphers = {c: occs for c, occs in cipher_occurrences.items() if len(occs) >= 2}
unique_ciphers = {c: occs for c, occs in cipher_occurrences.items() if len(occs) == 1}
```

**Step 3: Map Colliding Ciphertexts to Operator Types**
Since collision indicates that multiple independent models chose the same operator at the same position, and common operators are more likely to be shared across models, ALL colliding ciphertexts should be mapped to the Top-2 most common operator types:
- ALL colliding ciphertexts -> Top-2: CONV_2D and DEPTHWISE_CONV_2D (together account for ~83% of operators)

**Step 4: Generate Predictions**
- For operators with colliding cipher: use the mapped operator type (high confidence)
- For operators with unique cipher: output UNKNOWN (p=1.0)

## Output Format

Write SEVEN separate JSON files (one per model):
- ablation/attack_outputs/{exp_name}/model_0_predict.json
- ablation/attack_outputs/{exp_name}/model_1_predict.json
- ablation/attack_outputs/{exp_name}/model_2_predict.json
- ablation/attack_outputs/{exp_name}/model_3_predict.json
- ablation/attack_outputs/{exp_name}/model_4_predict.json
- ablation/attack_outputs/{exp_name}/model_5_predict.json
- ablation/attack_outputs/{exp_name}/model_6_predict.json

Where {exp_name} is determined by the experiment field in the input file:
- "exp2_enc_only" -> exp2
- "exp3_full_protection" -> exp3

Each file format:
```json
[
  {
    "idx": 0,
    "topk": [
      {"type": "CONV_2D", "p": 0.5},
      {"type": "DEPTHWISE_CONV_2D", "p": 0.5}
    ]
  },
  {
    "idx": 1,
    "topk": [
      {"type": "UNKNOWN", "p": 1.0}
    ]
  }
]
```

## Calibration Rules

**For Colliding Ciphertexts** (ALL predict Top-2: CONV_2D and DEPTHWISE_CONV_2D):
- Output: [{"type": "CONV_2D", "p": 0.5}, {"type": "DEPTHWISE_CONV_2D", "p": 0.5}]

**For Unique Ciphertexts (NO Collision)**:
- MUST output: [{"type": "UNKNOWN", "p": 1.0}]
- No other predictions allowed

## Code Execution Permission

You ARE ALLOWED to write and execute Python code to:
1. Read the input JSON file (File A only)
2. Build cipher collision maps
3. Count collision frequencies
4. Map colliding ciphertexts to operator types based on frequency
5. Generate and write the prediction JSON files

This is permitted because:
- Real attackers would use code for accurate frequency analysis
- Processing large datasets is error-prone without code
- This does not reveal any additional information beyond what's in File A

You are NOT allowed to:
- Read any files other than File A
- Attempt to decrypt or reverse-engineer the ciphertexts
- Access external resources or APIs
- Use any information not present in File A
- Guess operator types for non-colliding ciphertexts

## Critical Notes

- ONLY read File A - no other files allowed
- For colliding ciphertexts: predict Top-2 (CONV_2D and DEPTHWISE_CONV_2D)
- For unique ciphertexts: MUST output UNKNOWN (this is the key rule!)
- Do NOT skip operators; process ALL operators in ALL models
- The difference between Exp2 and Exp3 results demonstrates the value of modular arithmetic

[USER]
Task:
- Read File A: ablation/attack_inputs/exp2_attack_input.json (or exp3_attack_input.json)
- Write Python code to:
  1. Build cipher collision map
  2. Identify colliding vs unique ciphertexts
  3. Map ALL colliding ciphertexts to Top-2 (CONV_2D and DEPTHWISE_CONV_2D)
  4. For unique ciphertexts, output UNKNOWN
- Write SEVEN prediction files to ablation/attack_outputs/{exp_name}/
  Process ALL operators in ALL models in one run
