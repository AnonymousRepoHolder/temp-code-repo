[SYSTEM]
You are analyzing TFLite models protected by modular arithmetic virtualization (NO encryption).

## Attack Scenario: Modulus Brute-Force Attack

**Protection Status**:
- Modular arithmetic: v_code = r_op * modulus + real_code (where r_op is random)
- Encryption: REMOVED (v_code is plaintext)

**Attacker's Achievement**:
The attacker has successfully brute-forced the modulus value through heuristic scoring.
For each operator, you have:
1. The plaintext virtualized code (v_code)
2. The recovered modulus value

**Your Task**:
For each operator in each model, infer the operator type by:
1. Computing real_code = v_code % modulus
2. Looking up the operator type from the opcode mapping

## Input Files

You are ONLY allowed to read the following two files. Do NOT read any other files.

- File A: ablation/attack_inputs/exp1_attack_input.json
  Contains: v_code for each operator, modulus, prior knowledge for 3 models

- File B: ablation/attack_inputs/opcode_mapping.json
  Contains: opcode (real_code) to operator type name mapping

## File Access Restrictions

CRITICAL: You MUST NOT read any files other than File A and File B listed above.
- Do NOT read security_eval/operators.json
- Do NOT read any .tflite files
- Do NOT read any other JSON files
- Do NOT read any source code files
- ONLY read File A and File B, then write the output files

## Attack Method

**Step 1: Read the input files**
- Read File A to get v_code for each operator and the modulus
- Read File B to get the opcode to operator type mapping

**Step 2: Compute real_code**
For each operator:
```
real_code = v_code % modulus
```
The real_code is the actual TFLite opcode in range [0, 189].

**Step 3: Lookup operator type**
Use File B (opcode_mapping.json) to map real_code to operator type name.
The mapping is in the format: {"mapping": {"0": "ADD", "3": "CONV_2D", ...}}

**Step 4: Output prediction**
Since the opcode directly identifies the operator type, confidence should be very high (p >= 0.95).

## Output Format

Write THREE separate JSON files (one per model):
- ablation/attack_outputs/exp1/model_0_predict.json
- ablation/attack_outputs/exp1/model_1_predict.json
- ablation/attack_outputs/exp1/model_2_predict.json

Each file format:
```json
[
  {
    "idx": 0,
    "topk": [
      {"type": "CONV_2D", "p": 0.98},
      {"type": "UNKNOWN", "p": 0.02}
    ]
  }
]
```

## Calibration Rules

- **Very high confidence (p >= 0.95)**: The opcode directly maps to operator type
- **Lower confidence**: Only if the opcode is not found in the mapping (use UNKNOWN)
- Always output at least Top-2 with strictly descending probabilities (sum = 1.0)

## Critical Notes

- ONLY read File A and File B - no other files allowed
- This attack is straightforward: compute modulus, lookup opcode
- Expected accuracy: ~100% (opcode directly reveals operator type)
- Do NOT skip operators; process ALL operators in ALL models

## Code Execution Permission

You ARE ALLOWED to write and execute Python code to:
1. Read the input JSON files (File A and File B only)
2. Compute real_code = v_code % modulus for each operator
3. Lookup operator types from the opcode mapping
4. Generate and write the prediction JSON files

This is permitted because:
- Real attackers would use code for accurate computation
- Large number modular arithmetic is error-prone without code
- This does not reveal any additional information

You are NOT allowed to:
- Read any files other than File A and File B
- Attempt to decrypt or reverse-engineer any encryption
- Access external resources or APIs

[USER]
Task:
- Read File A: ablation/attack_inputs/exp1_attack_input.json
- Read File B: ablation/attack_inputs/opcode_mapping.json
- Write Python code to compute real_code = v_code % modulus for each operator
- Lookup operator type from the mapping and generate predictions
- Write THREE prediction files to ablation/attack_outputs/exp1/
  Process ALL operators in ALL models in one run
