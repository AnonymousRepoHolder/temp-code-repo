# NeuralVirtualizer: A Light-weight Neural Virtualization Framework for Hardening On-Device Deep Learning Models

Privacy-preserving virtualized models for TensorFlow Lite with **AES-256-CTR encryption across 4 independent domains** and **XOR-based key obfuscation**, rebuilt natively inside the TFLite runtime and verified against the original models.

NeuralVirtualizer converts an original `.tflite` model into two cryptographically secured assets — `<model>_v_infos.json` (encrypted metadata) and `<model>_params.bin` (raw parameter data) — and integrates a small C++ parser/builder into TensorFlow Lite so the virtualized model can be reconstructed at runtime via a single API: `tflite::FlatBufferModel::BuildFromVirtualizedFiles`.

Beyond per-domain protection, builtin options are processed as a dedicated encrypted block: encode → statistical smart padding → float-slot quantization (×1e3) → AES‑256‑CTR under the Param domain, stored as `v_builtin_options` (Base64). Shape entries carry an extended header `[num_dims:uint32][dtype_code:uint32][param_slot:int32]` with true‑random padding of dims up to a compact K to hide tensor signatures.

For security, the C++ binary stores only XOR‑obfuscated keys; real AES keys are recovered at runtime via `real = obfuscated XOR master_mask XOR domain_offset`, matching the plaintext keys used by the Python virtualizer.

The system supports batch virtualization workflow: (1) generate shared cryptographic keys once with `scripts/generate_keys.py`, (2) virtualize multiple models using the same keys, (3) inject obfuscated keys into C++ backend with `scripts/inject_keys_to_backend.py` after all models are processed. Obfuscated encryption keys (via XOR with master mask and domain-specific offsets) are injected into the C++ source code, then the backend is compiled once to parse all virtualized models in the batch.

A C++ test harness (`test_comparison`) loads both the virtualized and original models and reports functional parity, timing, and RSS memory.

This repository intentionally does not vend the full TensorFlow source tree. Instead, it contains:

- A Python virtualizer (`virtualizer/`, `utils/`) that produces `<model>_v_infos.json` and `<model>_params.bin` from an input `.tflite`.
- Key management scripts (`scripts/`) for generating shared cryptographic keys (`generate_keys.py`) and injecting obfuscated keys into C++ backend (`inject_keys_to_backend.py`).
- A set of C++ sources under `tensorflow_src_parser/` that must be overlaid (copied) into an official TensorFlow v2.18.1 checkout before building the shared library.
- A C++ comparison program (`test_comparison/`) that validates virtualized vs. original behavior.


## Why NeuralVirtualizer

### Security
- **Four independent encryption domains**: Operator types, parameter positions, graph connections, and shape information independently encrypted with AES-256-CTR.
- **XOR key obfuscation**: Real encryption keys protected via `real_key = obfuscated_key XOR master_mask XOR domain_offset` to prevent direct extraction from binary.
- **Random index mapping**: Operators assigned truly random indices (not sequential) to break structural patterns.
- **Smart padding**: Builtin options padded with statistical common values to obscure operator type fingerprints.
- **Builtin options encryption (block-level)**: Builtin options are encoded into fixed/variable-length int32 slots (floats quantized ×1e3), then encrypted as one block under the Param domain key; ciphertext stored as Base64 in `v_builtin_options`.
- **Shape extended header + random padding**: Each encrypted shape block carries an extended header `[num_dims:uint32][dtype_code:uint32][param_slot:int32]`. When `num_dims < K (default 6)`, tail dims are filled with true random integers instead of zeros to mask structure.
- **Parameter entry normalization (≥2/Op)**: Python virtualizer appends dummy entries so every operator has at least two parameter entries; the C++ runtime filters dummy entries (strictly `num_dims==0 && param_slot<0`) after decryption to harden against side channels.
- **Cryptographically secure key generation**: Using Python `secrets` module (CSPRNG).
- **Dynamic key injection**: Obfuscated keys embedded in compiled binary; never stored in JSON or configuration files.

### Integration
- Native TFLite factory: `BuildFromVirtualizedFiles` (parity with `BuildFromFile`).
- Minimal API surface: single entry point, standard TFLite workflow thereafter.
- Automated validation: C++ harness validates accuracy, timing, and memory against original model.

### Performance
- **Comparable inference performance**: Decryption occurs once during model loading; inference uses plaintext data in the same format as native TFLite.
- **Memory-optimized loading**: Zero-copy architecture with optional mmap eliminates duplicate parameter copies.
- Experimental results: performance comparable to native TFLite, functional equivalence verified (MSE/MAE ~0).


## Workflow

Batch virtualization workflow (shared keys for multiple models)

```
Step 1: Generate shared keys
      scripts/generate_keys.py
            ↓
      keys_and_offsets.bin (168 bytes, plaintext)
            ├─ 2 modulus values (op_modulus, conn_modulus)
            ├─ 4 AES-256 keys (32 bytes each, plaintext)
            └─ 4 nonce bases (8 bytes each)

Step 2: Virtualize each model (repeatable for multiple models)
      Original .tflite
            ↓
      flatc (schema.fbs) → JSON (buffers stripped + repaired)
            ↓
      Python Virtualizer (virtualization.py)
            ├─ Fix placeholder op codes: repair `operator_codes` entries where `deprecated_builtin_code==127` using the mapping table
            ├─ Load plaintext keys from keys_and_offsets.bin
            ├─ Apply modular arithmetic (op types, connections)
            ├─ Apply random index mapping to operators
            ├─ Encrypt with AES-256-CTR (4 domains: op/param/graph/shape)
            ├─ Encode builtin_options → smart pad → quantize (float×1e3) → encrypt as a Param-domain block
            ├─ Shuffle encrypted arrays
            └─ Emit <model>_v_infos.json + <model>_params.bin

Step 3: Inject keys to C++ backend (once after all models)
      scripts/inject_keys_to_backend.py
            ├─ Read plaintext keys from keys_and_offsets.bin
            ├─ Generate XOR obfuscation (master_mask + 4 domain_offsets)
            ├─ Obfuscate keys: obfuscated = plaintext XOR master_mask XOR offset
            ├─ Inject to virtualized_model_parser.cc
            └─ Delete keys_and_offsets.bin (burn after use)

Step 4: Compile and use
      C++ Runtime Rebuild (inside TFLite)
            ├─ virtualized_model_parser
            │    ├─ XOR deobfuscation: plaintext = obfuscated XOR master_mask XOR offset
            │    ├─ Base64 decode → AES-256-CTR decrypt (OpenSSL EVP)
            │    ├─ Zero-copy parameter loading (optional mmap for large models)
            │    └─ Reconstruct op/param/graph/shape structures via pointer views
            ├─ virtualized_model_builder → assemble flatbuffer
            └─ BuildFromVirtualizedFiles → FlatBufferModel → Interpreter
```

C++ integration path

```
FlatBufferModel::BuildFromVirtualizedFiles(info, params)
  └─ parser::BuildVirtualizedFlatbuffer(...)  [virtualized_model_loader]
       ├─ VirtualizedModelParser   [virtualized_model_parser.{h,cc}]
       ├─ VirtualizedModelBuilder  [virtualized_model_builder.{h,cc}]
       └─ Returns std::vector<uint8_t> (model flatbuffer)
  └─ VerifyAndBuildFromAllocation(...) → InterpreterBuilder → Interpreter
```


## Repository Layout (before cloning TensorFlow)

- `virtualizer/`
  - `virtualization.py` — main entry; drives flatc → JSON reduction/repair → fixes placeholder op codes (`deprecated_builtin_code==127`) → loads plaintext keys from `keys_and_offsets.bin`; applies modular arithmetic virtualization and random index mapping; encrypts metadata with AES-256-CTR across four independent domains (op/param/graph/shape); emits `<model>_v_infos.json` (encrypted metadata) & `<model>_params.bin` (raw binary).
  - `ops_virtualizer.py` — extracts op types, builtin options, and graph input slots; implements random index mapping with truly random indices; encodes builtin options, applies statistical smart padding and float-slot quantization (×1e3), then encrypts the whole builtin-options block under the Param domain; encrypts operator type codes under the Op domain; performs shape encryption using an extended header `[num_dims][dtype_code][param_slot]` and random padding for dims; all ciphertexts are Base64-encoded.
  - `params_virtualizer.py` — extracts parameters (float32/int32), writes `<model>_params.bin`; encrypts parameter position information (start_pos, length) using AES-256-CTR; encodes ciphertext as Base64.
  - `graph_virtualizer.py` — builds forward connections and branches from JSON; encrypts graph connection IDs using AES-256-CTR; encodes ciphertext as Base64.

- `utils/`
  - `utils.py` — shared helpers (builtin op code map, dtype→JSON conversion) and robust extraction from Interpreter + JSON; post-fixes unresolved input/output tensor mappings to ensure type safety.

- `scripts/`
  - `generate_keys.py` — generates shared cryptographic parameters for batch virtualization; creates `keys_and_offsets.bin` (168 bytes) containing 2 modulus values, 4 AES-256 plaintext keys (32 bytes each), and 4 nonce bases (8 bytes each); cleans up old virtualization files to prevent key mismatch.
  - `inject_keys_to_backend.py` — reads plaintext keys from `keys_and_offsets.bin`; generates XOR obfuscation parameters (master_mask + 4 domain_offsets); obfuscates keys using `obfuscated_key = plaintext_key XOR master_mask XOR domain_offset`; injects obfuscated keys, XOR parameters, modulus values, and nonce bases into `virtualized_model_parser.cc`; deletes `keys_and_offsets.bin` (burn after use).

- `tensorflow_src_parser/` (overlay files to copy into a TensorFlow v2.18.1 checkout)
  - `tensorflow/lite/parser/`
    - `virtualized_model_parser.{h,cc}` — reads `<model>_v_infos.json` (encrypted metadata) / `<model>_params.bin` (raw binary); implements XOR deobfuscation to recover real keys at runtime (`real_key = obfuscated_key XOR master_mask XOR domain_offset`); decrypts operator types, parameter positions, graph connections, and shape information using AES-256-CTR with OpenSSL EVP API; embedding obfuscated encryption keys (4 × 32-byte), nonce bases (4 × 8-byte), XOR master mask (32-byte), and domain offsets (4 × 1-byte) are injected by `scripts/inject_keys_to_backend.py`; implements Base64 decoding; reconstructs op/param/graph/shape; validates shapes; **zero-copy parameter loading via pointer views (`DevirtualizedParam.data_ptr` points directly to `params_array_` or mmap region)**; optional mmap support eliminates read() syscall and buffer copying for large models.
    - `virtualized_model_builder.{h,cc}` — assembles the flatbuffer (Buffers/Tensors/Operators/SubGraph/Model) using pointer views; writes parameter bytes by dtype without lossy conversions; releases heavy artifacts after build.
    - `virtualized_model_loader.{h,cc}` — builds the virtualized flatbuffer and wraps it in a TFLite Allocation; translates errors into `absl::Status`.
    - `BUILD` — Bazel rules for the above libraries.
  - `tensorflow/lite/core/`
    - `model_builder_virtualized.cc` — adds `FlatBufferModel::BuildFromVirtualizedFiles` factory that feeds the existing TFLite verification path.
    - `BUILD` — Bazel rules updated to link the new entry point (must be copied as well).

- `test_comparison/`
  - `test_virtualized_interpreter.cc` — single-model correctness test; supports `--model_name`, `--num_inputs`, `--seed`, and optional `--output_json`.
  - `test_virtualized_only.cc` / `test_original_only.cc` — single-model performance and peak RSS tests; support `--model_name`, `--num_iters`, `--seed`, and optional `--output_json`.
  - `test_suite_runner.cc` — batch runner for `RQ1` / `RQ2`; supports model sets, batch execution, ratio calculation, and markdown/json report generation.
  - `experiment_utils.{h,cc}` — shared single-model execution logic, model-set resolution, and report serialization.
  - `comparison_utils.{h,cc}` — low-level utilities for inputs/outputs, metrics, printing, dynamic tests, and deterministic input seeding.
  - `memory_monitor.{h,cc}` — cross‑platform RSS monitor (Linux: `/proc/self/status`, Windows: `GetProcessMemoryInfo`).
  - `CMakeLists.txt` — builds all test binaries and links Bazel‑built `libtensorflowlite.so`.

- `tflite_model/` — sample `.tflite` models consumed by the virtualizer and test harness.
- `model_converter/` — optional converters to produce `.tflite` (e.g., DistilGPT‑2 helpers), using transformers==4.54.0.
- `Dockerfile` — reproducible build/test environment (Ubuntu 24.04.1 LTS, Python 3.12, Bazelisk, flatc v25.2.10, Clang-20, Node.js 22; includes security evaluation dependencies: ortools, networkx, matplotlib).
- `schema.fbs` — FlatBuffers schema used by `flatc -t` during virtualization.


## Features

### Security
- **AES-256-CTR encryption**: Four independent encryption domains (operator types, parameter positions, graph connections, shape information).
- **XOR key obfuscation**: Real encryption keys protected via `real_key = obfuscated_key XOR master_mask XOR domain_offset` to prevent direct extraction from binary.
- **Random index mapping**: Operators assigned truly random indices (not sequential) to break structural patterns.
- **Smart padding**: Builtin options padded with statistical common values (with special float handling) to obscure operator type fingerprints.
- **Builtin options block encryption**: Encode → pad → quantize (float×1e3) → encrypt under Param-domain key; stored as Base64 `v_builtin_options`.
- **Shape header and random padding**: Extended header `[num_dims][dtype_code][param_slot]`; dims padded with true random integers up to K=6 when short.
- **Normalized parameter entries**: Ensure ≥2 entries per operator by adding dummy entries (filtered at runtime), reducing side-channel leakage.
- **Array shuffling**: The operators array is shuffled after encryption; per‑operator arrays are not shuffled (robustness relies on semantic nonces and randomized indices).
- **Semantic nonces**: Nonce construction based on semantic identifiers (op_index, input_slot) for robustness.
- **Cryptographically secure key generation**: Python `secrets` module (CSPRNG); 256-bit key space.
- **Domain separation**: Independent key-nonce pairs for four data domains.
- **Dynamic key injection**: Obfuscated keys embedded in compiled C++ binary during virtualization; never in JSON.
- **Base64 encoding**: All ciphertext stored as Base64 strings; ensures cross-platform compatibility.
- **Nonce uniqueness**: Strict construction (8-byte base + 8-byte counter) guarantees no reuse.

### Architecture
- Virtualized model rebuild inside TFLite (no plaintext `.tflite` at rest).
- Native factory: `FlatBufferModel::BuildFromVirtualizedFiles` (parity with `BuildFromFile`).
- Data separation: Encrypted metadata (`v_infos.json`) and raw parameters (`params.bin`) useless without keys.
- Single API entry point; minimal integration overhead.

### Performance
- **Comparable inference performance**: Decryption occurs once during model loading; inference uses plaintext data in standard TFLite format.
- **Memory-optimized loading**:
  - Zero-copy architecture: DevirtualizedParam stores pointers (data_ptr) directly to params_array_ or mmap region
  - Optional mmap support: Eliminates read() syscall overhead and intermediate host buffers before building; parameter bytes are still copied once into the final FlatBuffer buffers (chunked copy + immediate page discard on Linux/Windows)
  - Pointer views throughout: No duplicate parameter data allocation during model construction
  - DetachedBuffer zero-copy: builder should preferentially deliver via `flatbuffers::DetachedBuffer` to the TFLite `Allocation` to avoid extra copies; fall back to the vector-based path when necessary
  - Trade-off: Temporary peak memory increase (params + flatbuffer co-exist until construction completes)
- End-to-end validation: accuracy metrics (MSE/MAE/MaxAE/RelMAE/Top-1), inference timing, RSS memory.
- dtype support: float32 + int32 parameters.

### Parameter Dedup & Copy Strategy
- Python side deduplicates identical underlying segments when building `<model>_params.bin` (same `start_pos`/`length` written once).
- C++ builder sorts copy tasks by physical address (`data_ptr`) for sequential mmap access, shares Buffers by identical (`data_ptr`,`size`), and releases pages immediately (Linux: `MADV_DONTNEED`; Windows: `DiscardVirtualMemory`) to reduce peak RSS.


## Security Design

NeuralVirtualizer implements a layered security approach across 4 phases:

### Phase 1: Shape Encryption
- All tensor shapes encrypted with independent AES-256-CTR key-nonce pair.
- Prevents architecture fingerprinting through shape analysis.
- Semantic nonce: `(op_index << 32) | shape_index`.
- Extended plaintext format per entry: `[num_dims:uint32][dtype_code:uint32][param_slot:int32][dims:int32 × L]`.
- Random padding for dims: when `num_dims < K (default 6)`, fill the tail with true random integers to avoid zero-pattern leakage.

### Phase 1b: Builtin Options Encryption (Block-level)
- Encode builtin options per operator; apply statistical smart padding (float slots use safe defaults), quantize floats by ×1e3 into int32 slots.
- Encrypt the entire slot sequence under the Param-domain key using AES‑256‑CTR with semantic nonce `((op_index<<32)|0x00000002)`.
- Store ciphertext as Base64 string `v_builtin_options`.

### Phase 1c: Parameter Entries Normalization
- Enforce at least two parameter entries per operator on the Python side by adding dummy entries.
- On the C++ side, discard dummies strictly defined by `num_dims==0 && param_slot<0` after decrypting the shape header.
- Hardens against information leakage via entry counts and improves uniformity across operators.

### Phase 2: Random Index Mapping & Operators Shuffling
- Operators assigned truly random indices (via `random.sample`) instead of sequential 0,1,2,...
- The operators array is shuffled after encryption; per‑operator `v_forward_connections` are not shuffled, but semantic nonces and randomized indices make structural recovery difficult.
- All 4 encryption domains use semantic nonces (op_index, input_slot) instead of array positions.
- Breaks position-based pattern recognition and prevents structural inference.

### Phase 3: Smart Padding
- Builtin options padded to fixed length (8) using statistical common values.
- Obscures operator type differences (e.g., CONV_2D vs POOL_2D appear similar).
- Special handling for floating-point positions (marked as 'float', padded with safe default 1.0).
- Super-long options preserved without truncation to avoid information loss.

### Phase 4: XOR Key Obfuscation
- Real encryption keys protected via XOR with master mask + domain-specific offsets.
- Binary contains only obfuscated keys; real keys recovered at runtime.
- Formula: `real_key[i] = obfuscated_key[i] XOR master_mask[i] XOR domain_offset`
- Significantly increases difficulty of static key extraction from compiled binary.


## Quickstart (Docker, recommended)

The following flow contains first‑time TensorFlow checkout and overlay steps. Run all commands inside the container.

### 1) Build the Docker image (host)

```bash
docker build -t NeuralVirtualizer .
```

### 2) Start a container (host)

```bash
docker run --rm -it NeuralVirtualizer
```

### 3) First‑time only: clone TensorFlow v2.18.1

```bash
# Use a separate upstream checkout path
export TF_UPSTREAM_ROOT=/NeuralVirtualizer/tensorflow_src
export TF_GIT_REF=v2.18.1

# Clone official TensorFlow (Lite lives under tensorflow/lite)
git clone --depth 1 --branch ${TF_GIT_REF} https://github.com/tensorflow/tensorflow.git ${TF_UPSTREAM_ROOT}
```

### 4) overlay NeuralVirtualizer sources via cp (Execute before key injection and compiling)

```bash
# Overlay NeuralVirtualizer integration using cp (recursive, verbose)
# 1) Parser/Builder/Loader (and their BUILD rules)
cp -rv /NeuralVirtualizer/tensorflow_src_parser/tensorflow/lite/parser \
      ${TF_UPSTREAM_ROOT}/tensorflow/lite/

# 2) Core factory to expose BuildFromVirtualizedFiles + core BUILD
cp -rv /NeuralVirtualizer/tensorflow_src_parser/tensorflow/lite/core \
      ${TF_UPSTREAM_ROOT}/tensorflow/lite/
```

### 5) Batch virtualization (3-phase workflow)

**Phase 1: Generate shared keys (once for all models)**
```bash
python3 scripts/generate_keys.py
# Outputs: keys_and_offsets.bin (168 bytes, plaintext)
# Contains: 2 modulus values + 4 AES-256 keys + 4 nonce bases
```

**Phase 2: Virtualize each model (repeatable)**
```bash
# Virtualize the predefined 11-model evaluation set in one command (`paper11`)
python3 virtualizer/virtualization.py --model_set=paper11

# Or virtualize every .tflite file currently present under tflite_model/
python3 virtualizer/virtualization.py --all_models

# Or virtualize a custom subset
python3 virtualizer/virtualization.py --models=squeezenet,ssd,distilgpt2-official
# Each outputs: <model>_v_infos.json, <model>_params.bin at repo root
# All models share the same keys from keys_and_offsets.bin
```

**Phase 3: Inject keys to C++ backend (once after all models)**
```bash
python3 scripts/inject_keys_to_backend.py
# Reads keys_and_offsets.bin → generates XOR obfuscation
# Injects to virtualized_model_parser.cc → deletes keys_and_offsets.bin
# Note: Ensure Step 4 overlay has been performed so that
# ${TF_UPSTREAM_ROOT}/tensorflow/lite/parser/virtualized_model_parser.cc
# is present with injection markers; otherwise injection will fail.
```

### 6) Configure and build TFLite with the virtualized parser

```bash
cd ${TF_UPSTREAM_ROOT}
bazel version
python3 configure.py      # Press Enter for CPU‑only defaults
bazel build -c opt //tensorflow/lite:libtensorflowlite.so
# Artifact: ${TF_UPSTREAM_ROOT}/bazel-bin/tensorflow/lite/libtensorflowlite.so
```

### 7) Build and run the C++ comparison test

```bash
cd /NeuralVirtualizer/test_comparison
mkdir -p build && cd build
cmake .. && make
cd ../..

# Run the full 11-model evaluation set on x86 and generate RQ1/RQ2 reports
./test_comparison/build/test_suite_runner \
  --suite=all \
  --model_set=paper11 \
  --correctness_inputs=1000 \
  --perf_iters=1000 \
  --seed=42 \
  --platform_label=x86 \
  --output_dir=results-x86
# Note: output_dir is overwritten by default. Add --fail_if_exists if you want a safety check.

# Optional: rerun a single model or a single measurement path
./test_comparison/build/test_virtualized_interpreter \
  --model_name=squeezenet \
  --num_inputs=1000 \
  --seed=42 \
  --output_json=results-x86/raw/squeezenet.correctness.json

./test_comparison/build/test_virtualized_only \
  --model_name=squeezenet \
  --num_iters=1000 \
  --seed=42 \
  --output_json=results-x86/raw/squeezenet.virtualized.json

./test_comparison/build/test_original_only \
  --model_name=squeezenet \
  --num_iters=1000 \
  --seed=42 \
  --output_json=results-x86/raw/squeezenet.original.json
```

Notes

- **First-time setup**: Run steps 3-7 in sequence.
- **Subsequent runs** (start a new virtualization batch):
  - Step 3 (clone TF): Skip - reuse the existing TF checkout.
  - Step 4 (overlay): **Must re-run** before injection & build to ensure parser/builder/loader files with markers are present.
  - Step 5 (virtualization):
    - Phase 1 deletes all old `*_v_infos.json` and `*_params.bin` files, then generates new `keys_and_offsets.bin`.
    - Phase 2: Virtualize **all models** in this batch before proceeding to Phase 3.
    - Phase 3 deletes `keys_and_offsets.bin` after injection - no more models can be added to this batch.
    - If virtualization warns "Connection modulus too small", increase `conn_modulus` in `scripts/generate_keys.py` and restart from Phase 1.
  - Step 6 (build): **Must rebuild** TFLite after Phase 3 to compile the injected keys.
  - Step 7 (test): Prefer `test_suite_runner` for batch execution and report generation; keep the single-model binaries for debugging or partial reruns.
- **Important**: All models in a batch must be virtualized (Phase 2) before running Phase 3 (inject keys). Once Phase 3 is complete, you cannot add more models - you must start a new batch from Phase 1.
- **Android builds & on-device testing**: See the section "Android Device (arm64) Build & Testing" below for SDK/NDK setup, Bazel Android build, and ADB-run steps.

## Manual setup (optional, non‑Docker)

You will need Linux, Python 3, flatc, Bazelisk/Bazel compatible with TF v2.18.1, a C++ toolchain (gcc/clang), and CMake. Also ensure Node.js `jsonrepair` and OpenSSL dev packages are installed.

- Clone TensorFlow v2.18.1 in a separate directory (e.g., `${TF_UPSTREAM_ROOT}`).
- Overlay this repo’s `tensorflow_src_parser` subtrees via `cp -rv` as shown above (parser/ and core/ into `tensorflow/lite/`).
- Install prerequisites: `flatc` (version matching schema), `npm i -g jsonrepair`, and `libssl-dev` on Ubuntu.
- Run `python3 configure.py` → `bazel build -c opt //tensorflow/lite:libtensorflowlite.so`.
- Build `test_comparison` with CMake and run `test_suite_runner` (or any single-model binary) from the repo root.


## Use in your C++ application

Once `libtensorflowlite.so` has been built, use the virtualized assets as a drop‑in alternative to `.tflite`:

```cpp
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"

// 1) Load the virtualized model
std::unique_ptr<tflite::FlatBufferModel> v_model =
    tflite::FlatBufferModel::BuildFromVirtualizedFiles("<model>_v_infos.json", "<model>_params.bin");

// 2) Create an interpreter
std::unique_ptr<tflite::Interpreter> interpreter;
tflite::ops::builtin::BuiltinOpResolver resolver;
tflite::InterpreterBuilder(*v_model, resolver)(&interpreter);

// 3) Allocate tensors and run
interpreter->AllocateTensors();
// ... set inputs, then
interpreter->Invoke();
```

Everything after the factory call mirrors the standard `.tflite` path (`BuildFromFile`).

## Android Device (arm64) Build & Testing

Build Android arm64 artifacts in the container and run functional/performance tests on a real device via ADB. The virtualization workflow (key generation → virtualize → inject keys) stays the same; differences are SDK/NDK installation, Bazel's Android config, and ADB push/run steps.

- Verified versions (adjustable):
  - Android SDK Platform: `platforms;android-34`
  - Build-Tools: `34.0.0`
  - NDK: `25.2.9519653`
  - Minimum API level: `android-24`
  - CMake generator: `Ninja`
  - Example device: Redmi K50 Ultra (Android 14, Snapdragon 8+ Gen1, 12GB RAM)

1) One-time setup inside the container
```bash
apt-get update && apt-get install -y adb openjdk-17-jdk ninja-build

export ANDROID_SDK_ROOT=/opt/android-sdk
mkdir -p "$ANDROID_SDK_ROOT/cmdline-tools"
curl -fL https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip -o "$ANDROID_SDK_ROOT/cmdline-tools/cmdline-tools.zip"
unzip -q "$ANDROID_SDK_ROOT/cmdline-tools/cmdline-tools.zip" -d "$ANDROID_SDK_ROOT/cmdline-tools"
mv "$ANDROID_SDK_ROOT/cmdline-tools/cmdline-tools" "$ANDROID_SDK_ROOT/cmdline-tools/latest"
rm -f "$ANDROID_SDK_ROOT/cmdline-tools/cmdline-tools.zip"
export PATH=$ANDROID_SDK_ROOT/cmdline-tools/latest/bin:$ANDROID_SDK_ROOT/platform-tools:$PATH

yes | sdkmanager --sdk_root=$ANDROID_SDK_ROOT --licenses
sdkmanager --sdk_root=$ANDROID_SDK_ROOT "platform-tools" "platforms;android-34" "build-tools;34.0.0" "ndk;25.2.9519653"
export ANDROID_NDK_HOME=$ANDROID_SDK_ROOT/ndk/25.2.9519653
```

2) Overlay integration and batch virtualization (same as desktop)
```bash
export TF_UPSTREAM_ROOT=/NeuralVirtualizer/tensorflow_src
export TF_GIT_REF=v2.18.1

git clone --depth 1 --branch ${TF_GIT_REF} https://github.com/tensorflow/tensorflow.git ${TF_UPSTREAM_ROOT}
cp -rv /NeuralVirtualizer/tensorflow_src_parser/tensorflow/lite/parser ${TF_UPSTREAM_ROOT}/tensorflow/lite/
cp -rv /NeuralVirtualizer/tensorflow_src_parser/tensorflow/lite/core   ${TF_UPSTREAM_ROOT}/tensorflow/lite/

python3 scripts/generate_keys.py
python3 virtualizer/virtualization.py --model_set=paper11
python3 scripts/inject_keys_to_backend.py
```

3) Build Android arm64 TFLite with Bazel
```bash
cd ${TF_UPSTREAM_ROOT}
bazel version
python3 configure.py     # Choose Android=YES; NDK API level=24; SDK=/opt/android-sdk; NDK=$ANDROID_NDK_HOME
bazel build -c opt --config=android_arm64 //tensorflow/lite:libtensorflowlite.so
# Artifact: ${TF_UPSTREAM_ROOT}/bazel-bin/tensorflow/lite/libtensorflowlite.so
```

4) Build test binaries (CMake + Ninja, arm64)
```bash
cd /NeuralVirtualizer/test_comparison && mkdir -p build-android-arm64 && cd build-android-arm64
cmake -G Ninja .. \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24 \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DTENSORFLOW_LITE_LIB=/NeuralVirtualizer/tensorflow_src/bazel-bin/tensorflow/lite/libtensorflowlite.so
ninja -v
# If your device needs the C++ runtime, also place libc++_shared.so in this folder for later push
```

5) Export artifacts from container to the host
```bash
# Inside the container, pack the artifacts
cd / && tar czf /tmp/neuralvirtualizer_arm64_artifacts.tgz \
  NeuralVirtualizer/tensorflow_src/bazel-bin/tensorflow/lite/libtensorflowlite.so \
  NeuralVirtualizer/test_comparison/build-android-arm64/test_* \
  NeuralVirtualizer/*.json NeuralVirtualizer/*.bin \
  NeuralVirtualizer/tflite_model

# On the host (use docker ps to locate <container_id>)
docker cp <container_id>:/tmp/neuralvirtualizer_arm64_artifacts.tgz /path/to/save/
```

6) Prepare ADB on Windows (Git Bash) and push via USB
```bash
# Extract the tarball
tar -xzf neuralvirtualizer_arm64_artifacts.tgz

# Install/configure ADB if not present
curl -LO https://dl.google.com/android/repository/platform-tools-latest-windows.zip
mkdir -p /c/Android && unzip -q platform-tools-latest-windows.zip -d /c/Android
export PATH="/c/Android/platform-tools:$PATH"
adb version

# Select USB device only (-d) and push files
export MSYS2_ARG_CONV_EXCL="*"
adb -d devices
adb -d shell 'mkdir -p /data/local/tmp/NeuralVirtualizer/tensorflow_src/bazel-bin/tensorflow/lite /data/local/tmp/NeuralVirtualizer/test_comparison/build-android-arm64 /data/local/tmp/NeuralVirtualizer/tflite_model'
adb -d push NeuralVirtualizer/tensorflow_src/bazel-bin/tensorflow/lite/libtensorflowlite.so /data/local/tmp/NeuralVirtualizer/tensorflow_src/bazel-bin/tensorflow/lite/
adb -d push NeuralVirtualizer/test_comparison/build-android-arm64/test_* /data/local/tmp/NeuralVirtualizer/test_comparison/build-android-arm64/
# If needed by your device:
# adb -d push NeuralVirtualizer/test_comparison/build-android-arm64/libc++_shared.so /data/local/tmp/NeuralVirtualizer/test_comparison/build-android-arm64/
adb -d push NeuralVirtualizer/*.json /data/local/tmp/NeuralVirtualizer/ || true
adb -d push NeuralVirtualizer/*.bin  /data/local/tmp/NeuralVirtualizer/ || true
adb -d push NeuralVirtualizer/tflite_model/*.tflite /data/local/tmp/NeuralVirtualizer/tflite_model/ || true
```

7) Run tests on the device
```bash
adb -d shell
cd /data/local/tmp/NeuralVirtualizer
chmod +x test_comparison/build-android-arm64/test_*
export LD_LIBRARY_PATH=/data/local/tmp/NeuralVirtualizer/tensorflow_src/bazel-bin/tensorflow/lite:/data/local/tmp/NeuralVirtualizer/test_comparison/build-android-arm64:$LD_LIBRARY_PATH

# Run the full 11-model evaluation set on the device and generate RQ1/RQ2 reports locally
./test_comparison/build-android-arm64/test_suite_runner \
  --suite=all \
  --model_set=paper11 \
  --correctness_inputs=1000 \
  --perf_iters=1000 \
  --seed=42 \
  --platform_label=arm64 \
  --output_dir=results-arm64
# Note: output_dir is overwritten by default. Add --fail_if_exists if you want a safety check.

# Optional: rerun a single model
./test_comparison/build-android-arm64/test_virtualized_interpreter \
  --model_name=squeezenet \
  --num_inputs=1000 \
  --seed=42 \
  --output_json=results-arm64/raw/squeezenet.correctness.json
```

After the suite finishes, pull the generated reports back to the host:
```bash
adb -d pull /data/local/tmp/NeuralVirtualizer/results-arm64 ./results-arm64
```

Troubleshooting
- Missing `adb`: Install platform-tools and ensure it's on `PATH`.
- `No such file or directory`: Usually due to a missing `libc++_shared.so` or an incomplete `LD_LIBRARY_PATH`.
- Permissions: Use `/data/local/tmp` and run `chmod +x` on test binaries.
- Device selection: Use `adb -d` to force USB device (avoid `-e` emulator).

Performance
- On-device latency and memory closely match desktop behavior; see the generated Android-side `RQ2.md` (e.g., Redmi K50 Ultra, Android 14).


## Testing & metrics

The `test_comparison` suite includes:

**test_virtualized_interpreter**:
- Loads both the virtualized model (`<model>_v_infos.json` + `<model>_params.bin`) and the original `.tflite` file.
- Generates identical random inputs (dtype‑aware) for static/dynamic shapes.
- Supports `--num_inputs=<N>` and `--seed=<S>`; for dynamic models, the total input budget is evenly distributed across the predefined length buckets.
- Reports MSE, MAE, MaxAE, RelMAE, and Top‑1 agreement rate, and can emit a structured JSON file via `--output_json`.

**test_virtualized_only / test_original_only**:
- Isolated performance measurement for each model type.
- Support `--num_iters=<N>` and `--seed=<S>`.
- Report:
  - Timing: model loading time + total inference time.
  - Memory: peak RSS from model loading start through the first inference completion (PeakMemoryTracker with 5ms sampling interval, Linux/Android/Windows).
  - Optional JSON output via `--output_json`.

**test_suite_runner**:
- Runs `RQ1`, `RQ2`, or both in batch mode with a single command.
- Supports `--model_name`, `--models`, `--model_set`, and `--all_models`.
- Generates:
  - `RQ1.md` — model-level functional equivalence table.
  - `RQ2.md` — model-level inference latency and peak RSS table.
  - `summary.json` and `raw/<model>.json` — structured intermediate results.


## Performance & Memory Notes (See generated `RQ2.md` for details)

### Inference Performance
- **One-time decryption cost**: Decryption occurs once during `BuildFromVirtualizedFiles`; inference uses plaintext data thereafter.
- **Performance comparable to native TFLite**: See the generated `RQ2.md` for detailed benchmarks across the selected model set.
- Functional equivalence verified via comprehensive metrics (MSE/MAE/MaxAE/RelMAE/Top-1).

### Loading Performance
- **Initial decryption**: One-time cost during model loading (AES-256-CTR via OpenSSL EVP, Base64 decoding).
- **XOR deobfuscation**: Negligible overhead (4 simple XOR operations on 32-byte keys).
- **Optional mmap**: Eliminates read() syscall and buffer copying for large models (e.g., distilgpt2-official 460MB).
- Total loading overhead dominated by flatbuffer construction and graph analysis, not cryptographic operations.

### Memory Usage
- **Zero-copy architecture**: DevirtualizedParam stores pointers (data_ptr) directly to params_array_ or mmap region, eliminating duplicate parameter copies during model construction.
- **Delayed release strategy**: Parameter data (params_array_/mmap) remains alive until flatbuffer construction completes, then released. This causes temporary peak memory increase during model loading.
- **Model-dependent impact** (see the generated `RQ2.md` for detailed measurements):
  - Small models (<100MB): Minimal peak RSS overhead (typically ≤10%, e.g., squeezenet 24.8/24.5 ≈ 101%, mobilenet 28.0/27.8 ≈ 101%)
  - Large models (distilgpt2-official 460MB): Higher peak RSS (942/627 ≈ 150%) due to simultaneous presence of raw parameters and constructed flatbuffer during loading phase
- **Optional mmap**: Can reduce peak RSS for large models by eliminating params_array_ heap allocation; parameters accessed directly from OS page cache.
- Memory overhead primarily from resident flatbuffer copy and optional delegate workspace.
- Note: Peak RSS measurements capture maximum memory usage from model loading start through the first inference completion (PeakMemoryTracker with 5ms sampling in test_virtualized_only/test_original_only). This keeps the memory metric focused on runtime footprint rather than long-run harness buffering. The peak typically occurs during model loading when both raw parameters and constructed flatbuffer temporarily co-exist.

## Security Evaluation: Comparative Analysis

The `security_eval/` directory contains adversarial testing scripts that evaluate and compare the security of **NeuralVirtualizer** (this work) against **ModelObfuscator** (baseline method from prior work) using state-of-the-art Large Language Models as attackers.

### Comparative Security Testing

ModelObfuscator protects on-device models by obfuscating the original `.tflite` file, while NeuralVirtualizer virtualizes it. We evaluate their resistance to structural recovery attacks by simulating scenarios where adversaries have access to:
- ✅ Obfuscated/virtualized model files on device
- ❌ Compiled inference binary
- ❌ Encryption keys (NeuralVirtualizer only; deleted after virtualization)

**Attack goal**: Recover the model's computational graph structure and operator types using only the obfuscated artifacts.

### Two Baseline Methods Under Test

#### 1. ModelObfuscator (Prior Work - Baseline)
- **Artifact**: `obf.tflite` (recompiled operators with structure preserved)
- **Operator Types**: ❌ Completely unrecognizable (operators recompiled into custom implementations)
- **Graph Structure**: ⚠️ **Vulnerable** - structural patterns remain detectable
- **Test directory**: `security_eval/ModelObfuscator/struct/`

#### 2. NeuralVirtualizer (This Work)
- **Artifact**: `<model>_v_infos.json` (AES-256-CTR encrypted metadata) + `<model>_params.bin` (raw parameters)
- **Operator Types**: ✅ Encrypted under independent AES-256 key (tested in `security_eval/NeuralVirtualizer/opTypes/`)
- **Graph Structure**: ✅ **Fully protected** - LLMs cannot predict edges from encrypted data
- **Test directory**: `security_eval/NeuralVirtualizer/{opTypes,struct}/`

### Evaluation Workflow

#### Phase 1: Prepare Ground Truth Graphs (10 Test Models)

```bash
# Generate ground truth JSONs for the original .tflite artifacts
for model in squeezenet posenet fruit lenet mobilenet skin mnasnet efficientnet ssd depth_estimation; do
  python3 security_eval/tflite2json.py --model_name "$model"
done
# Outputs: security_eval/model_json/<model>.json
```

#### Phase 2: Static File-Level Reverse Engineering Baseline

This baseline compares two artifact types under the same schema-aware static parsing setting:
- `Original .tflite`: direct static recovery from the standard FlatBuffer artifact
- `NeuralVirtualizer artifacts`: direct static recovery from `<model>_v_infos.json + <model>_params.bin`

The static attack scripts intentionally operate on delivered files only. For NeuralVirtualizer, they remove the Base64 wrapper and interpret the decoded bytes directly, but **do not** invoke backend restoration logic or use any embedded keys. Although attackers are assumed to also possess `<model>_params.bin`, the static baseline focuses on `Operator Recovery` and `Structure Recovery`; raw parameter bytes are not directly actionable for these two tasks without decrypting `v_position_data` and `v_shape`.

**One-command batch run (recommended)**:

```bash
bash security_eval/NeuralVirtualizer/run_static_all.sh

# Final summary:
#   security_eval/NeuralVirtualizer/static_summary.json
#
# Intermediate outputs:
#   security_eval/NeuralVirtualizer/opTypes/predict/static_original/
#   security_eval/NeuralVirtualizer/opTypes/predict/static_virtualized/
#   security_eval/NeuralVirtualizer/opTypes/real_original/
#   security_eval/NeuralVirtualizer/opTypes/eval/static_original/
#   security_eval/NeuralVirtualizer/opTypes/eval/static_virtualized/
#   security_eval/NeuralVirtualizer/struct/predict/static_original/
#   security_eval/NeuralVirtualizer/struct/predict/static_virtualized/
#   security_eval/NeuralVirtualizer/struct/eval/static_original/
#   security_eval/NeuralVirtualizer/struct/eval/static_virtualized/
```

**Manual run for one model** (example: model id `5`, i.e., `mobilenet`):

```bash
# 1) Original .tflite -> operator recovery
python3 security_eval/NeuralVirtualizer/opTypes/static_attack.py \
  --model_name 5 \
  --artifact original \
  --tag static_original
python3 security_eval/NeuralVirtualizer/opTypes/compare_eval.py \
  --model_name 5 \
  --LLM static_original \
  --real_subdir real_original

# 2) Original .tflite -> structure recovery
python3 security_eval/NeuralVirtualizer/struct/static_attack.py \
  --model_name 5 \
  --artifact original \
  --tag static_original
python3 security_eval/NeuralVirtualizer/struct/visualize_struct.py \
  --model_name 5 \
  --LLM static_original

# 3) Virtualized artifacts -> operator recovery
python3 security_eval/NeuralVirtualizer/opTypes/static_attack.py \
  --model_name 5 \
  --artifact virtualized \
  --tag static_virtualized
python3 security_eval/NeuralVirtualizer/opTypes/compare_eval.py \
  --model_name 5 \
  --LLM static_virtualized \
  --real_subdir real

# 4) Virtualized artifacts -> structure recovery
python3 security_eval/NeuralVirtualizer/struct/static_attack.py \
  --model_name 5 \
  --artifact virtualized \
  --tag static_virtualized
python3 security_eval/NeuralVirtualizer/struct/visualize_struct.py \
  --model_name 5 \
  --LLM static_virtualized

# 5) Aggregate all generated static results (after you run the desired models)
python3 security_eval/NeuralVirtualizer/summarize_static_results.py \
  --orig_tag static_original \
  --virt_tag static_virtualized \
  --model_ids 5
```

**Metrics reported by `static_summary.json`**:
- `Operator Recovery`: mean Top-1 exact recovery rate from `opTypes/eval/<tag>/*_eval.json`
- `Structure Recovery`: mean ground-truth similarity from `struct/eval/<tag>/*_similarity.json`
- `Static Reconstruction`: number of models whose operator recovery is `1.0` and structure recovery is `1.0`

#### Phase 3: Attack ModelObfuscator

**Test**: Structure Recovery from `obf.tflite`

```bash
# 1) Generate obfuscated TFLite files (simulating ModelObfuscator output)
python3 security_eval/ModelObfuscator/struct/obf_tflite2json.py

# 2) LLM attempts graph reconstruction (via API, see Prompt.md)
#    Input:  security_eval/ModelObfuscator/struct/obf_model/<model>_obf.json
#    Output: security_eval/ModelObfuscator/struct/predict/<LLM>/<model>_predict.json
#    Result: ⚠️ LLM successfully predicts nodes and edges

# 3) Evaluate reconstruction accuracy via ILP graph matching
python3 security_eval/ModelObfuscator/struct/visualize_struct.py --LLM claude --model_name 5
# Or batch test (10 models × 2 LLMs = 20 tasks):
bash security_eval/ModelObfuscator/struct/run_all.sh

# Expected result: ⚠️ High similarity scores (0.6-0.9)
#                  → ModelObfuscator vulnerable to structure recovery
```

**Operator Type Recovery**: Not tested (recompiled operators are inherently unrecognizable).

#### Phase 4: Attack NeuralVirtualizer

**Test 1**: Operator Type Recovery from `v_infos.json`

```bash
# LLM attempts to predict operator types from encrypted metadata
# (Same workflow as ModelObfuscator attack)
# Input:  security_eval/NeuralVirtualizer/opTypes/v_infos/<model>_v_infos.json
# Output: security_eval/NeuralVirtualizer/opTypes/predict/<LLM>/<model>_predict.json

# Evaluation:
python3 security_eval/NeuralVirtualizer/opTypes/compare_eval.py --LLM claude

# Expected result: ✅ LLM guesses common operators (CONV, RELU) based on prior knowledge
#                     but cannot infer actual types from encrypted metadata
#                  ✅ Validation: Model 0 (diverse operator distribution) shows very low accuracy
#                     → Proves AES-256 encryption prevents true operator type inference
```

**Test 2**: Structure Recovery from `v_infos.json`

```bash
# LLM attempts graph reconstruction from encrypted metadata
# (Same workflow as ModelObfuscator attack)

# Expected result: ✅ No edges predicted (LLM outputs empty "edges": [])
#                  → Full encryption prevents structural inference
```

### Graph Matching Algorithm (ILP-Based)

To quantify structural recovery success, we use **Integer Linear Programming (ILP)** via OR-Tools CP-SAT solver to match predicted graphs against ground truth:

**Formulation**:
- **Variables**: `x[p,g] ∈ {0,1}` (whether predicted node p maps to ground truth node g)
- **Constraints**:
  - One-to-one node mapping
  - Degree-based pruning (forbid mappings with large degree mismatches)
- **Objective**: Maximize `10×TP - FP` (True Positive edges heavily weighted)

**F1-Dominant Similarity** (prevents sub-model false positives and excessive coverage penalty):
```
similarity = F1 × (0.4 + 0.3 × coverage_pred + 0.3 × coverage_gt)

where:
  F1 = 2×TP / (2×TP + FP + FN)
  coverage_pred = mapped_nodes / total_pred_nodes
  coverage_gt = mapped_nodes / total_gt_nodes

  Formula ensures F1 remains dominant (minimum 40% weight) while coverage
  contributes auxiliary information (maximum 60% combined weight). This prevents
  low coverage from completely overshadowing high-quality edge matching.
```

**Multi-level Ranking** (for model identification):
1. Similarity (desc) → 2. TP count (desc) → 3. Matched nodes (desc) → 4. Model ID (asc)

### Performance Optimizations

- **Parallelization**: 10-process pool evaluates 10 candidate models concurrently (~15-20s on 24-core CPU)
- **Degree pruning**: Reduces ILP search space by 60-80%
- **Warm-start hints**: Degree-based initial mapping accelerates convergence
- **Selective FP variables**: Caps False Positive variables at 5000 to prevent combinatorial explosion

### Evaluation Metrics

**Per-model output** (`<model>_similarity.json`):
```json
{
  "query_model_id": 5,
  "llm": "claude-sonnet-4_5",
  "ranking": [
    {"rank": 1, "model_name": "mobilenet", "similarity": 0.6204, "tp": 29, "fp": 7, "fn": 1},
    {"rank": 2, "model_name": "mnasnet", "similarity": 0.5123, "tp": 24, "fp": 8, "fn": 6},
    ...
  ],
  "top1_correct": true,    // Ground truth ranked #1
  "ground_truth_rank": 1,
  "total_time": 18.3
}
```

**Success criteria**:
- ✅ **Top-1 identification**: Ground truth model ranks #1 (or tied)
- ✅ **High similarity (>0.6)**: Strong structural match
- ⚠️ **Tied Top-1**: Multiple models with identical similarity (structural ambiguity)

### Comparative Results

**ModelObfuscator** (baseline):
- ✅ Operator types: Fully protected (recompiled, unrecognizable)
- ⚠️ **Graph structure: VULNERABLE** (LLM predicts edges with high accuracy)
  - Top-1 identification: **High success rate** (similarity 0.6-0.9)
  - TP edges: 25-29 out of 30 typical edges recovered
  - **Conclusion**: Structural patterns leak through obfuscation

**NeuralVirtualizer** (this work):
- ✅ **Operator types: Protected** (low LLM prediction accuracy <20%)
- ✅ **Graph structure: FULLY PROTECTED** (LLM cannot predict edges)
  - Top-1 identification: **Fails** (no edges predicted → similarity = 0)
  - TP edges: 0 (LLM outputs `"edges": []`)
  - **Conclusion**: AES-256-CTR encryption prevents all structural inference

### Visualization

Three-panel comparison (`<model>_{top1}_vis.png`):
- **Left**: Ground truth graph (blue nodes, gray edges)
- **Middle**: Predicted graph (blue=matched nodes, orange=unmatched/dummy)
- **Right**: True Positive edges only (green edges)

**Title annotations**:
- `✓ Correct`: Model correctly identified as Top-1
- `✗ Wrong, GT: <name> @ Rank N`: Misidentification

### Directory Structure

```
security_eval/
├── model_json/                 # Ground truth graphs (10 models)
├── tflite2json.py             # Generate ground truth from .tflite
├── ModelObfuscator/           # Test baseline method security
│   └── struct/
│       ├── obf_model/         # Obfuscated TFLite artifacts
│       ├── predict/           # LLM predictions (contains edges ⚠️)
│       ├── eval/              # ILP evaluation results
│       ├── visualize_struct.py  # Graph matching evaluator
│       ├── run_all.sh         # Batch test (10 models × 2 LLMs)
│       └── Prompt.md          # Attack prompt for LLMs
└── NeuralVirtualizer/          # Test this work's security
    ├── opTypes/               # Operator type recovery attack
    │   ├── v_infos/           # Encrypted metadata inputs
    │   ├── predict/           # LLM predictions
    │   ├── compare_eval.py    # Type prediction evaluator
    │   └── Prompt.md
    └── struct/                # Structure recovery attack
        ├── v_infos/           # Encrypted metadata inputs
        ├── predict/           # LLM predictions (empty edges ✅)
        └── eval/              # Evaluation results
```

### Dependencies

Security evaluation requires additional packages:

```bash
pip install ortools networkx matplotlib
```

- **ortools**: CP-SAT solver for ILP graph matching
- **networkx**: Graph data structures and algorithms
- **matplotlib**: Visualization of predicted vs ground truth graphs

### Key Takeaways

This comparative evaluation demonstrates:

1. **ModelObfuscator limitations**: While operators are fully protected through recompilation, **structural patterns remain vulnerable** to LLM-based attacks (high edge recovery accuracy).

2. **NeuralVirtualizer superiority**: Independent AES-256-CTR encryption across 4 domains (operator types, parameters, connections, shapes) **prevents all structural inference** - LLMs cannot predict edges from encrypted metadata.

3. **Practical impact**: Adversaries with access to obfuscated files can:
   - ModelObfuscator: Reconstruct graph topology → identify model architecture → potential IP theft
   - NeuralVirtualizer: Obtain only random-looking ciphertext → no actionable intelligence

See `security_eval/ModelObfuscator/struct/run_all.sh` and `security_eval/NeuralVirtualizer/opTypes/compare_eval.py` for reproducing these results.


## Extensibility

- Android: End-to-end ARM64 builds and on-device testing are documented in the section "Android Device (arm64) Build & Testing". A sample app is not included; Bazel/NDK build and ADB-run steps are provided.
- C API: If you need a pure‑C surface, consider adding a thin wrapper (e.g., `TfLiteModelCreateFromVirtualizedFiles`).
- **File Renaming for Production**: The `<model>_` prefix in virtualized file names (`<model>_v_infos.json`, `<model>_params.bin`) is used for testing convenience to identify different models. In production environments, these files can be renamed arbitrarily (e.g., `data1.bin`, `config.json`) to completely hide model identity information without affecting functionality. The API accepts any file paths:
  ```cpp
  // Example: renamed files for obfuscation
  auto model = tflite::FlatBufferModel::BuildFromVirtualizedFiles(
      "/app/assets/a7f3d2.json",  // renamed from squeezenet_v_infos.json
      "/app/assets/9e4c1b.bin"   // renamed from squeezenet_params.bin
  );
  ```


## Security Considerations

### Threat Model

**What NeuralVirtualizer protects against:**
- ✅ Static analysis of JSON files: All sensitive metadata encrypted with AES-256-CTR; useless without keys.
- ✅ Enumeration attacks: No small search space to brute-force (2^256 key space).
- ✅ Known-plaintext attacks: CTR mode with unique nonces prevents XOR-based key recovery.
- ✅ Parameter extraction: `params.bin` mapping requires decrypted position information from JSON.
- ✅ Structure fingerprinting: Random index mapping and smart padding obscure model architecture.

**What NeuralVirtualizer does NOT fully protect against:**
- ⚠️ Reverse engineering compiled binary: Professional attackers can extract obfuscated keys and XOR parameters from `libtensorflowlite.so`, though XOR deobfuscation logic significantly increases extraction difficulty compared to plaintext keys.
- ⚠️ Dynamic debugging: Runtime memory dumps or Frida hooks can capture real keys during model loading.
- ⚠️ Hardware access: Physical device access enables memory extraction attacks.

### Security Enhancements (Optional)

For higher security requirements, consider:

1. **Advanced Code Obfuscation**: Combine with LLVM-Obfuscator for even stronger protection of XOR deobfuscation logic.
2. **White-box Cryptography**: Replace XOR obfuscation with white-box AES implementation for key management.
3. **Anti-Debugging**: Detect GDB/Frida and refuse model loading if detected.
4. **Hardware Encryption**: Store obfuscated keys in TPM or TEE/SGX for near-perfect protection.
5. **Key Derivation**: Derive keys from device-specific values to prevent cross-device reuse.

### Recommended Use Cases

**Best suited for:**
- ✅ Commercial edge deployments (IoT, mobile apps)
- ✅ Protecting proprietary model architectures
- ✅ Compliance with data protection regulations
- ✅ Preventing casual model extraction

**Consider alternatives for:**
- ⚠️ Models worth >$1M (use TEE + remote attestation)
- ⚠️ Defense against nation-state adversaries
- ⚠️ Environments where compiled binary is easily accessible


## Troubleshooting

### Decryption Errors
- **"Failed to decrypt operator type code" or similar errors**
  - Cause: Obfuscated encryption keys and XOR parameters in C++ binary don't match those used to encrypt `<model>_v_infos.json`.
  - Solution: Rebuild `libtensorflowlite.so` (step 6 in Quickstart) after running `scripts/inject_keys_to_backend.py`.
  - Prevention: Always rebuild TFLite after step 5 Phase 3 (inject keys).
- **"Failed to decrypt metadata/param/graph/shape"**
  - Causes: Version mismatch between virtualizer and C++ runtime, stale library without re-injected keys, or corrupted Base64 content.
  - Solutions: Re-run overlay (Workflow step 4), re-run key injection (Workflow step 5), then rebuild TFLite (Workflow step 6); confirm files are intact.

### Build Issues
- **"Virtualized model failed to build"**
  - Ensure `<model>_v_infos.json` and `<model>_params.bin` are present and readable.
  - Confirm `virtualized_model_*.{h,cc}` and `BUILD` files copied into TF checkout before building.
  - Verify `virtualized_model_parser.cc` contains injected obfuscated keys and XOR parameters (search for `op_encryption_key_ = {0x` and `xor_master_mask_ = {0x`).

- **"undefined reference to EVP_CIPHER_CTX_new"**
  - Cause: OpenSSL not linked.
  - Solution: Install OpenSSL development packages (`libssl-dev` on Ubuntu).

- **"jsonrepair not found" or JSON repair fails**
  - Cause: Node.js tool `jsonrepair` not installed.
  - Solution: `npm i -g jsonrepair` (already installed in Docker image).

- **flatc missing or version mismatch**
  - Cause: `flatc` not installed or incompatible with schema.
  - Solution: Install a recent `flatc` (Docker includes a compatible version).

### Runtime Issues
- **"TensorFlow Lite library not found"**
  - Confirm `libtensorflowlite.so` is built and discoverable by test harness.
  - Check `LD_LIBRARY_PATH` (Linux) or `PATH` (Windows) includes Bazel output directory.

- **"RSS=<unavailable>"**
  - Linux: `/proc/self/status` must be readable.
  - Windows: Ensure Psapi is linked (test harness already does this).

- **Output mismatch (MSE/MAE > threshold)**
  - Check you virtualized the exact `.tflite` used for the original path.
  - Verify TFLite library was rebuilt after last virtualization (key mismatch can cause silent corruption).
  - Confirm dtype and dynamic shape handling.

- **Injection failed: expected 15 markers, found fewer**
  - Cause: Overlay (Step 4) not performed or target C++ file changed; injection placeholders missing.
  - Solution: Re‑run Step 4 overlay, then run the injection script again; verify path `${TF_UPSTREAM_ROOT}/tensorflow/lite/parser/virtualized_model_parser.cc` exists with placeholders.


## License & Acknowledgements

- Project license: Apache License 2.0. See `LICENSE`.
- Third-party attributions: See `NOTICE` and `THIRD_PARTY_NOTICES.md`.
- Built on TensorFlow Lite and FlatBuffers; see upstream projects for their respective licenses.
