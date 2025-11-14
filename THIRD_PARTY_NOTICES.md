Third-Party Notices
===================

This project includes or integrates with the following third-party
software. Each third-party project is licensed under its own terms.

- TensorFlow Lite
  - Copyright (c) Google LLC
  - License: Apache License 2.0
  - Notes: Integration via parser/builder/loader and core entry points
    requires overlaying files into a TensorFlow v2.18.1 checkout.

- FlatBuffers
  - Copyright (c) Google LLC
  - License: Apache License 2.0
  - Notes: `schema.fbs` is used with `flatc -t` during virtualization.

For full license text of this project, see `LICENSE`.
For attribution details, see `NOTICE`.

- Abseil (Abseil Authors / Google LLC)
  - License: Apache License 2.0
  - Notes: Used via TensorFlow Lite integration for status handling (`absl::Status`, `absl::StatusOr`). Pulled as Bazel external `@com_google_absl` during TFLite build.

- nlohmann/json (Niels Lohmann)
  - License: MIT License
  - Notes: Header-only JSON library used by the virtualized parser/builder (`nlohmann::json`). Pulled as Bazel external `@nlohmann_json_lib`. When distributing binaries, include the MIT license text with the artifact.

- OpenSSL
  - Copyright (c) The OpenSSL Project Authors
  - License: Apache License 2.0 (OpenSSL 3.x)
  - Notes: Used by the C++ parser for AES-256-CTR decryption via EVP. Most modern Linux distributions link OpenSSL 3.x under Apache-2.0.

- cryptography (PyCA)
  - License: Apache License 2.0
  - Notes: Python library used by the virtualizer to perform AES-256-CTR encryption for metadata.

- NumPy
  - License: BSD 3-Clause License
  - Notes: Used in the Python virtualizer for parameter tensor serialization and numeric processing.

Build-time tools (used in Docker or development workflows)
---------------------------------------------------------

- Bazelisk
  - License: Apache License 2.0
  - Notes: Used to manage Bazel versions compatible with TensorFlow.

- LLVM/Clang
  - License: Apache License 2.0 with LLVM Exceptions
  - Notes: Used as an alternative toolchain in some build environments.

- Node.js
  - License: MIT License
  - Notes: Installed in the Docker image for auxiliary tooling.

- jsonrepair (npm package)
  - License: See upstream project
  - Notes: Installed in the Docker image for JSON repair utilities during development.

- flatc (FlatBuffers compiler)
  - License: Apache License 2.0 (part of FlatBuffers)
  - Notes: Used to produce JSON via `flatc -t` as part of the virtualization pipeline.
