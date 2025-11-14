# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 The NeuralVirtualizer Authors
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install build essentials, Python 3.12, and other tooling required for TensorFlow Lite builds.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    curl \
    git \
    openjdk-11-jdk \
    pkg-config \
    cmake \
    lsb-release \
    wget \
    software-properties-common \
    gnupg \
    dos2unix \
    unzip \
    zip \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    rm -rf /var/lib/apt/lists/*

# Install Bazelisk (manages Bazel versions to match TensorFlow requirements).
RUN curl -L "https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64" \
    -o /usr/local/bin/bazel && \
    chmod +x /usr/local/bin/bazel

# Install flatc
RUN curl -L "https://github.com/google/flatbuffers/releases/download/v25.2.10/Linux.flatc.binary.clang%2B%2B-18.zip" \
    -o /tmp/flatc.zip && \
    unzip -o /tmp/flatc.zip -d /usr/local/bin && \
    chmod +x /usr/local/bin/flatc && \
    rm /tmp/flatc.zip

# Install Node.js 22 and jsonrepair
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    npm install -g jsonrepair && \
    rm -rf /var/lib/apt/lists/*

# Set up workspace directory and copy the current project tree (including tensorflow_src).
WORKDIR /NeuralVirtualizer
COPY . /NeuralVirtualizer

# Normalize line endings for Bazel-related scripts copied from Windows hosts.
RUN find /NeuralVirtualizer -type f \( -name '*.sh' -o -name '*.bzl' -o -name 'BUILD' -o -name 'BUILD.bazel' \) -print0 \
    | xargs -0 dos2unix

# Install Python dependencies
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Install Clang-20
RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 20 && \
    update-alternatives --install /usr/bin/clang clang /usr/lib/llvm-20/bin/clang-20 100

ENV TF_SRC_ROOT=/NeuralVirtualizer/tensorflow_src

CMD ["/bin/bash"]

