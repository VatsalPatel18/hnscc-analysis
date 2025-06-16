#!/bin/bash

# Script to download the HNSCC dataset from Hugging Face
# Usage: ./download_dataset.sh [TARGET_DIR]
# The dataset will be cloned into TARGET_DIR or into
# Case-Study-HNSCC-ML-Copilot-Agent if TARGET_DIR is not specified.

set -e

REPO_URL="https://huggingface.co/datasets/VatsalPatel18/Case-Study-HNSCC-ML-Copilot-Agent"
TARGET_DIR="${1:-Case-Study-HNSCC-ML-Copilot-Agent}"

if [ -d "$TARGET_DIR" ]; then
    echo "Target directory $TARGET_DIR already exists. Skipping clone."
else
    git clone "$REPO_URL" "$TARGET_DIR"
fi

