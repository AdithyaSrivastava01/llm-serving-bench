#!/usr/bin/env bash
set -euo pipefail

MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "turboderp/Llama-3.1-1B-Instruct"
)

CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"

echo "Downloading models to $CACHE_DIR"
echo "================================"

for model in "${MODELS[@]}"; do
    echo ""
    echo "Downloading: $model"
    huggingface-cli download "$model" --quiet || {
        echo "WARNING: Failed to download $model. You may need to accept the license on HuggingFace."
        echo "Visit: https://huggingface.co/$model"
    }
done

echo ""
echo "Done. Models cached at: $CACHE_DIR"
