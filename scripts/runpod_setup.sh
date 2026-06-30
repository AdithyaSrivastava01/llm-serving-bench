#!/usr/bin/env bash
# RunPod deployment script for full benchmark run
# Template: RunPod PyTorch 2.4 (NOT vLLM template)
# GPU: 2x A100 SXM 80GB
# Disk: 200GB container + 200GB volume
# Estimated runtime: 3-4 hours
set -euo pipefail

echo "=== 1. System packages ==="
apt-get update && apt-get install -y git

echo "=== 2. Clone repo ==="
cd /workspace
git clone https://github.com/AdithyaSrivastava01/llm-serving-bench.git
cd llm-serving-bench

echo "=== 3. Install serving engines ==="
pip install vllm "sglang[all]"

echo "=== 4. Install benchmark deps ==="
pip install typer pydantic pyyaml aiohttp pandas pyarrow matplotlib seaborn pynvml numpy scipy

echo "=== 5. Download model ==="
export HF_HOME=/workspace/.cache/huggingface
huggingface-cli download Qwen/Qwen3-8B

echo "=== 6. Verify GPU access ==="
python3 -c "import torch; print(f'GPUs: {torch.cuda.device_count()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Run the full benchmark:"
echo "  cd /workspace/llm-serving-bench"
echo "  export PYTHONPATH=\$(pwd)/src"
echo "  export HF_HOME=/workspace/.cache/huggingface"
echo "  python3 -c \"from llm_bench.cli import app; app()\" run --mode process --num-gpus 2 --config configs/full_matrix.yaml"
echo ""
echo "After completion:"
echo "  python3 -c \"from llm_bench.cli import app; app()\" analyze"
echo "  python3 -c \"from llm_bench.cli import app; app()\" report"
echo ""
echo "Push results:"
echo "  git config user.email 'adithya@example.com'"
echo "  git config user.name 'Adithya Srivastava'"
echo "  git add -f results/"
echo "  git commit -m 'data: full benchmark results vLLM vs SGLang on 2x A100'"
echo "  git push origin master"
