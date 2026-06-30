#!/usr/bin/env bash
set -euo pipefail

echo "=== Checking VLLM_API_KEY ==="
echo "VLLM_API_KEY=${VLLM_API_KEY:-not set}"

echo ""
echo "=== Checking RunPod token ==="
if [ -f /etc/runpod/token ]; then
    echo "Found: $(cat /etc/runpod/token)"
else
    echo "No /etc/runpod/token"
fi

echo ""
echo "=== Testing without auth ==="
curl -s -w "\nHTTP_CODE:%{http_code}\n" http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-8B","prompt":"Hello","max_tokens":5}' || echo "FAILED"

echo ""
echo "=== Testing with VLLM_API_KEY ==="
curl -s -w "\nHTTP_CODE:%{http_code}\n" http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${VLLM_API_KEY:-none}" \
  -d '{"model":"Qwen/Qwen3-8B","prompt":"Hello","max_tokens":5}' || echo "FAILED"

echo ""
echo "=== Testing health ==="
curl -s -w "\nHTTP_CODE:%{http_code}\n" http://localhost:8000/health || echo "FAILED"

echo ""
echo "=== Testing models list ==="
curl -s http://localhost:8000/v1/models || echo "FAILED"

echo ""
echo "=== Checking env for keys ==="
env | grep -i -E "key|token|auth|secret" || echo "No matching env vars"
