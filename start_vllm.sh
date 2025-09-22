#!/bin/bash

# Start vLLM server with proper configuration for the RAG API
echo "Starting vLLM server..."

# Basic command
python -m vllm.entrypoints.openai.api_server \
    --model neuralmagic/pixtral-12b-quantized.w4a16 \
    --host 0.0.0.0 \
    --port 8002 \
    --served-model-name neuralmagic/pixtral-12b-quantized.w4a16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --enable-log-requests

echo "vLLM server started on http://localhost:8002"
echo "Test the server with: python test_vllm.py"
