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
    --trust-remote-code

# Check if the Python command was successful
if [ $? -eq 0 ]; then
    echo "vLLM server started successfully on http://localhost:8002"
else
    echo "Failed to start vLLM server"
    exit 1
fi