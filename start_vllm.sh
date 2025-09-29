#!/bin/bash

# Start vLLM server with proper configuration for the RAG API
echo "Starting vLLM server..."

# Start vLLM server with nohup to run in background
nohup python -m vllm.entrypoints.openai.api_server \
    --model neuralmagic/pixtral-12b-quantized.w4a16 \
    --host 0.0.0.0 \
    --port 8002 \
    --served-model-name neuralmagic/pixtral-12b-quantized.w4a16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    > vllm_server.log 2>&1 &

# Get the process ID
VLLM_PID=$!

# Save PID to file for easy management
echo $VLLM_PID > vllm_server.pid

echo "vLLM server started in background with PID: $VLLM_PID"
echo "Output is being logged to: vllm_server.log"
echo "Process ID saved to: vllm_server.pid"
echo "Server should be available at: http://localhost:8002"
echo ""
echo "To stop the server, run: kill $VLLM_PID"
echo "Or use: kill \$(cat vllm_server.pid)"