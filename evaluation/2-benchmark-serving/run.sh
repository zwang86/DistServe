#!/bin/bash

# Usage: ./distserve_eval.sh --model {MODEL} --gpu-memory-util {GPU_UTIL} --dataset {DATASET}


PORT=8000
BASE_TTFT=0.25
BASE_TPOT=0.1
CONTEXT_TP=1
CONTEXT_PP=1
DECODING_TP=1
DECODING_PP=1

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --gpu-memory-util) GPU_MEMORY_UTIL="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --base-ttft) BASE_TTFT="--base-ttft $2"; shift ;;
        --base-tpot) BASE_TPOT="--base-tpot $2"; shift ;;
        --port) PORT="$2"; shift ;;
        --context-tp) CONTEXT_TP="$2"; shift ;;
        --context-pp) CONTEXT_PP="$2"; shift ;;
        --decoding-tp) DECODING_TP="$2"; shift ;;
        --decoding-pp) DECODING_PP="$2"; shift ;;
        *) echo "Unknown parameter: $1" && exit 1 ;;
    esac
    shift
done

if [ -z "$MODEL" ] || [ -z "$GPU_MEMORY_UTIL" ] || [ -z "$DATASET" ]; then
    echo "Error: Missing arguments. You must provide --model, --gpu-memory-util, and --dataset."
    exit 1
fi

# Start the distserve server in the background
echo "Activating conda environment and starting server..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate distserve

echo "Starting server with model: $MODEL and GPU memory utilization: $GPU_MEMORY_UTIL"
python -m distserve.api_server.distserve_api_server \
    --host localhost \
    --port "$PORT" \
    --model "$MODEL" \
    --tokenizer "$MODEL" \
    \
    --context-tensor-parallel-size $CONTEXT_TP \
    --context-pipeline-parallel-size $CONTEXT_PP \
    --decoding-tensor-parallel-size $DECODING_TP \
    --decoding-pipeline-parallel-size $DECODING_PP \
    \
    --block-size 16 \
    --max-num-blocks-per-req 128 \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --swap-space 16 \
    \
    --context-sched-policy fcfs \
    --context-max-batch-size 128 \
    --context-max-tokens-per-batch 8192 \
    \
    --decoding-sched-policy fcfs \
    --decoding-max-batch-size 1024 \
    --decoding-max-tokens-per-batch 65536 \
    > server.log 2>&1 &

SERVER_PID=$!

echo "Server started with PID $SERVER_PID"

cleanup() {
    if ps -p $SERVER_PID > /dev/null; then
        echo "Terminating server with PID $SERVER_PID"
        kill $SERVER_PID
        wait $SERVER_PID 2>/dev/null
        echo "Server terminated."
    fi
}

trap cleanup EXIT

check_server_health() {
    curl -s "http://localhost:$PORT" > /dev/null
    return $?
}

echo "Waiting for the server to initialize..."
for i in {1..30}; do
    if check_server_health; then
        echo "Server is healthy!"
        break
    fi
    sleep 1
done

if ! check_server_health; then
    echo "Error: Server did not become healthy after 30 seconds. Exiting."
    exit 1
fi

echo "Server is up. Running benchmark with dataset: $DATASET"
python ./benchmark-serving.py --dataset "$DATASET" --verbose true --base-ttft $BASE_TTFT --base-tpot $BASE_TPOT
