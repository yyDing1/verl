#!/bin/bash

# Configuration
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-8B"}
BASE_PORT=31000          # First worker port (will increment for each worker)
NUM_WORKERS=4            # Number of workers (should match your GPU count)
ROUTER_PORT=30000        # Router port

# Launch workers
declare -a WORKER_URLS
for ((i=0; i<NUM_WORKERS; i++)); do
    PORT=$((BASE_PORT + i))
    WORKER_URLS+=("http://[::1]:$PORT")

    echo "Launching worker $i on GPU $i at port $PORT"
    CUDA_VISIBLE_DEVICES=$i python -m sglang.launch_server \
        --model-path $MODEL_PATH --host ::1 --port $PORT > /dev/null 2>&1 &
done

# Launch router
echo "Launching router at port $ROUTER_PORT with worker URLs:"
printf '  %s\n' "${WORKER_URLS[@]}"
python -m sglang_router.launch_router --host :: --port $ROUTER_PORT --worker-urls "${WORKER_URLS[@]}"

# Cleanup on exit
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT