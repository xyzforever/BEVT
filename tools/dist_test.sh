#!/bin/bash
GPUS=`nvidia-smi -L | wc -l`

RANK=0
NODE_COUNT=1
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500

CONFIG=$1
CHECKPOINT=$2
#GPUS=$3
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nnodes ${NODE_COUNT} \
        --node_rank ${RANK} \
        --master_addr ${MASTER_ADDR} \
        --master_port ${MASTER_PORT} \
        --nproc_per_node ${GPUS} \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:3}
# Arguments starting from the forth one are captured by ${@:4}

