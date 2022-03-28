#!/bin/bash
GPUS=`nvidia-smi -L | wc -l`


RANK=0
NODE_COUNT=1
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500
echo "rank: ${RANK}"
echo "node count: ${NODE_COUNT}"
echo "master addr: ${MASTER_ADDR}"
echo "master port: ${MASTER_PORT}"

CONFIG=$1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nnodes ${NODE_COUNT} \
        --node_rank ${RANK} \
        --master_addr ${MASTER_ADDR} \
        --master_port ${MASTER_PORT} \
        --nproc_per_node ${GPUS} \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:2}
# Any arguments from the third one are captured by ${@:2}
