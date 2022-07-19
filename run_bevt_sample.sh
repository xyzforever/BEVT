#!/bin/bash
GPUS=${1:-8}
BS=${2:-8}
EP=${3:-1}
USE_FULL_DATA=${4:-false}

export BEVT_EARLY_STOP=${5:-true}
export PYTHONPATH=$(pwd):${PYTHONPATH}
export OMP_NUM_THREADS=1

NODE_COUNT=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500

if [ ${USE_FULL_DATA} = true ] ; then
    prefix=/rocmbench/data/bevt/kinetics/kinetics400_256
    image_data_root=/rocmbench/data/bevt/ILSVRC2012/train
    image_ann_file_train=ILSVRC2012_name_train_list.txt
    repeat_times=5
else
    prefix=${HOME}/bevt/mini_data/kinetics/kinetics400_256
    image_data_root=${HOME}/bevt/mini_data/ILSVRC2012/train
    image_ann_file_train=reduced.ILSVRC2012_name_train_list.txt
    repeat_times=5000

    LOCAL_DATA_DIR=${HOME}/bevt
    if [ ! -d ${LOCAL_DATA_DIR} ] ; then
        mkdir ${LOCAL_DATA_DIR}
        cp /rocmbench/data/bevt/mini_data ${LOCAL_DATA_DIR} -r
    fi
fi

data_root=${prefix}/train_256
data_root_val=${prefix}/val_256
ann_file_train=${prefix}/train_256.txt
ann_file_val=${prefix}/val_256.txt
ann_file_test=${prefix}/val_256.txt

tokenizer_path=/rocmbench/data/bevt/dall_e_tokenizer_weight

# config dict will update according to the specified_configs.
specified_configs="tokenizer_path=${tokenizer_path} model.cls_head.vae_weight_path=${tokenizer_path}
                   model.backbone.pretrained=/rocmbench/data/bevt/swin_base_image_stream_pretrain.pth
                   data_root=${data_root} data_root_val=${data_root_val}
                   ann_file_train=${ann_file_train} ann_file_val=${ann_file_val} ann_file_test=${ann_file_test}
                   image_data_root=${image_data_root} image_ann_file_train=${image_ann_file_train}
                   data.videos_per_gpu=${BS} data.omni_videos_per_gpu='['${BS},$[8 * ${BS}]']'
                   data.train.0.ann_file=${ann_file_train} data.train.0.data_prefix=${data_root}
                   data.train.1.times=${repeat_times} data.train.1.dataset.ann_file=${image_ann_file_train}
                   data.train.1.dataset.data_prefix=${image_data_root} total_epochs=1"

# echo $specified_configs

echo "rank: ${NODE_RANK}"
echo "node count: ${NODE_COUNT}"
echo "master addr: ${MASTER_ADDR}"
echo "master port: ${MASTER_PORT}"

python -m torch.distributed.launch --nnodes ${NODE_COUNT} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    --nproc_per_node ${GPUS} \
    tools/train.py configs/recognition/swin/swin_base_patch244_window877_bevt_in1k_k400.py \
        --launcher pytorch --work-dir OUTPUT/swin_base_bevt_twostream \
        --cfg-options ${specified_configs} \
        --seed 0 --deterministic
