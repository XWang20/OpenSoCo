#! /bin/bash

export HADOOP_USER_NAME=tc_agi
export HADOOP_USER_PASSWORD=IH2U3AS1D

pip install model_center==0.1.3
ls /mnt/data/user/tc_agi/user/wangxing

GPUS_PER_NODE=8

if [ ${IDC} == klara-2-pek02 ]; then
    DISTRIBUTED_ARGS="--nnodes=${WORLD_SIZE} \
                    --nproc_per_node=${GPUS_PER_NODE} \
                    --node_rank=${RANK} \
                    --master_addr=${MASTER_ENDPOINT} \
                    --master_port=${MASTER_PORT}"
else
    DISTRIBUTED_ARGS="--nnodes=${WORLD_SIZE} \
                    --nproc_per_node=${GPUS_PER_NODE} \
                    --node_rank=${RANK} \
                    --rdzv_id=1 \
                    --rdzv_backend=c10d \
                    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"
fi

BASE_PATH="."
DATA_PATH="/mnt/data/user/tc_agi/user/wangxing"
SAVE_PATH="/data/"
HDFS_PATH="/user/tc_agi/user/wangxing/save"
DATASET_NAME="OpenSoCo_en"
TEST_DATASET="OpenSoCo_en"
CONFIG="deberta_prenorm"

OPTS=""
OPTS+=" --vocab-file ${BASE_PATH}/config/${CONFIG}.json"
OPTS+=" --model-config ${BASE_PATH}/config/${CONFIG}.json"
OPTS+=" --input-dataset ${DATA_PATH}/${DATASET_NAME}/"
OPTS+=" --test-dataset ${DATA_PATH}/valid/${TEST_DATASET}/"
OPTS+=" --load-path ${DATA_PATH}/save/${CONFIG}_${DATASET_NAME}/1e-4-init-embed"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --hdfs-save ${HDFS_PATH}/${CONFIG}_${DATASET_NAME}/1e-4-init-embed"

OPTS+=" --load init_checkpoint/deberta-bmtrain.pt"
OPTS+=" --warmup-iters 10000"
OPTS+=" --lr-decay-style linear"
OPTS+=" --lr-decay-iters 1000000"
OPTS+=" --weight-decay 0.01"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 524288"
OPTS+=" --start-step 0"
OPTS+=" --batch-size $((128 / ${WORLD_SIZE}))"
OPTS+=" --lr 1e-4"
OPTS+=" --save-iters 500"
OPTS+=" --log-iters 10"
OPTS+=" --gradient-accumulate 2"
OPTS+=" --train-iters 1000000"
OPTS+=" --report_to tensorboard"


CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/train.py ${OPTS}"
echo ${CMD}

mkdir -p ${SAVE_PATH}/${CONFIG}_${DATASET_NAME}

if [[ $NODE_RANK == 0 ]]&&[[ $DLS_TASK_NUMBER == 1 ]]; then
    ${CMD} 2>&1 | tee ${SAVE_PATH}/${CONFIG}_${DATASET_NAME}/1e-4-init-embed/logs_$(date +"%Y_%m_%d_%H_%M_%S").log
else
    ${CMD}
fi
