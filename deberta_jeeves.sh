#! /bin/bash

pip install model_center==0.1.3

# if [[ ${IDC} == luca-dev-pek02 ]]; then
#     MASTER_ADDR=localhost
#     MASTER_PORT=12423
#     NNODES=2
#     NODE_RANK=0
# else
#     MASTER_HOST="${NODE_NAME}"
#     MASTER_ADDR="${MASTER_ADDR}"
#     MASTER_PORT="${MASTER_PORT}"
#     NNODES=2
#     NODE_RANK="${RANK}"
# fi

MASTER_ADDR=localhost
MASTER_PORT=12423
NNODES=1
NODE_RANK=0

GPUS_PER_NODE=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH="."
DATA_PATH="/mnt/data/user/tc_agi/user/wangxing"
SAVE_PATH="/data/save"
HDFS_PATH="/user/tc_agi/user/wangxing/save"
DATASET_NAME="OpenSoCo_en"
TEST_DATASET="OpenSoCo_en"
CONFIG="deberta_prenorm"

OPTS=""
OPTS+=" --vocab-file ${BASE_PATH}/config/${CONFIG}.json"
OPTS+=" --model-config ${BASE_PATH}/config/${CONFIG}.json"
OPTS+=" --input-dataset ${DATA_PATH}/${DATASET_NAME}/"
OPTS+=" --test-dataset ${DATA_PATH}/${TEST_DATASET}/"
OPTS+=" --save ${SAVE_PATH}/${CONFIG}_${DATASET_NAME}/1e-4-init-embed"
OPTS+=" --hdfs-save ${HDFS_PATH}/${CONFIG}_${DATASET_NAME}/1e-4-init-embed"

OPTS+=" --load init_checkpoint/deberta-bmtrain.pt"
OPTS+=" --warmup-iters 10000"
OPTS+=" --lr-decay-style linear"
OPTS+=" --lr-decay-iters 1000000"
OPTS+=" --weight-decay 0.01"
OPTS+=" --clip-grad 1"
OPTS+=" --loss-scale 524288"
OPTS+=" --start-step 0"
OPTS+=" --batch-size 64"
OPTS+=" --lr 1e-4"
OPTS+=" --save-iters 2500"
OPTS+=" --log-iters 50"
OPTS+=" --gradient-accumulate 1"
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
