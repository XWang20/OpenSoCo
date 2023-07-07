export task
export datasets_name
export data_process_method

GPUS_PER_NODE=8

DISTRIBUTED_ARGS="--nnodes=${WORLD_SIZE} \
                --nproc_per_node=${GPUS_PER_NODE} \
                --node_rank=${RANK} \
                --master_addr=${MASTER_ENDPOINT} \
                --master_port=${MASTER_PORT}"

base_path=./
model_type='roberta'
seeds='42 123 1768'
lrs='5e-4 1e-4 5e-5 1e-5'

models='checkpoint-392500.pt'

for dataset_name in ${datasets_name}; do {
    for model in ${models}; do {
        for seed in ${seeds}; do {
            for lr in ${lrs}; do {
                python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ./src/fine_tuning.py \
                --model_name ${model} \
                --eval_strategy step \
                --model_type ${model_type} \
                --metric_for_best_model accuracy \
                --data_process_method ${data_process_method} \
                --dataset_path ./datasets/${task}/${dataset_name} \
                --dataset_name  ${dataset_name} \
                --learning_rate ${lr} \
                --output_dir /data/results/${task}/${model}/${dataset_name}/seed_${seed}/${lr} \
                --seed ${seed}
            }
            done
        }
        done
    }
    done
}
done
