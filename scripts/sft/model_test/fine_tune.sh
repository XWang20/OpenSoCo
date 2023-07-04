export task
export datasets_name

export MASTER_PORT=6025

base_path=./
data_process_method='single_label'
model_type='roberta'
seeds='42 123 1768'
lrs='5e-4 1e-4 5e-5 1e-5'

models='checkpoint-392500.pt'

for dataset_name in ${datasets_name}; do {
    for model in ${models}; do {
        for seed in ${seeds}; do {
            for lr in ${lrs}; do {
                torchrun \
                --master_port=${MASTER_PORT} ./src/fine_tuning.py \
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
