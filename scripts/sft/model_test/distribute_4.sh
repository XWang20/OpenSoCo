pip install model_center
pip install OpenCC
pip install scikit-learn

export data_process_method='single_label'

export task='irony'
export datasets_name='Sarcasm_Headlines'
bash scripts/sft/model_test/fine_tune.sh

export data_process_method='stance'

export task='stance'
export datasets_name='semeval2016task6a'
bash scripts/sft/model_test/fine_tune.sh