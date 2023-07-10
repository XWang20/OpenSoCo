pip install model_center
pip install OpenCC
pip install scikit-learn

export data_process_method='single_label'

export task='rumor'
export datasets_name='mrf'
bash scripts/sft/model_test/fine_tune.sh

export task='irony'
export datasets_name='Sarcasm_Headlines'
bash scripts/sft/model_test/fine_tune.sh

export datasets_name='semeval2018task3'
bash scripts/sft/model_test/fine_tune.sh

export task='humor'
export datasets_name='hahackathon'
bash scripts/sft/model_test/fine_tune.sh