pip install model_center
pip install OpenCC
pip install scikit-learn

export data_process_method='single_label'
export task='sentiment'
export datasets_name='twitter_US_airline_dataset'
bash scripts/sft/model_test/fine_tune.sh

export task='offensive'
export datasets_name='OLID'
bash scripts/sft/model_test/fine_tune.sh

export datasets_name='reddit_incivility'
bash scripts/sft/model_test/fine_tune.sh