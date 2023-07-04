pip install model_center
pip install OpenCC
pip install scikit-learn

export task='sentiment'
export datasets_name='twitter_US_airline_dataset'

bash scripts/sft/model_test/fine_tune.sh
