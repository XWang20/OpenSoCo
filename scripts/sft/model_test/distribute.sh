pip install model_center
pip install OpenCC
pip install scikit-learn

export task='sentiment'
export datasets_name='twitter_US_airline_dataset'

bash fine_tune.sh
