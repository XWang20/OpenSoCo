pip install model_center
pip install OpenCC
pip install scikit-learn

export data_process_method='single_label'

export task='categorization'
export datasets_name='BBC_News'
bash scripts/sft/model_test/fine_tune.sh
export datasets_name='tweetopic'
bash scripts/sft/model_test/fine_tune.sh

export task='bias'
export datasets_name='BuzzFeed-Webis'
bash scripts/sft/model_test/fine_tune.sh

export task='emotion'
export datasets_name='emoint'
bash scripts/sft/model_test/fine_tune.sh

export data_process_method='stance'

export task='sentiment'
export datasets_name='NewsMTSC'
bash scripts/sft/model_test/fine_tune.sh

export task='stance'
export datasets_name='semeval2016task6a'
bash scripts/sft/model_test/fine_tune.sh