pip install model_center
pip install OpenCC
pip install scikit-learn

# export data_process_method='single_label'
# export task='sentiment'
# export datasets_name='twitter_US_airline_dataset'
# bash scripts/sft/model_test/fine_tune.sh

# export task='offensive'
# export datasets_name='OLID'
# bash scripts/sft/model_test/fine_tune.sh

# export datasets_name='reddit_incivility'
# bash scripts/sft/model_test/fine_tune.sh

# export task='rumor'
# export datasets_name='mrf'
# bash scripts/sft/model_test/fine_tune.sh

# export task='irony'
# export datasets_name='Sarcasm_Headlines'
# bash scripts/sft/model_test/fine_tune.sh

# export datasets_name='semeval2018task3'
# bash scripts/sft/model_test/fine_tune.sh

# export task='humor'
# export datasets_name='hahackathon'
# bash scripts/sft/model_test/fine_tune.sh

# export task='categorization'
# export datasets_name='BBC_News'
# bash scripts/sft/model_test/fine_tune.sh
# export datasets_name='tweetopic'
# bash scripts/sft/model_test/fine_tune.sh

# export task='bias'
# export datasets_name='BuzzFeed-Webis'
# bash scripts/sft/model_test/fine_tune.sh

# export task='emotion'
# export datasets_name='emoint'
# bash scripts/sft/model_test/fine_tune.sh

export data_process_method='stance'

export task='sentiment'
export datasets_name='NewsMTSC'
bash scripts/sft/model_test/fine_tune.sh

export task='stance'
export datasets_name='semeval2016task6a'
bash scripts/sft/model_test/fine_tune.sh