#!/bin/sh
#conda activate kbqa-torch1.7
python code/KBQA_Runner.py  \
        --train_folder  data/train_WBQ \
        --dev_folder data/dev_WBQ \
        --test_folder data/test_WBQ \
        --vocab_file data/WBQ/vocab.txt \
        --KB_file data/WBQ/kb_cache.json \
        --M2N_file data/WBQ/m2n_cache.json \
        --QUERY_file data/WBQ/query_cache.json \
        --output_dir trained_model/WBQ \
        --config config/bert_config.json \
        --load_model trained_model/WBQ/new \
        --save_model author_retrain_batch50 \
        --max_hop_num 2 \
        --num_train_epochs 100 \
	--gpu_id 1\
        --do_train 1\
        --do_eval 1\
        --do_policy_gradient 1\
        --train_limit_number 50\
        --learning_rate 5e-6 \
