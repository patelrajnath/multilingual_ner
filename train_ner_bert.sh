#!/bin/bash

python train_ner_bert.py \
    --arch bert_ner \
    --batch_size 32 \
    --shuffle \
    --cache_features \
    --save_cache_features \
    --data_dir data/conll2003 \
    --train eng.train.train.csv \
    --test eng.testa.dev.csv \
    --onnx \
    
#     --use_projection
#     --cpu \
#     --cache_features \
#     --max_seq_len 50 \
#     --data_dir data/alliance \
#     --train alliance_train_text.txt.csv \
#     --test alliance_test_text.txt.csv \
    
#     --model_type electra \
#     --model_name google/electra-base-discriminator \
#     --embedding_dim 1024 \
#     --only_embedding

#     --model_type electra \
#     --model_name google/electra-base-discriminator \
#     --model_type distilbert \
#     --model_name bert-base-multilingual-cased \
#     --model_type distilbert \
#     --model_name distilbert-base-multilingual-cased \
#     --model_type roberta \
#     --model_name sentence-transformers/roberta-base-nli-stsb-mean-tokens \
#     --model_type roberta \
#     --model_name roberta-base \
#     --model_type xlmroberta \
#     --model_name deepset/xlm-roberta-large-squad2

#     --data_dir data/alliance \
#     --train alliance_train_text.txt.csv \
#     --test alliance_test_text.txt.csv \
#     --arch attn_bert_ner \
#     --cpu

#     --data_dir data/accounts \
#     --train accounts_train_text.txt.csv \
#     --test accounts_test_text.txt.csv \
#     --arch attn_bert_ner \
#     --cpu

#     --data_dir data/wallet \
#     --train wallet_train_text.txt.csv \
#     --test wallet_test_text.txt.csv \
#     --arch attn_bert_ner \
#     --cpu

#     --data_dir data/snips \
#     --train snips_train_text.txt.csv \
#     --test snips_test_text.txt.csv \
#     --arch attn_bert_ner \
#     --cpu

#     --data_dir data/nlu \
#     --train nlu_train_text.txt.csv \
#     --test nlu_test_text.txt.csv \
#     --arch attn_bert_ner \
#     --cpu

#     --data_dir data/accounts \
#     --train accounts_train_text.txt.csv \
#     --test accounts_test_text.txt.csv \
#     --arch bert_ner \
#     --cpu
    
#     --data_dir data/nlu \
#     --train nlu_train_text.txt.csv \
#     --test nlu_test_text.txt.csv \
#     --arch bert_ner \
#     --cpu

#     --data_dir data/conll2003 \
#     --train eng.train.train.csv \
#     --test eng.testa.dev.csv \
#     --arch bert_ner \
#     --cpu

#     --data_dir data/conll2003-de \
#     --train deuutf.train.train.csv \
#     --test deuutf.testa.dev.csv \
#     --arch bert_ner \
#     --cpu

