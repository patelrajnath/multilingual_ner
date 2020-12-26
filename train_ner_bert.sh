#!/bin/bash

python train_ner_bert.py \
    --data_dir data/accounts \
    --train accounts_train_text.txt.csv \
    --test accounts_test_text.txt.csv \
    --arch bert_ner_small \
    --cpu
    
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
#     --embedding 768 \
#     --hidden_layer_size 1024 \
#     --num_hidden_layers 1 \
#     --max_steps 15000 \
#     --epochs 30

#     --data_dir data/conll2003-de \
#     --train deuutf.train.train.csv \
#     --test deuutf.testa.dev.csv \
#     --embedding 768 \
#     --hidden_layer_size 1024 \
#     --max_steps 1500 \
#     --model_name bert-base-multilingual-cased
