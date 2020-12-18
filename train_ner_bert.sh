#!/bin/bash

python train_ner_bert.py \
    --data_dir data/conll2003 \
    --train eng.train.train.csv \
    --test eng.testa.dev.csv \
    --embedding 768 \
    --hidden_layer_size 1024 \
    --num_hidden_layers 2 \
    --max_steps 15000 \
    --epochs 30
#     --data_dir data/conll2003-de \
#     --train deuutf.train.train.csv \
#     --test deuutf.testa.dev.csv \
#     --embedding 768 \
#     --hidden_layer_size 1024 \
#     --max_steps 1500 \
#     --model_name bert-base-multilingual-cased
