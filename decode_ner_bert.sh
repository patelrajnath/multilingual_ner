#!/bin/bash

python decode_ner_bert.py \
    --data_dir data/conll2003 \
    --train eng.train.train.csv \
    --dev eng.testa.dev.csv \
    --test data/conll2003-de/deuutf.testa.dev.csv \
    --embedding 768 \
    --hidden_layer_size 1024 \
#     --data_dir data/conll2003-de \
#     --train deuutf.train.train.csv \
#     --test deuutf.testa.dev.csv \
#     --embedding 768 \
#     --hidden_layer_size 1024 \
