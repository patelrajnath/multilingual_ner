#!/bin/bash

python decode_ner_bert.py \
    --data_dir data/conll2003-de \
    --test deuutf.testa.dev.csv \
    --idx2labels data/conll2003-de/idx2labels.txt \
    --model outputs/deuutf_best_model_bert.pt

#     --data_dir data/conll2003 \
#     --test eng.testa.dev.csv \
#     --idx2labels data/conll2003-de/idx2labels.txt \
#     --model outputs/deuutf_best_model_bert.pt

#     --data_dir data/conll2003-de \
#     --test deuutf.testa.dev.csv \
#     --idx2labels data/conll2003-de/idx2labels.txt \
#     --model outputs/deuutf_best_model_bert.pt

#     --data_dir data/conll2003-de \
#     --train deuutf.train.train.csv \
#     --test deuutf.testa.dev.csv \
