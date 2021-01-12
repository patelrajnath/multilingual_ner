#!/bin/bash

python decode_ner_bert.py \
    --data_dir data/conll2003-de \
    --test deuutf.testb.dev.csv \
    --idx2labels data/conll2003/idx2labels.txt \
    --model outputs/eng_best_model_bert.pt \
    --cpu

#     --data_dir data/conll2002-es \
#     --test esp.testb.dev.csv \
#     --idx2labels data/conll2003/idx2labels.txt \
#     --model outputs/eng_best_model_bert.pt \
    
#     --data_dir data/conll2002-nl \
#     --test ned.testb.dev.csv \
#     --idx2labels data/conll2003/idx2labels.txt \
#     --model outputs/eng_best_model_bert.pt \

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
