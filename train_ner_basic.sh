#!/bin/bash

python train_ner_basic.py \
--data_dir data/snips \
--train_text snips_train_text.txt \
--train_label snips_train_labels.txt \
--test_text snips_test_text.txt \
--test_label snips_test_labels.txt \
--vocab words.txt --tag_set tags.txt \
--batch_size 32 \
--max_steps 15000 \
--hidden_layer_size 1024 \
--embedding 768 \
--num_hidden_layers 1
# --data_dir data/nlu \
# --train_text nlu_train_text.txt \
# --train_label nlu_train_labels.txt \
# --test_text nlu_test_text.txt \
# --test_label nlu_test_labels.txt \
# --vocab words.txt --tag_set tags.txt \
# --batch_size 32 \
# --max_steps 15000 \
# --hidden_layer_size 1024 \
# --embedding 768 \
# --num_hidden_layers 2
# --data_dir data/snips \
# --train_text snips_train_text.txt \
# --train_label snips_train_labels.txt \
# --test_text snips_test_text.txt \
# --test_label snips_test_labels.txt \
# --vocab words.txt --tag_set tags.txt \
# --batch_size 32 \
# --max_steps 15000 \
# --hidden_layer_size 1024 \
# --embedding 768 \
# --num_hidden_layers 2
# --data_dir data/ubuntu \
# --train_text ubuntu_train_text.txt \
# --train_label ubuntu_train_labels.txt \
# --test_text ubuntu_test_text.txt \
# --test_label ubuntu_test_labels.txt \
# --vocab words.txt --tag_set tags.txt \
# --batch_size 8
