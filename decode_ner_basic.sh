#!/bin/bash

python decode_ner_basic.py \
--data_dir data/nlu \
--test_text nlu_test_text.txt \
--test_label nlu_test_labels.txt \
--vocab words.txt --tag_set tags.txt \
--hidden_layer_size 1024 \
--embedding 768 \
--num_hidden_layers 1

# --data_dir data/snips \
# --test_text snips_test_text.txt \
# --test_label snips_test_labels.txt \
# --vocab words.txt --tag_set tags.txt \
# --hidden_layer_size 1024 \
# --embedding 768 \
# --num_hidden_layers 1

# --data_dir data/ubuntu \
# --test_text ubuntu_test_text.txt \
# --test_label ubuntu_test_labels.txt \
# --vocab words.txt --tag_set tags.txt
