#!/usr/bin/bash

python train_ner_basic.py --data_dir data/ubuntu \
--train_text ubuntu_train_text.txt \
--train_label ubuntu_train_labels.txt \
--test_text ubuntu_test_text.txt \
--test_label ubuntu_test_labels.txt \
--vocab words.txt --tag_set tags.txt \
--batch_size 8