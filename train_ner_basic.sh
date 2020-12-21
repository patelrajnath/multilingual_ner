#!/bin/bash

python train_ner_basic.py \
--data_dir data/nlu \
--train_text nlu_train_text.txt \
--train_label nlu_train_labels.txt \
--test_text nlu_test_text.txt \
--test_label nlu_test_labels.txt \
--vocab words.txt --tag_set tags.txt \
--batch_size 32 \
--arch attn_ner_medium \
--max_steps 1500 \
--attn_num_heads 1 \
--cpu

# --data_dir data/alliance \
# --train_text alliance_train_text.txt \
# --train_label alliance_train_labels.txt \
# --test_text alliance_test_text.txt \
# --test_label alliance_test_labels.txt \
# --vocab words.txt --tag_set tags.txt \
# --batch_size 32 \
# --arch attn_ner \
# --max_steps 1500 \
# --cpu

# --data_dir data/wallet \
# --train_text wallet_train_text.txt \
# --train_label wallet_train_labels.txt \
# --test_text wallet_test_text.txt \
# --test_label wallet_test_labels.txt \
# --vocab words.txt --tag_set tags.txt \
# --batch_size 32 \
# --arch attn_ner \
# --max_steps 1500 \
# --cpu

# --data_dir data/ubuntu \
# --train_text ubuntu_train_text.txt \
# --train_label ubuntu_train_labels.txt \
# --test_text ubuntu_test_text.txt \
# --test_label ubuntu_test_labels.txt \
# --vocab words.txt --tag_set tags.txt \
# --batch_size 16 \
# --arch ner \
# --max_steps 1500 \
# --cpu

# --data_dir data/accounts \
# --train_text accounts_train_text.txt \
# --train_label accounts_train_labels.txt \
# --test_text accounts_test_text.txt \
# --test_label accounts_test_labels.txt \
# --vocab words.txt --tag_set tags.txt \
# --batch_size 16 \
# --arch attn_ner \
# --max_steps 1500 \
# --cpu

# --data_dir data/nlu \
# --train_text nlu_train_text.txt \
# --train_label nlu_train_labels.txt \
# --test_text nlu_test_text.txt \
# --test_label nlu_test_labels.txt \
# --vocab words.txt --tag_set tags.txt \
# --batch_size 32 \
# --arch attn_ner \
# --max_steps 1500 \
# --cpu

# --data_dir data/snips \
# --train_text snips_train_text.txt \
# --train_label snips_train_labels.txt \
# --test_text snips_test_text.txt \
# --test_label snips_test_labels.txt \
# --vocab words.txt --tag_set tags.txt \
# --batch_size 32 \
# --arch attn_ner \
# --max_steps 1500 \
# --cpu
