#!/bin/bash

python eval_ner.py --output_dir outputs \
    --t_labels deuutf_label_bert.txt \
    --p_label deuutf_predict_bert.txt \
    --text deuutf_text_bert.txt

#     --t_labels alliance_label_bert.txt \
#     --p_labels alliance_predict_bert.txt \
#     --text alliance_text_bert.txt

#     --t_labels wallet_label_bert.txt \
#     --p_labels wallet_predict_bert.txt \
#     --text wallet_text_bert.txt

#     --t_labels snips_label_bert.txt \
#     --p_labels snips_predict_bert.txt \
#     --text snips_text_bert.txt

#     --t_labels accounts_label_bert.txt \
#     --p_labels accounts_predict_bert.txt \
#     --text accounts_text_bert.txt

#     --t_labels nlu_label_bert.txt \
#     --p_labels nlu_predict_bert.txt \
#     --text nlu_text_bert.txt

#     --t_labels eng_label_bert.txt \
#     --p_label eng_predict_bert.txt \
#     --text eng_text_bert.txt

#     --t_labels deuutf_label_bert.txt \
#     --p_label deuutf_predict_bert.txt \
#     --text deuutf_text_bert.txt

#     --t_labels ubuntu_label.txt \
#     --p_labels ubuntu_predict.txt \
#     --text ubuntu_text.txt
