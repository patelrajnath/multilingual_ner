#!/bin/bash

python eval_ner.py --output_dir outputs \
    --t_labels deuutf_label_bert.txt \
    --p_label deuutf_predict_bert.txt \
    --text deuutf_text_bert.txt
#     --t_labels ubuntu_label.txt \
#     --p_labels ubuntu_predict.txt \
#     --text ubuntu_text.txt
