#!/bin/bash

python eval_ner.py --output_dir outputs \
	--t_labels ubuntu_labels.txt \
	--p_labels ubuntu_predict.txt \
	--text ubuntu_text.txt
