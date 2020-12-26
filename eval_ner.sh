#!/bin/bash

# prefix='eng'
# prefix='deuutf'
prefix='alliance'
# prefix='wallet'
# prefix='accounts'
# prefix='ubuntu'
# prefix='nlu'
# prefix='snips'

postfix='_bert'
postfix=''

python eval_ner.py --output_dir outputs \
    --t_labels ${prefix}_label${postfix}.txt \
    --p_label ${prefix}_predict${postfix}.txt \
    --text ${prefix}_text${postfix}.txt
