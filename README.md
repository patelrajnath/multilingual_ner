### Multilingual NER
This project is to design and develop a light weight 
Multilingual NER system. The Multilingual capability 
we want to have as zero-shot learning from pretrained 
Contextualized Language Models.

### Quick Start

#### Prerequisites

* python (3.7+)
* Linux OS (For Windows use WSL)

#### Install dependencies-

```bash
$pip install -r requirements.txt
```

### Train and Test NER Models

#### Train Simple model
Change the data in shell script 
(by default it usage conll2003 English data)
```bash
$bash train_ner_basic.sh
```

#### Test Simple model
The default test data is conll2003 english "testa" partition.
```bash
$bash decode_ner_basic.sh
```

#### Train BERT model (multilingual)
Change the data in shell script 
(by default it usage conll2003 English data). 
```bash
$bash train_ner_bert.sh
```

The default model used in Distil-Bert. 
Change the model using `--model_type`, and `--model_name` options. 
For example, to change to `roberta` model use the following-
```bash
python train_ner_bert.py \
    --arch bert_ner \
    --batch_size 32 \
    --data_dir data/conll2003 \
    --train eng.train.train.csv \
    --test eng.testa.dev.csv \
    --model_type roberta \
    --model_name roberta-base \
    --cache_features \
    --in_memory_cache \
    --cpu \
```


#### Decode BERT model (multilingual)
The default test data is conll2003 German "testa" partition.
**Note**: We trained the model using only English data, and 
testing on German data. 

```bash
$bash decode_ner_bert.sh
```



### Author
Raj Nath Patel (patelrajnath@gmail.com)

Linkedin: https://www.linkedin.com/in/raj-nath-patel-2262b024/

### LICENSE
Copyright Raj Nath Patel 2020 - present

***multilingual_ner*** is free software: you can redistribute 
it and/or modify it under the terms of the GNU General Public 
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

You should have received a copy of the GNU General Public 
License along with Indic NLP Library. 
If not, see http://www.gnu.org/licenses/.