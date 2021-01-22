### Multilingual NER
This project is to design and develop a light weight 
Multilingual NER system. The Multilingual capability 
we want to have as zero-shot learning from pretrained 
Contextualized Language Models.

### Quick Start

#### Prerequisites

* python (3.7+)
* Linux OS (For Windows you need to install pip)

#### Install dependencies-

```bash
$pip install -r requirements.txt
```

### Train model
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
(by default it usage conll2003 English data)
```bash
$bash train_ner_bert.sh
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
Copyright Raj Nath Patel 2017 - present

***multilingual_ner*** is free software: you can redistribute 
it and/or modify it under the terms of the GNU General Public 
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

You should have received a copy of the GNU General Public 
License along with Indic NLP Library. 
If not, see http://www.gnu.org/licenses/.