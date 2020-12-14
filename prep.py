import re
import pandas as pd
# data_dir = "data/conll2003-de/NER-de-dev.tsv"
data_dir = "data/conll2003-de/NER-de-train.tsv"
with open(data_dir, encoding='utf8') as f, \
        open('data/conll2003-de/de-train-text.txt', 'w', encoding='utf8') as f_text, \
        open('data/conll2003-de/de-train-labels.txt', 'w', encoding='utf8') as f_label:
    s = ''
    e = ''
    text_list = []
    labels_list = []
    cls = []
    for line in f:
        if not line.startswith('#'):
            if line == '\n':
                # if len(s.strip().split()) != len(e.strip().split()):
                if not all([y.split("_")[0] == "O" for y in e.split()]) and \
                        len(s.strip().split()) == len(e.strip().split()):
                    text_list.append(s.strip())
                    labels_list.append(e.strip())
                    cls.append(False)
                s = ''
                e = ''
            else:
                words = line.split('\t')
                if len(words) == 4:
                    s += words[1].strip() + ' '
                    ent = re.sub('[^A-Z-]', '', words[3].strip())
                    e += ent.replace('-', '_') + ' '
df = pd.DataFrame([labels_list, text_list, cls])
df = df.transpose()
df.to_csv('data/conll2003-de/temp.csv', index=False, sep="\t")
