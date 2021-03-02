import json

import pandas
from datautils import Doc
from datautils.biluo_from_predictions import get_biluo
from datautils.iob_utils import offset_from_biluo

vocab_file = "D:\\Downloads\\ent_vocab_custom_all_filtered"
lookup_table = {}
with open(vocab_file, "r") as f:
    for line in f:
        item = json.loads(line)
        for title, language in item["entities"]:
            value = item["info_box"]
            if value:
                lookup_table[title.lower()] = value

data_df = pandas.read_csv('data/conll2003/eng.train.train.csv', sep='\t')
# data_df = pandas.read_csv('data/conll2003/eng.testa.dev.csv', sep='\t')
# data_df = pandas.read_csv('data/conll2003/eng.testb.dev.csv', sep='\t')
# data_df = pandas.read_csv('data/wallet/wallet_train_text.txt.csv', sep='\t')
# data_df = pandas.read_csv('data/accounts/accounts_train_text.txt.csv', sep='\t')
# data_df = pandas.read_csv('data/alliance/alliance_train_text.txt.csv', sep='\t')
# data_df = pandas.read_csv('data/nlu/nlu_train_text.txt.csv', sep='\t')
# data_df = pandas.read_csv('data/snips/snips_train_text.txt.csv', sep='\t')
# data_df = pandas.read_csv('data/ubuntu/ubuntu_train_text.txt.csv', sep='\t')
ent_count = 0
ent_coverage = 0
tag_features = {}
for index, row in data_df.iterrows():
    doc = Doc(row.text)
    tag_labels_true = row.labels.strip().replace('_', '-').split()
    biluo_tags_true = get_biluo(tag_labels_true)
    offset_true_labels = offset_from_biluo(doc, biluo_tags_true)
    print(row.text)
    for start, end, tag in offset_true_labels:
        ent_count += 1
        ent_text = row.text[start:end].lower()
        if ent_text in lookup_table:
            ent_coverage += 1
            try:
                tag_features[tag].append(lookup_table[ent_text])
            except:
                tag_features[tag] = [lookup_table[ent_text]]

print(ent_count, ent_coverage, ent_coverage / ent_count)

with open('feats_stats', 'w', encoding='utf-8') as fout:
    for tag in tag_features:
        feats_count = {}
        feats = tag_features[tag]
        for feat in feats:
            for key, value in feat.items():
                try:
                    feats_count[key] += 1
                except:
                    feats_count[key] = 1
        print(feats_count)
        sorted_value = {k: v for k, v in sorted(feats_count.items(), key=lambda item: item[1], reverse=True)}
        json.dump({'tag': tag, 'features': sorted_value}, fout, default=str)
        fout.write('\n')
