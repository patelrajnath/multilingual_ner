import json

import pandas
from datautils import Doc
from datautils.biluo_from_predictions import get_biluo
from datautils.iob_utils import offset_from_biluo


vocab_file = "D:\Downloads\ent_vocab_custom"
lookup_table = {}
with open(vocab_file, "r") as f:
    entities_json = [json.loads(line) for line in f]
    for item in entities_json:
        for title, language in item["entities"]:
            value = item["info_box"]
            if value:
                lookup_table[title.lower()] = value

# data_df = pandas.read_csv('data/conll2003/eng.train.train.csv', sep='\t')
# data_df = pandas.read_csv('data/wallet/wallet_train_text.txt.csv', sep='\t')
# data_df = pandas.read_csv('data/accounts/accounts_train_text.txt.csv', sep='\t')
# data_df = pandas.read_csv('data/alliance/alliance_train_text.txt.csv', sep='\t')
# data_df = pandas.read_csv('data/nlu/nlu_train_text.txt.csv', sep='\t')
# data_df = pandas.read_csv('data/snips/snips_train_text.txt.csv', sep='\t')
data_df = pandas.read_csv('data/ubuntu/ubuntu_train_text.txt.csv', sep='\t')
ent_count = 0
ent_coverage = 0
for index, row in data_df.iterrows():
    doc = Doc(row.text)
    tag_labels_true = row.labels.strip().replace('_', '-').split()
    biluo_tags_true = get_biluo(tag_labels_true)
    offset_true_labels = offset_from_biluo(doc, biluo_tags_true)
    print(row.text)
    for start, end, _ in offset_true_labels:
        ent_count += 1
        ent_text = row.text[start:end].lower()
        if ent_text in lookup_table:

            ent_coverage += 1

print(ent_count, ent_coverage, ent_coverage/ent_count)
