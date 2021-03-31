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

# data_df = pandas.read_csv('data/conll2003/eng.train.train.csv', sep='\t')
# data_df = pandas.read_csv('data/conll2003/eng.testa.dev.csv', sep='\t')
data_df = pandas.read_csv('data/conll2003/eng.testb.dev.csv', sep='\t')
# data_df = pandas.read_csv('data/wallet/wallet_train_text.txt.csv', sep='\t')
# data_df = pandas.read_csv('data/accounts/accounts_train_text.txt.csv', sep='\t')
# data_df = pandas.read_csv('data/alliance/alliance_train_text.txt.csv', sep='\t')
# data_df = pandas.read_csv('data/nlu/nlu_train_text.txt.csv', sep='\t')
# data_df = pandas.read_csv('data/snips/snips_train_text.txt.csv', sep='\t')
# data_df = pandas.read_csv('data/ubuntu/ubuntu_train_text.txt.csv', sep='\t')
ent_count = 0
ent_coverage = 0
tag_features = {}
new_texts = []
updated_labels = []
for index, row in data_df.iterrows():
    text_ = row.text
    doc = Doc(text_)
    labels = row.labels
    tag_labels_true = labels.strip().replace('_', '-').split()
    biluo_tags_true = get_biluo(tag_labels_true)
    offset_true_labels = offset_from_biluo(doc, biluo_tags_true)
    new_labels = labels.split()

    chunk_start = 0
    chunks = []
    # Convert text into chunks
    for start, end, _ in offset_true_labels:
        chunk_text = text_[chunk_start: start].strip()
        chunk_entity = text_[start: end].strip()
        chunk_start = end

        if chunk_text:
            chunks.append(chunk_text)

        if chunk_entity:
            chunks.append(chunk_entity)

    # Append the last chunk if not empty
    last_chunk = text_[chunk_start:].strip()
    if last_chunk:
        chunks.append(last_chunk)
    offset = 0
    has_overlap = False
    for chunk in chunks:
        has_kg = False
        chunk = chunk.lower()
        num_words = len(chunk.split())
        if chunk in lookup_table:
            has_kg = True
            has_overlap = True
        if not has_kg:
            for j in range(offset, num_words):
                new_labels[j] = 'O'
        offset += num_words
    if has_overlap:
        print(labels)
        updated_labels.append(' '.join(new_labels))
        new_texts.append(text_)
        print(text_)
        print(index)
        ent_coverage += 1
    ent_count += 1

print(ent_count, ent_coverage, ent_coverage / ent_count)

data = {'labels': updated_labels, 'text': new_texts}
df = pandas.DataFrame(data)
# df.to_csv('data/conll2003/eng.train.train_filtered.csv', sep='\t', index=False)
# df.to_csv('data/conll2003/eng.testa.dev_filtered.csv', sep='\t', index=False)
df.to_csv('data/conll2003/eng.testb.dev_filtered.csv', sep='\t', index=False)
