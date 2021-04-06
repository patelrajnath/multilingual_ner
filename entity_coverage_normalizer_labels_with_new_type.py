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
data_df = pandas.read_csv('data/conll2003/eng.testa.dev.csv', sep='\t')
# data_df = pandas.read_csv('data/conll2003/eng.testb.dev.csv', sep='\t')
# data_df = pandas.read_csv('data/wallet/wallet_train_text.txt.csv', sep='\t')
# data_df = pandas.read_csv('data/accounts/accounts_train_text.txt.csv', sep='\t')
# data_df = pandas.read_csv('data/alliance/alliance_train_text.txt.csv', sep='\t')
# data_df = pandas.read_csv('data/nlu/nlu_train_text.txt.csv', sep='\t')
# data_df = pandas.read_csv('data/snips/snips_train_text.txt.csv', sep='\t')
# data_df = pandas.read_csv('data/ubuntu/ubuntu_train_text.txt.csv', sep='\t')
# data_df = pandas.read_csv('data/tweeter_nlp/ner.txt.train.csv', sep='\t')
# data_df = pandas.read_csv('data/GMB/ner.csv', sep='\t')
# data_df = pandas.read_csv('data/kaggle-ner/ner.csv', sep=',')

ent_count = 0
selected = 0
ignored = 0
tag_features = {}
new_texts = []
updated_labels = []
for index, row in data_df.iterrows():
    text_ = row.text
    words = text_.split()
    doc = Doc(text_)
    labels = row.labels
    tag_labels_true = labels.strip().replace('_', '-').split()
    if len(words) != len(tag_labels_true):
        ignored += 1
        # print(index, row.text)
        continue
    biluo_tags_true = get_biluo(tag_labels_true)
    offset_true_labels = offset_from_biluo(doc, biluo_tags_true)
    # if index == 49:
    # print(text_)
    # print(tag_labels_true)
    # print(offset_true_labels)
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
    print(chunks)
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
            for j in range(offset, offset + num_words):
                # print(j)
                # print(new_labels[j])
                if new_labels[j] != 'O':
                    new_labels[j] = new_labels[j] + '_NOKG'
        offset += num_words
    # if has_overlap:
    #     print(new_labels)
    updated_labels.append(' '.join(new_labels))
    new_texts.append(text_)
    # print(text_)
    # print(index)
    selected += 1
    ent_count += 1
        # exit()


print(ent_count, selected, selected / ent_count)

data = {'labels': updated_labels, 'text': new_texts}
df = pandas.DataFrame(data)
# df.to_csv('data/conll2003/eng.train.train_normalized.csv', sep='\t', index=False)
df.to_csv('data/conll2003/eng.testa.dev_normalized.csv', sep='\t', index=False)
# df.to_csv('data/conll2003/eng.testb.dev_normalized.csv', sep='\t', index=False)
# df.to_csv('data/tweeter_nlp/ner.txt.train_filtered.csv', sep='\t', index=False)
# df.to_csv('data/GMB/ner_filtered.csv', sep='\t', index=False)
# df.to_csv('data/kaggle-ner/ner_filtered.csv', sep='\t', index=False)
