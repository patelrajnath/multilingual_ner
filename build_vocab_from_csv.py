import json
import sys

import pandas

file_csv = sys.argv[1]
out_dir = sys.argv[2]
data_df = pandas.read_csv(file_csv, delimiter='\t', encoding='utf8')
vocab = {}
for line in data_df.text.tolist():
    words = line.split()
    for word in words:
        try:
            vocab[word] += 1
        except:
            vocab[word] = 1

with open('{}/words.txt'.format(out_dir), 'w', encoding='utf8') as out:
    json.dump(vocab, out, indent=4)
    # print(vocab, file=out)

tags = {}
for line in data_df.labels.tolist():
    words = line.split()
    for word in words:
        try:
            tags[word] += 1
        except:
            tags[word] = 1

with open('{}/tags.txt'.format(out_dir), 'w', encoding='utf8') as out:
    json.dump(tags, out, indent=4)

print(len(vocab))
print(len(tags))
