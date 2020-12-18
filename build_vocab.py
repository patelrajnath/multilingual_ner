import json

vocab = {}
data_type = 'snips'
with open("data/{0}/{0}_train_text.txt".format(data_type), encoding='utf8') as inp:
    for line in inp:
        words = line.split()
        for word in words:
            try:
                vocab[word] += 1
            except:
                vocab[word] = 1

with open('data/{}/words.txt'.format(data_type), 'w', encoding='utf8') as out:
    json.dump(vocab, out, indent=4)
    # print(vocab, file=out)

tags = {}
with open("data/{0}/{0}_train_labels.txt".format(data_type), encoding='utf8') as inp:
    for line in inp:
        words = line.split()
        for word in words:
            try:
                tags[word] += 1
            except:
                tags[word] = 1

with open('data/{}/tags.txt'.format(data_type), 'w', encoding='utf8') as out:
    json.dump(tags, out, indent=4)

print(len(vocab))
print(len(tags))
