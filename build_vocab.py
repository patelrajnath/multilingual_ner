import sys

vocab = {}
with open("data/nlu_train_text.txt") as inp:
    for line in inp:
        words = line.split()
        for word in words:
            try:
                vocab[word] += 1
            except:
                vocab[word] = 1

with open('data/words.txt', 'w') as out:
    print(vocab, file=out)

tags = {}
with open("data/nlu_train_labels.txt") as inp:
    for line in inp:
        words = line.split()
        for word in words:
            try:
                tags[word] += 1
            except:
                tags[word] = 1

with open('data/tags.txt', 'w') as out:
    print(tags, file=out)
