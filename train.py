import numpy
from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.nn import NLLLoss, CrossEntropyLoss
from torch.optim import Adam


class SBertNer(nn.Module):
    def __init__(self, hidden_size, num_labels, hidden_dropout_prob=0.01):
        super(SBertNer, self).__init__()
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, sequence_output):
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        return outputs


def prepare_corpus(filename):
    entity_labels = dict()
    text = []
    labels = []
    with open(filename, encoding='utf8') as f:
        for line in f:
            s = []
            n = []
            for tokens in line.split():
                splits = tokens.split('|')
                if len(splits) == 3:
                    word, pos, ner = splits
                    s.append(word)
                    n.append(ner)
                    try:
                        entity_labels[ner] += 1
                    except:
                        entity_labels[ner] = 1
                else:
                    print(line)
                    break
            text.append(s)
            labels.append(n)
    le = LabelEncoder()
    ents = list(entity_labels.keys())
    le.fit_transform(ents)
    labels_encoded = []
    for label in labels:
        label_encoded = le.transform(label)
        labels_encoded.append(label_encoded)
    return text, labels_encoded, le


def chunks(iterable, iterable2, size):
    from itertools import chain, islice
    iterator = iter(iterable)
    iterator2 = iter(iterable2)
    for first, second in zip(iterator, iterator2):
        yield list(chain([first], islice(iterator, size - 1))), list(chain([second], islice(iterator2, size - 1)))


embedder = SentenceTransformer('distilbert-base-nli-mean-tokens')
text, encoded_labels, le = prepare_corpus('data/corpus.iob')
n_labels = len(le.classes_)
n_samples = 100
text = text[:n_samples]
encoded_labels = encoded_labels[:n_samples]

sorted_idx = numpy.argsort([len(s) for s in text])
sorted_text = [text[id] for id in sorted_idx]
labels = [encoded_labels[id] for id in sorted_idx]

train_encodings = []
for index, t in enumerate(sorted_text):
    t_encodings = embedder.encode(t, convert_to_tensor=True)
    train_encodings.append(t_encodings)

model = SBertNer(hidden_size=768, num_labels=n_labels)
loss = CrossEntropyLoss()
optimizer = Adam(model.parameters())
epochs = 45
for e in range(epochs):
    epoch_loss = 0
    for batch in chunks(train_encodings, labels, size=2):
        emb, label = batch
        emb = torch.cat(emb, dim=1)
        print(emb.shape, len(label))
        # print(emb.size)
        logits = model(emb)[0]
        labels = torch.tensor(label)
        output = loss(logits.view(-1, n_labels), labels.view(-1))
        output.backward()
        optimizer.step()
        epoch_loss += output.data
    print(epoch_loss)
