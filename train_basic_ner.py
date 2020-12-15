import ast
import time
from math import inf

import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score

from batchers import SamplingBatcher
from model_utils import save_state, load_model_state, set_seed

# Set seed to have consistent results
from token_emb import Embeddings

set_seed(seed_value=999)
np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning) 
# data_type = 'ubuntu'
# data_type = 'accounts'
data_type = 'nlu'
# data_type = 'snips'
vocab = {'UNK': 0, 'PAD': 1}
num_specials_tokens = len(vocab)
with open('data/{}/words.txt'.format(data_type), encoding='utf8') as f:
    words = ast.literal_eval(f.read()).keys()
    for i, l in enumerate(words):
        vocab[l] = i + num_specials_tokens

START_TAG = "<START>"
STOP_TAG = "<STOP>"
O_TAG = 'O'
tag_map = {START_TAG: 0, STOP_TAG: 1}
num_specials_tags = len(tag_map)
with open('data/{}/tags.txt'.format(data_type), encoding='utf8') as f:
    words = ast.literal_eval(f.read()).keys()
    for i, l in enumerate(words):
        tag_map[l] = i + num_specials_tags

train_sentences = []
train_labels = []

with open('data/{0}/{0}_train_text.txt'.format(data_type), encoding='utf8') as f:
    for sentence in f:
        # replace each token by its index if it is in vocab else use index of UNK
        s = [vocab[token] if token in vocab
            else vocab['UNK']
            for token in sentence.strip().split()]
        train_sentences.append(s)

with open('data/{0}/{0}_train_labels.txt'.format(data_type), encoding='utf8') as f:
    for sentence in f:
        # replace each label by its index
        l = [tag_map[label] for label in sentence.strip().split()]
        train_labels.append(l)

# Sort the data according to the length
# sorted_idx = np.argsort([len(s) for s in train_sentences])
# train_sentences = [train_sentences[id] for id in sorted_idx]
# train_labels = [train_labels[id] for id in sorted_idx]

test_sentences = []
test_labels = []
with open('data/{0}/{0}_test_text.txt'.format(data_type), encoding='utf8') as f:
    for sentence in f:
        # replace each token by its index if it is in vocab else use index of UNK
        s = [vocab[token] if token in vocab else vocab['UNK']
             for token in sentence.strip().split()]
        test_sentences.append(s)

with open('data/{0}/{0}_test_labels.txt'.format(data_type), encoding='utf8') as f:
    for sentence in f:
        # replace each label by its index
        l = [tag_map.get(label, tag_map.get('O')) for label in sentence.strip().split()]
        test_labels.append(l)

count = 1
train_sentences_fixed = []
train_labels_fixed = []
for t, l in zip(train_sentences, train_labels):
    count += 1
    if len(t) != len(l):
        print(f'Error:{len(t)}, {len(l)}, {count}')
    else:
        train_sentences_fixed.append(t)
        train_labels_fixed.append(l)

train_sentences = train_sentences_fixed
train_labels = train_labels_fixed

test_sentences_fixed = []
test_labels_fixed = []
for t, l in zip(test_sentences, test_labels):
    count += 1
    if len(t) != len(l):
        print(f'Error:{len(t)}, {len(l)}, {count}')
    else:
        test_sentences_fixed.append(t)
        test_labels_fixed.append(l)
test_sentences = test_sentences_fixed
test_labels = test_labels_fixed


class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        self.params = params
        # maps each token to an embedding_dim vector
        self.embedding = nn.Embedding(params.vocab_size, params.embedding_dim)
        # self.embedding = Embeddings(params.vocab_size, params.embedding_dim)

        # the LSTM takens embedded sentence
        self.lstm = nn.LSTM(self.params.embedding_dim, self.params.hidden_layer_size // 2,
                            batch_first=True, bidirectional=True)
        # fc layer transforms the output to give the final output layer
        self.fc = nn.Linear(self.params.hidden_layer_size, self.params.number_of_tags)

    def forward(self, s):
        # apply the embedding layer that maps each token to its embedding
        s = self.embedding(s)  # dim: batch_size x batch_max_len x embedding_dim

        # run the LSTM along the sentences of length batch_max_len
        s, _ = self.lstm(s)  # dim: batch_size x batch_max_len x lstm_hidden_dim

        # reshape the Variable so that each row contains one token
        s = s.reshape(-1, s.shape[2])  # dim: batch_size*batch_max_len x lstm_hidden_dim

        # apply the fully connected layer and obtain the output for each token
        s = self.fc(s)  # dim: batch_size*batch_max_len x num_tags

        return F.log_softmax(s, dim=1)  # dim: batch_size*batch_max_len x num_tags


def loss_fn(outputs, labels, mask):
    # the number of tokens is the sum of elements in mask
    num_labels = int(torch.sum(mask).item())

    # pick the values corresponding to labels and multiply by mask
    outputs = outputs[range(outputs.shape[0]), labels]*mask

    # cross entropy loss for all non 'PAD' tokens
    return -torch.sum(outputs)/num_labels


class hparamset():
    def __init__(self):
        self.batchsize = 16
        self.max_sts_score = 5
        self.balance_data = False
        self.output_size = None
        self.activation = 'relu'
        self.hidden_layer_size = 512
        self.num_hidden_layers = 1
        self.embedding_dim = 256
        self.batch_size = 32
        self.dropout = 0.1
        self.optimizer = 'sgd'
        self.learning_rate = 0.01
        self.lr_decay_pow = 1
        self.epochs = 100
        self.seed = 999
        self.max_steps = 1500
        self.patience = 100
        self.eval_each_epoch = True
        self.vocab_size = len(vocab)
        self.number_of_tags = len(tag_map)


params = hparamset()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
model = Net(params=params)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters())

batcher = SamplingBatcher(np.asarray(train_sentences, dtype=object), np.asarray(train_labels, dtype=object),
                          batch_size=params.batch_size, pad_id=vocab['PAD'])

updates = 1
total_loss = 0
best_loss = +inf
stop_training = False
start_time = time.time()
for epoch in range(params.epochs):
    for batch in batcher:
        updates += 1
        batch_data, batch_labels, batch_len, mask_x, mask_y = batch
        optimizer.zero_grad()
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        mask_y = mask_y.to(device)
        # pass through model, perform backpropagation and updates
        output_batch = model(batch_data)
        loss = loss_fn(output_batch, batch_labels, mask_y)
        # loss = model.neg_log_likelihood(batch_data, batch_labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.data
        if updates % params.patience == 0:
            print(f'Epoch: {epoch}, Updates:{updates}, Loss: {total_loss}')
            if best_loss > total_loss:
                save_state('best_model.pt', model, loss_fn, optimizer, updates)
                best_loss = total_loss
            total_loss = 0
        if updates % params.max_steps == 0:
            stop_training = True
            break

    if stop_training:
        break
print('Training time:{}'.format(time.time()-start_time))
updates = load_model_state('best_model.pt', model)
with open('label.txt', 'w') as t, open('predict.txt', 'w') as p:
    with torch.no_grad():
        model.eval()
        prediction = []
        true_labels = []
        for text, label in zip(test_sentences, test_labels):
            text = torch.LongTensor(text).unsqueeze(0).to(device)
            lable = torch.LongTensor(label).unsqueeze(0).to(device)
            predict = model(text)
            predict_labels = predict.argmax(dim=1)
            predict_labels = predict_labels.view(-1)
            # score, predict_labels = model.forward_crf(batch_data)
            lable = lable.view(-1)
            a = predict_labels.cpu().data.tolist()
            b = lable.cpu().data.tolist()
            prediction.extend(a)
            true_labels.extend(b)
            a = [str(i) for i in a]
            b = [str(i) for i in b]
            t.write(' '.join(b) + '\n')
            p.write(' '.join(a) + '\n')

    t = list()
    p = list()
    for a, b in zip(true_labels, prediction):
        if a == tag_map.get(O_TAG) and b == tag_map.get(O_TAG):
            continue
        t.append(a)
        p.append(b)
    print(len(t), len(p))
    print(f1_score(t, p, average='micro') * 100)
    print(f1_score(t, p, average='macro') * 100)
    print(f1_score(t, p, average='weighted') * 100)
    print(precision_score(t, p, average='weighted', zero_division=True) * 100)
    print(recall_score(t, p, average='weighted', zero_division=True) * 100)
