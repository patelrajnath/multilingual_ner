import ast
import os
import time
from math import inf

import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

from batchers import SamplingBatcher
from document.doc import Doc
from eval.biluo_from_predictions import get_biluo
from eval.iob_utils import offset_from_biluo
from model_utils import save_state, load_model_state, set_seed

# Set seed to have consistent results
set_seed(seed_value=999)
np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning) 
data_type = 'accounts'
# data_type = 'alliance'
# data_type = 'wallet'
# data_type = 'ubuntu'
# data_type = 'snips'
# data_type = 'nlu'
vocab = {'UNK': 0, 'PAD': 1}
num_specials_tokens = len(vocab)
with open('data/{}/words.txt'.format(data_type), encoding='utf8') as f:
    words = ast.literal_eval(f.read()).keys()
    for i, l in enumerate(words):
        vocab[l] = i + num_specials_tokens

idx_to_word = {vocab[key]: key for key in vocab}

START_TAG = "<START>"
STOP_TAG = "<STOP>"
O_TAG = 'O'
tag_to_idx = {START_TAG: 0, STOP_TAG: 1}
num_specials_tags = len(tag_to_idx)
with open('data/{}/tags.txt'.format(data_type), encoding='utf8') as f:
    words = ast.literal_eval(f.read()).keys()
    for i, l in enumerate(words):
        tag_to_idx[l] = i + num_specials_tags

idx_to_tag = {tag_to_idx[key]: key for key in tag_to_idx}

train_sentences = []
train_labels = []

with open('data/{0}/{0}_train_text.txt'.format(data_type), encoding='utf8') as f:
    for sentence in f:
        # replace each token by its index if it is in vocab else use index of UNK
        s = [vocab[token] if token in vocab else vocab['UNK'] for token in sentence.strip().split()]
        train_sentences.append(s)

with open('data/{0}/{0}_train_labels.txt'.format(data_type), encoding='utf8') as f:
    for sentence in f:
        # replace each label by its index
        l = [tag_to_idx.get(label, tag_to_idx.get('O')) for label in sentence.strip().split()]
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
        l = [tag_to_idx.get(label, tag_to_idx.get('O')) for label in sentence.strip().split()]
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
        self.max_sts_score = 5
        self.balance_data = False
        self.output_size = None
        self.activation = 'relu'
        self.hidden_layer_size = 512
        self.num_hidden_layers = 1
        self.embedding_dim = 256
        self.batch_size = 8
        self.dropout = 0.1
        self.optimizer = 'sgd'
        self.learning_rate = 0.01
        self.lr_decay_pow = 1
        self.epochs = 0
        self.seed = 999
        self.max_steps = 1500
        self.patience = 100
        self.eval_each_epoch = True
        self.vocab_size = len(vocab)
        self.number_of_tags = len(tag_to_idx)


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
out_dir = 'outputs'
try:
    os.makedirs(out_dir)
except:
    pass

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
                save_state(f'{out_dir}/{data_type}_best_model.pt', model, loss_fn, optimizer, updates)
                best_loss = total_loss
            total_loss = 0
        if updates % params.max_steps == 0:
            stop_training = True
            break

    if stop_training:
        break

print('Training time:{}'.format(time.time()-start_time))


def get_idx_to_tag(label_ids):
    return [idx_to_tag.get(idx) for idx in label_ids]


def get_idx_to_word(words_ids):
    return [idx_to_word.get(idx) for idx in words_ids]


updates = load_model_state(f'{out_dir}/{data_type}_best_model.pt', model)

ne_class_list = set()
true_labels_for_testing = []
results_of_prediction = []
with open(f'{out_dir}/{data_type}_label.txt', 'w', encoding='utf8') as t, \
        open(f'{out_dir}/{data_type}_predict.txt', 'w', encoding='utf8') as p, \
        open(f'{out_dir}/{data_type}_text.txt', 'w', encoding='utf8') as textf:
    with torch.no_grad():
        model.eval()
        prediction_label_ids = []
        true_label_ids = []
        cnt = 0
        for text, label in zip(test_sentences, test_labels):
            cnt += 1
            text_tensor = torch.LongTensor(text).unsqueeze(0).to(device)
            lable = torch.LongTensor(label).unsqueeze(0).to(device)
            predict = model(text_tensor)
            predict_labels = predict.argmax(dim=1)
            predict_labels = predict_labels.view(-1)
            lable = lable.view(-1)

            predicted_labels = predict_labels.cpu().data.tolist()
            true_labels = lable.cpu().data.tolist()
            tag_labels_predicted = get_idx_to_tag(predicted_labels)
            tag_labels_true = get_idx_to_tag(true_labels)
            text_ = get_idx_to_word(text)

            tag_labels_predicted = ' '.join(tag_labels_predicted)
            tag_labels_true = ' '.join(tag_labels_true)
            text_ = ' '.join(text_)
            p.write(tag_labels_predicted + '\n')
            t.write(tag_labels_true + '\n')
            textf.write(text_ + '\n')

            tag_labels_true = tag_labels_true.strip().replace('_', '-').split()
            tag_labels_predicted = tag_labels_predicted.strip().replace('_', '-').split()
            biluo_tags_true = get_biluo(tag_labels_true)
            biluo_tags_predicted = get_biluo(tag_labels_predicted)

            doc = Doc(text_)
            offset_true_labels = offset_from_biluo(doc, biluo_tags_true)
            offset_predicted_labels = offset_from_biluo(doc, biluo_tags_predicted)

            ent_labels = dict()
            for ent in offset_true_labels:
                start, stop, ent_type = ent
                ent_type = ent_type.replace('_', '')
                ne_class_list.add(ent_type)
                if ent_type in ent_labels:
                    ent_labels[ent_type].append((start, stop))
                else:
                    ent_labels[ent_type] = [(start, stop)]
            true_labels_for_testing.append(ent_labels)

            ent_labels = dict()
            for ent in offset_predicted_labels:
                start, stop, ent_type = ent
                ent_type = ent_type.replace('_', '')
                if ent_type in ent_labels:
                    ent_labels[ent_type].append((start, stop))
                else:
                    ent_labels[ent_type] = [(start, stop)]
            results_of_prediction.append(ent_labels)


from eval.quality import calculate_prediction_quality
f1, precision, recall, results = \
    calculate_prediction_quality(true_labels_for_testing,
                                 results_of_prediction,
                                 tuple(ne_class_list))
print(f1, precision, recall, results)
