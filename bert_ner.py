import time
from math import inf

import torch
from torch import nn
from torch.nn import functional as F
from models import bert_data
from models.embedders import BERTEmbedder


class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        self.params = params
        # maps each token to an embedding_dim vector
        # self.embedding = nn.Embedding(params.vocab_size, params.embedding_dim)
        # self.embedding = Embeddings(params.vocab_size, params.embedding_dim)
        self.embeddings = BERTEmbedder.create(model_name=model_name, device=device, mode=mode, is_freeze=is_freeze)

        # the LSTM takens embedded sentence
        self.lstm = nn.LSTM(self.params.embedding_dim, self.params.hidden_layer_size // 2,
                            batch_first=True, bidirectional=True)
        # fc layer transforms the output to give the final output layer
        self.fc = nn.Linear(self.params.hidden_layer_size, self.params.number_of_tags)

    def forward(self, s):
        # apply the embedding layer that maps each token to its embedding
        s = self.embeddings(s)  # dim: batch_size x batch_max_len x embedding_dim

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
        self.hidden_layer_size = 1024
        self.num_hidden_layers = 1
        self.embedding_dim = 768
        self.batch_size = 32
        self.dropout = 0.1
        self.optimizer = 'sgd'
        self.learning_rate = 0.01
        self.lr_decay_pow = 1
        self.epochs = 10
        self.seed = 999
        self.max_steps = 1500
        self.patience = 10
        self.eval_each_epoch = True
        self.number_of_tags = 9


data = bert_data.LearnData.create(
    train_df_path="data/conll2003/eng.train.train.csv",
    valid_df_path="data/conll2003/eng.testa.dev.csv",
    idx2labels_path="data/conll2003/idx2labels2.txt",
    clear_cache=True,
    model_name="bert-base-cased"
)
model_name='bert-base-multilingual-cased'
mode="weighted"
is_freeze=True
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
params = hparamset()
model = Net(params)
optimizer = torch.optim.Adam(model.parameters())

updates = 1
total_loss = 0
best_loss = +inf
stop_training = False

for epoch in range(params.epochs):
    for batch in data.train_dl:
        updates += 1
        input_, labels_mask, input_type_ids, labels = batch
        labels = labels.view(-1)
        labels_mask = labels_mask.view(-1)
        output = model(batch)
        loss = loss_fn(output, labels, labels_mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.data

        if updates % params.patience == 0:
            print(f'Epoch: {epoch}, Updates:{updates}, Loss: {total_loss}')
            if best_loss > total_loss:
                best_loss = total_loss
            total_loss = 0

        if updates % params.max_steps == 0:
            stop_training = True
            break

    if stop_training:
        break
