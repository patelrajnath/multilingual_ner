import ast
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from batchers import SamplingBatcher

vocab = {'UNK': 0, 'PAD': 1}
num_specials_tokens = len(vocab)
with open('data/words.txt') as f:
    words = ast.literal_eval(f.read()).keys()
    for i, l in enumerate(words):
        vocab[l] = i + num_specials_tokens

START_TAG = "<START>"
STOP_TAG = "<STOP>"
tag_map = {START_TAG: 0, STOP_TAG: 1}
num_specials_tags = len(tag_map)
with open('data/tags.txt') as f:
    words = ast.literal_eval(f.read()).keys()
    for i, l in enumerate(words):
        tag_map[l] = i + num_specials_tags

train_sentences = []
train_labels = []

with open('data/nlu_train_text.txt') as f:
    for sentence in f:
        # replace each token by its index if it is in vocab else use index of UNK
        s = [vocab[token] if token in vocab
            else vocab['UNK']
            for token in sentence.strip().split()]
        train_sentences.append(s)

with open('data/nlu_train_labels.txt') as f:
    for sentence in f:
        # replace each label by its index
        l = [tag_map[label] for label in sentence.strip().split()]
        train_labels.append(l)

# Sort the data according to the length
sorted_idx = np.argsort([len(s) for s in train_sentences])
train_sentences = [train_sentences[id] for id in sorted_idx]
train_labels = [train_labels[id] for id in sorted_idx]

test_sentences = []
test_labels = []
with open('data/nlu_test_text.txt') as f:
    for sentence in f:
        # replace each token by its index if it is in vocab else use index of UNK
        s = [vocab[token] if token in vocab else vocab['UNK']
             for token in sentence.strip().split()]
        test_sentences.append(s)

with open('data/nlu_test_labels.txt') as f:
    for sentence in f:
        # replace each label by its index
        l = [tag_map[label] for label in sentence.strip().split()]
        test_labels.append(l)

count = 1
for t, l in zip(train_sentences, train_labels):
    count += 1
    if len(t) != len(l):
        print(f'Error:{len(t)}, {len(l)}, {count}')


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        self.params = params
        # maps each token to an embedding_dim vector
        self.embedding = nn.Embedding(params.vocab_size, params.embedding_dim)

        # the LSTM takens embedded sentence
        self.lstm = nn.LSTM(params.embedding_dim, self.params.hidden_layer_size // 2,
                            num_layers=1, bidirectional=True)
        # fc layer transforms the output to give the final output layer
        self.fc = nn.Linear(params.hidden_layer_size, params.number_of_tags)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(params.number_of_tags, params.number_of_tags))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_map[START_TAG], :] = -10000
        self.transitions.data[:, tag_map[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.params.hidden_layer_size // 2),
                torch.randn(2, 1, self.params.hidden_layer_size // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.params.vocab_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][tag_map[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.params.vocab_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.params.vocab_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[tag_map[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.embedding(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.params.hidden_layer_size)
        lstm_feats = self.fc(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([tag_map[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[tag_map[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward_crf(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

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


def loss_fn(outputs, labels):
    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.view(-1)

    # mask out 'PAD' tokens
    mask = (labels >= 0).float()

    # the number of tokens is the sum of elements in mask
    num_tokens = int(torch.sum(mask).item())

    # pick the values corresponding to labels and multiply by mask
    outputs = outputs[range(outputs.shape[0]), labels]*mask

    # cross entropy loss for all non 'PAD' tokens
    return -torch.sum(outputs)/num_tokens


class hparamset():
    def __init__(self):
        self.batchsize = 16
        self.max_sts_score = 5
        self.balance_data = False
        self.output_size = None
        self.activation = 'relu'
        self.hidden_layer_size = 512
        self.num_hidden_layers = 1
        self.batch_size = 16
        self.dropout = 0.1
        self.optimizer = 'sgd'
        self.learning_rate = 0.7
        self.lr_decay_pow = 1
        self.epochs = 100
        self.seed = 999
        self.max_steps = 15000
        self.patience = 100
        self.eval_each_epoch = True
        self.vocab_size = len(vocab)
        self.embedding_dim = 256
        self.number_of_tags = len(tag_map)


params = hparamset()
model = Net(params=params)

optimizer = torch.optim.Adam(model.parameters())

batcher = SamplingBatcher(np.asarray(train_sentences), np.asarray(train_labels), batch_size=32, pad_id=vocab['PAD'])

updates = 1
total_loss = 0
num_epochs = 30
for epoch in range(num_epochs):
    for batch in batcher:
        updates += 1
        batch_data, batch_labels = batch
        optimizer.zero_grad()
        # pass through model, perform backpropagation and updates
        output_batch = model(batch_data)
        loss = loss_fn(output_batch, batch_labels)
        # loss = model.neg_log_likelihood(batch_data, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.data
        if updates % params.patience == 0:
            print(f'Epoch: {epoch}, Loss: {total_loss}')
            total_loss = 0

batcher_test = SamplingBatcher(np.asarray(test_sentences), np.asarray(test_labels),
                          batch_size=32, pad_id=vocab['PAD'])

with torch.no_grad():
    model.eval()
    prediction = []
    true_labels = []
    for batch in batcher_test:
        batch_data, batch_labels = batch
        predict = model(batch_data)
        predict_labels = predict.argmax(dim=1)
        prediction.extend(predict_labels.view(-1))
        true_labels.extend(batch_labels.view(-1))
    print(f1_score(prediction, true_labels, average='weighted') * 100)
