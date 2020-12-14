import time
from math import inf

import torch
from torch import nn
from torch.nn import functional as F

from model_utils import set_seed, save_state, load_model_state
from models import bert_data, tqdm
from models.embedders import BERTEmbedder
from models.bert_data import get_data_loader_for_predict
from sklearn_crfsuite.metrics import flat_classification_report
from analyze_utils.utils import bert_labels2tokens, voting_choicer
from analyze_utils.plot_metrics import get_bert_span_report

set_seed(seed_value=999)


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
        self.patience = 100
        self.eval_each_epoch = True
        self.number_of_tags = 9


model_name = 'bert-base-multilingual-cased'
mode="weighted"
is_freeze=True
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
params = hparamset()
model = Net(params)
model = model.to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters())

data = bert_data.LearnData.create(
    train_df_path="data/conll2003/eng.train.train.csv",
    valid_df_path="data/conll2003/eng.testa.dev.csv",
    idx2labels_path="data/conll2003/idx2labels2.txt",
    clear_cache=True,
    model_name="bert-base-multilingual-cased",
    batch_size=params.batch_size
)

updates = 1
total_loss = 0
best_loss = +inf
stop_training = False
start = time.time()
for epoch in range(params.epochs):
    for batch in data.train_dl:
        updates += 1
        optimizer.zero_grad()
        input_, labels_mask, input_type_ids, labels = batch
        labels = labels.view(-1).to(device)
        labels_mask = labels_mask.view(-1).to(device)
        output = model(batch)
        loss = loss_fn(output, labels, labels_mask)
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
print(f'Training time: {time.time() - start}')


def transformed_result(preds, mask, id2label, target_all=None, pad_idx=0):
    preds_cpu = []
    targets_cpu = []
    lc = len(id2label)
    if target_all is not None:
        for batch_p, batch_t, batch_m in zip(preds, target_all, mask):
            for pred, true_, bm in zip(batch_p, batch_t, batch_m):
                sent = []
                sent_t = []
                bm = bm.sum().cpu().data.tolist()
                for p, t in zip(pred[:bm], true_[:bm]):
                    p = p.cpu().data.tolist()
                    p = p if p < lc else pad_idx
                    sent.append(p)
                    sent_t.append(t.cpu().data.tolist())
                preds_cpu.append([id2label[w] for w in sent])
                targets_cpu.append([id2label[w] for w in sent_t])
    else:
        for batch_p, batch_m in zip(preds, mask):

            for pred, bm in zip(batch_p, batch_m):
                assert len(pred) == len(bm)
                bm = bm.sum().cpu().data.tolist()
                sent = pred[:bm].cpu().data.tolist()
                preds_cpu.append([id2label[w] for w in sent])
    if target_all is not None:
        return preds_cpu, targets_cpu
    else:
        return preds_cpu


def transformed_result_cls(preds, target_all, cls2label, return_target=True):
    preds_cpu = []
    targets_cpu = []
    for batch_p, batch_t in zip(preds, target_all):
        for pred, true_ in zip(batch_p, batch_t):
            preds_cpu.append(cls2label[pred.cpu().data.tolist()])
            if return_target:
                targets_cpu.append(cls2label[true_.cpu().data.tolist()])
    if return_target:
        return preds_cpu, targets_cpu
    return preds_cpu


def predict(dl, model, id2label, id2cls=None):
    model.eval()
    idx = 0
    preds_cpu = []
    preds_cpu_cls = []
    for batch in tqdm(dl, total=len(dl), leave=False, desc="Predicting"):
        idx += 1
        labels_mask, labels_ids = batch[1], batch[3]
        preds = model(batch)
        preds = preds.argmax(dim=1)
        preds = preds.view(labels_mask.shape)
        if id2cls is not None:
            preds, preds_cls = preds
            preds_cpu_ = transformed_result_cls([preds_cls], [preds_cls], id2cls, False)
            preds_cpu_cls.extend(preds_cpu_)

        preds_cpu_ = transformed_result([preds], [labels_mask], id2label)
        preds_cpu.extend(preds_cpu_)
    if id2cls is not None:
        return preds_cpu, preds_cpu_cls
    return preds_cpu


updates = load_model_state('best_model.pt', model)
dl = get_data_loader_for_predict(data, df_path='data/conll2003-de/temp.csv')

with torch.no_grad():
    preds = predict(dl, model, data.train_ds.idx2label)
    pred_tokens, pred_labels = bert_labels2tokens(dl, preds)
    true_tokens, true_labels = bert_labels2tokens(dl, [x.bert_labels for x in dl.dataset])
    # print(true_tokens, true_labels)
    assert pred_tokens == true_tokens
    tokens_report = flat_classification_report(true_labels, pred_labels,
                                               labels=data.train_ds.idx2label[4:], digits=4)
    print(tokens_report)
