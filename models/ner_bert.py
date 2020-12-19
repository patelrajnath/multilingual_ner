import torch
from torch import nn

from models.embedders import BERTEmbedder
from torch.nn import functional as F

from models.attn import MultiHeadAttention


class BertNER(nn.Module):
    def __init__(self, model_params, options, device):
        super(BertNER, self).__init__()
        self.model_params = model_params
        # maps each token to an embedding_dim vector
        # self.embedding = nn.Embedding(params.vocab_size, params.embedding_dim)
        self.embeddings = BERTEmbedder.create(model_name=options.model_name,
                                              device=device, mode=options.mode, is_freeze=options.is_freeze)

        # the LSTM takens embedded sentence
        self.lstm = nn.LSTM(self.model_params.embedding_dim, self.model_params.hidden_layer_size // 2,
                            batch_first=True, bidirectional=True)
        # fc layer transforms the output to give the final output layer
        self.fc = nn.Linear(self.model_params.hidden_layer_size, self.model_params.number_of_tags)

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


class AttnBertNER(nn.Module):
    def __init__(self, model_params, options, device):
        super(AttnBertNER, self).__init__()
        self.model_params = model_params
        # maps each token to an embedding_dim vector
        # self.embedding = nn.Embedding(params.vocab_size, params.embedding_dim)
        self.embeddings = BERTEmbedder.create(model_name=options.model_name,
                                              device=device, mode=options.mode,
                                              is_freeze=options.freeze_bert_weights)

        # the LSTM takens embedded sentence
        self.lstm = nn.LSTM(self.model_params.embedding_dim, self.model_params.hidden_layer_size // 2,
                            batch_first=True, bidirectional=True)

        self.attn = MultiHeadAttention(d_v=options.attn_dim_val,
                                       d_k=options.attn_dim_key,
                                       d_model=self.model_params.hidden_layer_size,
                                       n_heads=options.attn_num_heads,
                                       dropout=options.attn_dropout)

        # fc layer transforms the output to give the final output layer
        self.fc = nn.Linear(self.model_params.hidden_layer_size, self.model_params.number_of_tags)

    def forward(self, s):
        # apply the embedding layer that maps each token to its embedding
        s = self.embeddings(s)  # dim: batch_size x batch_max_len x embedding_dim

        # run the LSTM along the sentences of length batch_max_len
        s, _ = self.lstm(s)  # dim: batch_size x batch_max_len x lstm_hidden_dim

        # Apply attn to get better word dependencies
        s = self.attn(s)

        # reshape the Variable so that each row contains one token
        s = s.reshape(-1, s.shape[2])  # dim: batch_size*batch_max_len x lstm_hidden_dim

        # apply the fully connected layer and obtain the output for each token
        s = self.fc(s)  # dim: batch_size*batch_max_len x num_tags

        return F.log_softmax(s, dim=1)  # dim: batch_size*batch_max_len x num_tags
