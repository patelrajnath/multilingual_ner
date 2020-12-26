from torch import nn
from transformers import BertModel

from models import BaseModel, register_model, register_model_architecture
from models.embedders import BERTEmbedder
from torch.nn import functional as F

from models.attn import MultiHeadAttention


@register_model('bert_ner')
class BertNER(BaseModel):
    def __init__(self, args):
        super(BertNER, self).__init__()
        self.args = args
        # maps each token to an embedding_dim vector
        # self.embedding = nn.Embedding(params.vocab_size, params.embedding_dim)
        bert_model = BertModel.from_pretrained(args.model_name)
        self.embeddings = bert_model.get_input_embeddings()
        self.embeddings.weight.requires_grad = False
        # self.embeddings = BERTEmbedder.create(model_name=args.model_name,
        #                                       device=args.device, mode=args.mode,
        #                                       is_freeze=args.freeze_bert_weights)
        # the LSTM takens embedded sentence
        self.lstm = nn.LSTM(self.args.embedding_dim, self.args.hidden_layer_size // 2,
                            batch_first=True, bidirectional=True)
        # fc layer transforms the output to give the final output layer
        self.fc = nn.Linear(self.args.hidden_layer_size, self.args.number_of_tags)

    @classmethod
    def build_model(cls, args):
        """

        :param args:
        :return:
        """
        bert_ner_base(args)

        return cls(args)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        group = parser.add_argument_group('Model Options')
        group.add_argument('--hidden_layer_size', type=int, help='Hidden layer size.')
        group.add_argument('--num_hidden_layers', type=int, help='Number of hidden layers.')
        group.add_argument('--embedding_dim', type=int, help='Word embedding size..')
        group.add_argument('--activation', type=str, help='The activation function.')
        group.add_argument('--dropout', type=float, help='The value of the dropout.')
        group.add_argument('--model_name', type=str)
        group.add_argument('--mode', type=str,)
        group.add_argument('--freeze_bert_weights', type=str)
        return group

    def forward(self, tensor, mask=None):
        # apply the embedding layer that maps each token to its embedding
        tensor = self.embeddings(tensor)  # dim: batch_size x batch_max_len x embedding_dim

        # run the LSTM along the sentences of length batch_max_len
        tensor, _ = self.lstm(tensor)  # dim: batch_size x batch_max_len x lstm_hidden_dim

        # reshape the Variable so that each row contains one token
        tensor = tensor.reshape(-1, tensor.shape[2])  # dim: batch_size*batch_max_len x lstm_hidden_dim

        # apply the fully connected layer and obtain the output for each token
        tensor = self.fc(tensor)  # dim: batch_size*batch_max_len x num_tags

        return F.log_softmax(tensor, dim=1)  # dim: batch_size*batch_max_len x num_tags


@register_model('attn_bert_ner')
class AttnBertNER(BaseModel):
    def __init__(self, args):
        super(AttnBertNER, self).__init__()
        self.args = args
        # maps each token to an embedding_dim vector
        # self.embedding = nn.Embedding(params.vocab_size, params.embedding_dim)
        self.embeddings = BERTEmbedder.create(model_name=args.model_name,
                                              device=args.device, mode=args.mode,
                                              is_freeze=args.freeze_bert_weights)

        # the LSTM takens embedded sentence
        self.lstm = nn.LSTM(args.embedding_dim, args.hidden_layer_size // 2,
                            batch_first=True, bidirectional=True)

        self.attn = MultiHeadAttention(d_v=args.attn_dim_val,
                                       d_k=args.attn_dim_key,
                                       d_model=args.hidden_layer_size,
                                       n_heads=args.attn_num_heads,
                                       dropout=args.attn_dropout)

        # fc layer transforms the output to give the final output layer
        self.fc = nn.Linear(args.hidden_layer_size, args.number_of_tags)

    @classmethod
    def build_model(cls, args):
        """

        :param args:
        :return:
        """
        bert_ner_base(args)
        return cls(args)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        group = parser.add_argument_group('Model Options')
        group.add_argument('--hidden_layer_size', type=int, default=512,
                           help='Hidden layer size.')
        group.add_argument('--num_hidden_layers', type=int, default=1,
                           help='Number of hidden layers.')
        group.add_argument('--embedding_dim', type=int, default=256,
                           help='Word embedding size..')
        group.add_argument('--activation', type=str, default='relu',
                           help='The activation function.')
        group.add_argument('--dropout', type=float, default=0.1,
                           help='The value of the dropout.')
        group.add_argument('--model_name', type=str, default='bert-base-multilingual-cased')
        group.add_argument('--mode', type=str, default='weighted')
        group.add_argument('--freeze_bert_weights', type=str, default=True)
        group.add_argument('--attn_dropout', type=float, default=0.3,
                           help='Attn dropout.')
        group.add_argument('--attn_num_heads', type=int, default=1,
                           help='Attn heads.')
        group.add_argument('--attn_dim_val', type=int, default=64,
                           help='Attn dimension of values.')
        group.add_argument('--attn_dim_key', type=int, default=64,
                           help='Attn dimension of Keys.')
        return group

    def forward(self, tensor, mask=None):
        # apply the embedding layer that maps each token to its embedding
        tensor = self.embeddings(tensor)  # dim: batch_size x batch_max_len x embedding_dim

        # run the LSTM along the sentences of length batch_max_len
        tensor, _ = self.lstm(tensor)  # dim: batch_size x batch_max_len x lstm_hidden_dim

        # Apply attn to get better word dependencies
        tensor, _ = self.attn(tensor, tensor, tensor, mask)

        # reshape the Variable so that each row contains one token
        tensor = tensor.reshape(-1, tensor.shape[2])  # dim: batch_size*batch_max_len x lstm_hidden_dim

        # apply the fully connected layer and obtain the output for each token
        tensor = self.fc(tensor)  # dim: batch_size*batch_max_len x num_tags

        return F.log_softmax(tensor, dim=1)  # dim: batch_size*batch_max_len x num_tags


@register_model_architecture('bert_ner', 'bert_ner')
def bert_ner_base(args):
    args.hidden_layer_size = getattr(args, 'hidden_layer_size', 768)
    args.num_hidden_layers = getattr(args, 'num_hidden_layers', 1)
    args.embedding_dim = getattr(args, 'embedding_dim', 768)
    args.activation = getattr(args, 'activation', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.model_name = getattr(args, 'model_name', 'bert-base-multilingual-cased')
    args.mode = getattr(args, 'mode', 'weighted')
    args.freeze_bert_weights = getattr(args, 'freeze_bert_weights', True)


@register_model_architecture('bert_ner', 'bert_ner_medium')
def bert_ner_medium(args):
    args.hidden_layer_size = getattr(args, 'hidden_layer_size', 1024)
    args.num_hidden_layers = getattr(args, 'num_hidden_layers', 1)
    args.embedding_dim = getattr(args, 'embedding_dim', 768)
    args.activation = getattr(args, 'activation', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.model_name = getattr(args, 'model_name', 'bert-base-multilingual-cased')
    args.mode = getattr(args, 'mode', 'weighted')
    args.freeze_bert_weights = getattr(args, 'freeze_bert_weights', True)


@register_model_architecture('attn_bert_ner', 'attn_bert_ner')
def attn_bert_ner(args):
    args.attn_dropout = getattr(args, 'attn_dropout', 0.3)
    args.attn_num_heads = getattr(args, 'attn_num_heads', 1)
    args.attn_dim_val = getattr(args, 'attn_dim_val', 64)
    args.attn_dim_key = getattr(args, 'attn_dim_key', 64)
    bert_ner_base(args)


@register_model_architecture('attn_bert_ner', 'attn_bert_ner_medium')
def attn_bert_ner_medium(args):
    args.attn_dropout = getattr(args, 'attn_dropout', 0.3)
    args.attn_num_heads = getattr(args, 'attn_num_heads', 1)
    args.attn_dim_val = getattr(args, 'attn_dim_val', 64)
    args.attn_dim_key = getattr(args, 'attn_dim_key', 64)
    bert_ner_medium(args)
