import torch.nn as nn
import torch.nn.functional as F

from models import register_model_architecture, register_model, BaseModel
from models.attn import MultiHeadAttention


def get_attn_pad_mask(seq_q, seq_k):
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    b_size, len_q = seq_q.size()
    b_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(data_utils.PAD).unsqueeze(1)  # b_size x 1 x len_k
    return pad_attn_mask.expand(b_size, len_q, len_k)  # b_size x len_q x len_k


def get_attn_subsequent_mask(seq):
    assert seq.dim() == 2
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()

    return subsequent_mask


@register_model('ner')
class BasicNER(BaseModel):
    def __init__(self, args):
        super(BasicNER, self).__init__()
        self.params = args
        # maps each token to an embedding_dim vector
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim)
        # self.embedding = Embeddings(params.vocab_size, params.embedding_dim)

        # the LSTM takens embedded sentence
        self.lstm = nn.LSTM(self.params.embedding_dim, self.params.hidden_layer_size // 2,
                            batch_first=True, bidirectional=True)
        # fc layer transforms the output to give the final output layer
        self.fc = nn.Linear(self.params.hidden_layer_size, self.params.number_of_tags)

    @classmethod
    def build_model(cls, args):
        """
        :param self:
        :param args:
        :param options:
        :return:
        """
        "Helper: Construct a model from hyperparameters."

        # make sure all arguments are present in older models
        ner_base(args)

        return cls(args)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        group = parser.add_argument_group('Model Options')
        group.add_argument('--hidden_layer_size', type=int,
                            help='Hidden layer size.')
        group.add_argument('--num_hidden_layers', type=int,
                            help='Number of hidden layers.')
        group.add_argument('--embedding_dim', type=int,
                            help='Word embedding size..')
        group.add_argument('--activation', type=str,
                            help='The activation function.')
        group.add_argument('--dropout', type=float,
                            help='The value of the dropout.')
        return group

    def forward(self, tensor, mask=None):
        # apply the embedding layer that maps each token to its embedding
        tensor = self.embedding(tensor)  # dim: batch_size x batch_max_len x embedding_dim

        # run the LSTM along the sentences of length batch_max_len
        tensor, _ = self.lstm(tensor)  # dim: batch_size x batch_max_len x lstm_hidden_dim

        # reshape the Variable so that each row contains one token
        tensor = tensor.reshape(-1, tensor.shape[2])  # dim: batch_size*batch_max_len x lstm_hidden_dim

        # apply the fully connected layer and obtain the output for each token
        tensor = self.fc(tensor)  # dim: batch_size*batch_max_len x num_tags

        return F.log_softmax(tensor, dim=1)  # dim: batch_size*batch_max_len x num_tags


@register_model('attn_ner')
class AttnNER(BaseModel):
    def __init__(self, args):
        super(AttnNER, self).__init__()
        self.args = args
        # maps each token to an embedding_dim vector
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim)
        # self.embedding = Embeddings(params.vocab_size, params.embedding_dim)

        # the LSTM takens embedded sentence
        self.lstm = nn.LSTM(self.args.embedding_dim, self.args.hidden_layer_size // 2,
                            batch_first=True, bidirectional=True)
        self.attn = MultiHeadAttention(d_v=args.attn_dim_val,
                                       d_k=args.attn_dim_key,
                                       d_model=self.args.hidden_layer_size,
                                       n_heads=args.attn_num_heads,
                                       dropout=args.attn_dropout)

        # fc layer transforms the output to give the final output layer
        self.fc = nn.Linear(self.args.hidden_layer_size, self.args.number_of_tags)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        group = parser.add_argument_group('Model Options')
        group.add_argument('--hidden_layer_size', type=int,
                            help='Hidden layer size.')
        group.add_argument('--num_hidden_layers', type=int,
                            help='Number of hidden layers.')
        group.add_argument('--embedding_dim', type=int,
                            help='Word embedding size..')
        group.add_argument('--activation', type=str,
                            help='The activation function.')
        group.add_argument('--dropout', type=float,
                            help='The value of the dropout.')
        group.add_argument('--attn_dropout', type=float,
                            help='Attn dropout.')
        group.add_argument('--attn_num_heads', type=int,
                            help='Attn heads.')
        group.add_argument('--attn_dim_val', type=int,
                            help='Attn dimension of values.')
        group.add_argument('--attn_dim_key', type=int,
                            help='Attn dimension of Keys.')
        return group

    @classmethod
    def build_model(cls, args):
        """
        :param args:
        :return:
        """
        "Helper: Construct a model from hyperparameters."
        # make sure all arguments are present in older models
        attn_ner_base(args)

        return cls(args)

    def forward(self, tensor, mask=None):
        # apply the embedding layer that maps each token to its embedding
        tensor = self.embedding(tensor)  # dim: batch_size x batch_max_len x embedding_dim

        # run the LSTM along the sentences of length batch_max_len
        tensor, _ = self.lstm(tensor)  # dim: batch_size x batch_max_len x lstm_hidden_dim

        # Apply attn to get better word dependencies
        tensor, _ = self.attn(tensor, tensor, tensor, mask)

        # reshape the Variable so that each row contains one token
        tensor = tensor.reshape(-1, tensor.shape[2])  # dim: batch_size*batch_max_len x lstm_hidden_dim

        # apply the fully connected layer and obtain the output for each token
        tensor = self.fc(tensor)  # dim: batch_size*batch_max_len x num_tags

        return F.log_softmax(tensor, dim=1)  # dim: batch_size*batch_max_len x num_tags


@register_model_architecture('ner', 'ner_tiny')
def ner_tiny_architecture(args):
    args.hidden_layer_size = getattr(args, 'hidden_layer_size', 128)
    args.num_hidden_layers = getattr(args, 'num_hidden_layers', 1)
    args.embedding_dim = getattr(args, 'embedding_dim', 64)
    args.activation = getattr(args, 'activation', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)


@register_model_architecture('ner', 'ner_small')
def ner_small(args):
    args.hidden_layer_size = getattr(args, 'hidden_layer_size', 256)
    args.num_hidden_layers = getattr(args, 'num_hidden_layers', 1)
    args.embedding_dim = getattr(args, 'embedding_dim', 128)
    args.activation = getattr(args, 'activation', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)


@register_model_architecture('ner', 'ner')
def ner_base(args):
    args.hidden_layer_size = getattr(args, 'hidden_layer_size', 512)
    args.num_hidden_layers = getattr(args, 'num_hidden_layers', 1)
    args.embedding_dim = getattr(args, 'embedding_dim', 256)
    args.activation = getattr(args, 'activation', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)


@register_model_architecture('ner', 'ner_medium')
def ner_medium(args):
    args.hidden_layer_size = getattr(args, 'hidden_layer_size', 768)
    args.num_hidden_layers = getattr(args, 'num_hidden_layers', 2)
    args.embedding_dim = getattr(args, 'embedding_dim', 512)
    args.activation = getattr(args, 'activation', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)


@register_model_architecture('attn_ner', 'attn_ner_tiny')
def attn_ner_tiny(args):
    args.attn_dropout = getattr(args, 'attn_dropout', 0.3)
    args.attn_num_heads = getattr(args, 'attn_num_heads', 1)
    args.attn_dim_val = getattr(args, 'attn_dim_val', 64)
    args.attn_dim_key = getattr(args, 'attn_dim_key', 64)
    ner_tiny_architecture(args)


@register_model_architecture('attn_ner', 'attn_ner_small')
def attn_ner_small(args):
    args.attn_dropout = getattr(args, 'attn_dropout', 0.3)
    args.attn_num_heads = getattr(args, 'attn_num_heads', 1)
    args.attn_dim_val = getattr(args, 'attn_dim_val', 64)
    args.attn_dim_key = getattr(args, 'attn_dim_key', 64)
    ner_small(args)


@register_model_architecture('attn_ner', 'attn_ner')
def attn_ner_base(args):
    args.attn_dropout = getattr(args, 'attn_dropout', 0.3)
    args.attn_num_heads = getattr(args, 'attn_num_heads', 1)
    args.attn_dim_val = getattr(args, 'attn_dim_val', 64)
    args.attn_dim_key = getattr(args, 'attn_dim_key', 64)
    ner_base(args)


@register_model_architecture('attn_ner', 'attn_ner_medium')
def attn_ner_medium(args):
    args.attn_dropout = getattr(args, 'attn_dropout', 0.3)
    args.attn_num_heads = getattr(args, 'attn_num_heads', 3)
    args.attn_dim_val = getattr(args, 'attn_dim_val', 128)
    args.attn_dim_key = getattr(args, 'attn_dim_key', 128)
    ner_medium(args)
