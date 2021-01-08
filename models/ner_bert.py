from torch import nn

from models import BaseModel, register_model, register_model_architecture
from models.embedders import PretrainedEmbedder
from torch.nn import functional as F

from models.attn import MultiHeadAttention
from models.layers.decoders import CRFDecoder
from models.model_utils import loss_fn, get_device


@register_model('bert_ner')
class BertNER(BaseModel):
    def __init__(self, args):
        super(BertNER, self).__init__()
        self.args = args
        self.device = get_device(args)

        if self.args.use_projection:
            self.lstm_input_dim = self.args.projection_dim
        else:
            self.lstm_input_dim = self.args.embedding_dim

        # maps each token to an embedding_dim vector
        self.embeddings = PretrainedEmbedder(self.args, device=self.device)

        # the LSTM takens embedded sentence
        self.bert_projection = nn.Linear(self.args.embedding_dim, self.args.projection_dim)

        self.lstm = nn.LSTM(self.lstm_input_dim, self.args.hidden_layer_size // 2,
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
        group.add_argument('--projection_dim', type=int, help='Bert embedding projection dimension.')
        group.add_argument('--activation', type=str, help='The activation function.')
        group.add_argument('--dropout', type=float, help='The value of the dropout.')
        group.add_argument('--model_name', type=str)
        group.add_argument('--model_type', type=str)
        group.add_argument('--mode', type=str)
        group.add_argument('--freeze_bert_weights', type=bool)
        group.add_argument('--only_embedding', action='store_true')
        group.add_argument('--use_projection', action='store_true')
        group.add_argument('--onnx', action='store_true')
        return group

    def get_logits(self, input_, attn_mask=None):
        input_ = self._get_inputs_dict(input_)

        # apply the embedding layer that maps each token to its embedding
        tensor = self.embeddings(input_)  # dim: batch_size x batch_max_len x embedding_dim

        if self.args.use_projection:
            # Project the bert embeddings to lower dimensions
            tensor = self.bert_projection(tensor)  # dim: batch_size x batch_max_len x bert_projection

        # run the LSTM along the sentences of length batch_max_len
        tensor, _ = self.lstm(tensor)  # dim: batch_size x batch_max_len x lstm_hidden_dim

        # reshape the Variable so that each row contains one token
        tensor = tensor.reshape(-1, tensor.shape[2])  # dim: batch_size*batch_max_len x lstm_hidden_dim

        # apply the fully connected layer and obtain the output for each token
        tensor = self.fc(tensor)  # dim: batch_size*batch_max_len x num_tags

        return F.log_softmax(tensor, dim=1)  # dim: batch_size*batch_max_len x num_tags

    def forward(self, batch, attn_mask=None):
        logits = self.get_logits(batch, attn_mask)
        return logits.argmax(dim=1)

    def score(self, batch, attn_mask=None):
        input_, labels_mask, input_type_ids, labels = batch
        logits = self.get_logits(batch, attn_mask)
        labels = labels.view(-1).to(self.device)
        labels_mask = labels_mask.view(-1).to(self.device)
        return loss_fn(logits, labels, labels_mask)


@register_model('bert_crf_ner')
class BertCRFNER(BaseModel):
    def __init__(self, args):
        super(BertCRFNER, self).__init__()
        self.args = args
        self.device = get_device(args)

        if self.args.use_projection:
            self.lstm_input_dim = self.args.projection_dim
        else:
            self.lstm_input_dim = self.args.embedding_dim

        # maps each token to an embedding_dim vector
        self.embeddings = PretrainedEmbedder(self.args, device=self.device)

        # the LSTM takens embedded sentence
        self.bert_projection = nn.Linear(self.args.embedding_dim, self.args.projection_dim)

        self.lstm = nn.LSTM(self.lstm_input_dim, self.args.hidden_layer_size // 2,
                            batch_first=True, bidirectional=True)
        # CRF layer transforms the output to give the final output layer
        self.crf = CRFDecoder.create(self.args.number_of_tags, self.args.hidden_layer_size, self.device)

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
        group.add_argument('--projection_dim', type=int, help='Bert embedding projection dimension.')
        group.add_argument('--activation', type=str, help='The activation function.')
        group.add_argument('--dropout', type=float, help='The value of the dropout.')
        group.add_argument('--model_name', type=str)
        group.add_argument('--model_type', type=str)
        group.add_argument('--mode', type=str)
        group.add_argument('--freeze_bert_weights', type=bool)
        group.add_argument('--only_embedding', action='store_true')
        group.add_argument('--use_projection', action='store_true')

        return group

    def lstm_output(self, input_, attn_mask=None):
        input_ = self._get_inputs_dict(input_)

        # apply the embedding layer that maps each token to its embedding
        tensor = self.embeddings(input_)  # dim: batch_size x batch_max_len x embedding_dim

        if self.args.use_projection:
            # Project the bert embeddings to lower dimensions
            tensor = self.bert_projection(tensor)  # dim: batch_size x batch_max_len x bert_projection

        # run the LSTM along the sentences of length batch_max_len
        tensor, _ = self.lstm(tensor)  # dim: batch_size x batch_max_len x lstm_hidden_dim

        return tensor

    def forward(self, batch, attn_mask=None):
        input_, labels_mask, input_type_ids, labels = batch
        tensor = self.lstm_output(batch, attn_mask)
        return self.crf.forward(tensor, labels_mask)

    def score(self, batch, attn_mask=None):
        input_, labels_mask, input_type_ids, labels = batch
        tensor = self.lstm_output(batch, attn_mask)
        return self.crf.score(tensor, labels_mask, labels)


@register_model('attn_bert_ner')
class AttnBertNER(BaseModel):
    def __init__(self, args):
        super(AttnBertNER, self).__init__()
        self.args = args
        self.device = get_device(args)

        if self.args.use_projection:
            self.lstm_input_dim = self.args.projection_dim
        else:
            self.lstm_input_dim = self.args.embedding_dim

        # maps each token to an embedding_dim vector
        self.embeddings = PretrainedEmbedder(self.args, device=self.device)

        self.bert_projection = nn.Linear(self.args.embedding_dim, self.args.projection_dim)

        # the LSTM takens embedded sentence
        self.lstm = nn.LSTM(self.lstm_input_dim, args.hidden_layer_size // 2,
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
        group.add_argument('--hidden_layer_size', type=int,
                           help='Hidden layer size.')
        group.add_argument('--num_hidden_layers', type=int,
                           help='Number of hidden layers.')
        group.add_argument('--embedding_dim', type=int,
                           help='Word embedding size..')
        group.add_argument('--projection_dim', type=int,
                           help='Bert embedding projection dimension.')
        group.add_argument('--activation', type=str,
                           help='The activation function.')
        group.add_argument('--dropout', type=float,
                           help='The value of the dropout.')
        group.add_argument('--model_name', type=str)
        group.add_argument('--model_type', type=str)
        group.add_argument('--mode', type=str)
        group.add_argument('--freeze_bert_weights', action='store_true')
        group.add_argument('--only_embedding', action='store_true')
        group.add_argument('--use_projection', action='store_true')

        group.add_argument('--attn_dropout', type=float,
                           help='Attn dropout.')
        group.add_argument('--attn_num_heads', type=int,
                           help='Attn heads.')
        group.add_argument('--attn_dim_val', type=int,
                           help='Attn dimension of values.')
        group.add_argument('--attn_dim_key', type=int,
                           help='Attn dimension of Keys.')
        return group

    def get_logits(self, input_, attn_mask=None):
        input_ = self._get_inputs_dict(input_)

        # apply the embedding layer that maps each token to its embedding
        tensor = self.embeddings(input_)  # dim: batch_size x batch_max_len x embedding_dim

        if self.args.use_projection:
            # Project the bert embeddings to lower dimensions
            tensor = self.bert_projection(tensor)  # dim: batch_size x batch_max_len x bert_projection

        # run the LSTM along the sentences of length batch_max_len
        tensor, _ = self.lstm(tensor)  # dim: batch_size x batch_max_len x lstm_hidden_dim

        # Apply attn to get better word dependencies
        tensor, _ = self.attn(tensor, tensor, tensor, attn_mask)

        # reshape the Variable so that each row contains one token
        tensor = tensor.reshape(-1, tensor.shape[2])  # dim: batch_size*batch_max_len x lstm_hidden_dim

        # apply the fully connected layer and obtain the output for each token
        tensor = self.fc(tensor)  # dim: batch_size*batch_max_len x num_tags

        return F.log_softmax(tensor, dim=1)  # dim: batch_size*batch_max_len x num_tags

    def forward(self, batch, attn_mask=None):
        logits = self.get_logits(batch, attn_mask)
        return logits.argmax(dim=1)

    def score(self, batch, attn_mask=None):
        input_, labels_mask, input_type_ids, labels = batch
        logits = self.get_logits(batch, attn_mask)
        labels = labels.view(-1).to(self.device)
        labels_mask = labels_mask.view(-1).to(self.device)
        return loss_fn(logits, labels, labels_mask)


@register_model_architecture('bert_crf_ner', 'bert_crf_ner_tiny')
def bert_crf_ner_tiny(args):
    args.hidden_layer_size = getattr(args, 'hidden_layer_size', 128)
    args.num_hidden_layers = getattr(args, 'num_hidden_layers', 1)
    args.embedding_dim = getattr(args, 'embedding_dim', 768)
    args.projection_dim = getattr(args, 'projection_dim', 64)
    args.activation = getattr(args, 'activation', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.model_name = getattr(args, 'model_name', 'distilbert-base-multilingual-cased')
    args.model_type = getattr(args, 'model_type', 'distilbert')
    args.mode = getattr(args, 'mode', 'weighted')
    args.freeze_bert_weights = getattr(args, 'freeze_bert_weights', True)
    args.only_embedding = getattr(args, 'only_embedding', False)
    args.use_projection = getattr(args, 'use_projection', False)
    args.onnx = getattr(args, 'onnx', False)


@register_model_architecture('bert_crf_ner', 'bert_crf_ner_small')
def bert_ner_small(args):
    args.hidden_layer_size = getattr(args, 'hidden_layer_size', 256)
    args.num_hidden_layers = getattr(args, 'num_hidden_layers', 1)
    args.embedding_dim = getattr(args, 'embedding_dim', 768)
    args.projection_dim = getattr(args, 'projection_dim', 128)
    args.activation = getattr(args, 'activation', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.model_name = getattr(args, 'model_name', 'distilbert-base-multilingual-cased')
    args.model_type = getattr(args, 'model_type', 'distilbert')
    args.mode = getattr(args, 'mode', 'weighted')
    args.freeze_bert_weights = getattr(args, 'freeze_bert_weights', True)
    args.only_embedding = getattr(args, 'only_embedding', False)
    args.use_projection = getattr(args, 'use_projection', False)
    args.onnx = getattr(args, 'onnx', False)


@register_model_architecture('bert_crf_ner', 'bert_crf_ner')
def bert_ner_base(args):
    args.hidden_layer_size = getattr(args, 'hidden_layer_size', 512)
    args.num_hidden_layers = getattr(args, 'num_hidden_layers', 1)
    args.embedding_dim = getattr(args, 'embedding_dim', 768)
    args.projection_dim = getattr(args, 'projection_dim', 256)
    args.activation = getattr(args, 'activation', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.model_name = getattr(args, 'model_name', 'distilbert-base-multilingual-cased')
    args.model_type = getattr(args, 'model_type', 'distilbert')
    args.mode = getattr(args, 'mode', 'weighted')
    args.freeze_bert_weights = getattr(args, 'freeze_bert_weights', True)
    args.only_embedding = getattr(args, 'only_embedding', False)
    args.use_projection = getattr(args, 'use_projection', False)
    args.onnx = getattr(args, 'onnx', False)


@register_model_architecture('bert_crf_ner', 'bert_crf_ner_medium')
def bert_ner_base(args):
    args.hidden_layer_size = getattr(args, 'hidden_layer_size', 768)
    args.num_hidden_layers = getattr(args, 'num_hidden_layers', 1)
    args.embedding_dim = getattr(args, 'embedding_dim', 768)
    args.projection_dim = getattr(args, 'projection_dim', 512)
    args.activation = getattr(args, 'activation', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.model_name = getattr(args, 'model_name', 'distilbert-base-multilingual-cased')
    args.model_type = getattr(args, 'model_type', 'distilbert')
    args.mode = getattr(args, 'mode', 'weighted')
    args.freeze_bert_weights = getattr(args, 'freeze_bert_weights', True)
    args.only_embedding = getattr(args, 'only_embedding', False)
    args.use_projection = getattr(args, 'use_projection', False)
    args.onnx = getattr(args, 'onnx', False)


@register_model_architecture('bert_ner', 'bert_ner_tiny')
def bert_ner_tiny(args):
    args.hidden_layer_size = getattr(args, 'hidden_layer_size', 128)
    args.num_hidden_layers = getattr(args, 'num_hidden_layers', 1)
    args.embedding_dim = getattr(args, 'embedding_dim', 768)
    args.projection_dim = getattr(args, 'projection_dim', 64)
    args.activation = getattr(args, 'activation', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.model_name = getattr(args, 'model_name', 'distilbert-base-multilingual-cased')
    args.model_type = getattr(args, 'model_type', 'distilbert')
    args.mode = getattr(args, 'mode', 'weighted')
    args.freeze_bert_weights = getattr(args, 'freeze_bert_weights', True)
    args.only_embedding = getattr(args, 'only_embedding', False)
    args.use_projection = getattr(args, 'use_projection', False)
    args.onnx = getattr(args, 'onnx', False)


@register_model_architecture('bert_ner', 'bert_ner_small')
def bert_ner_small(args):
    args.hidden_layer_size = getattr(args, 'hidden_layer_size', 256)
    args.num_hidden_layers = getattr(args, 'num_hidden_layers', 1)
    args.embedding_dim = getattr(args, 'embedding_dim', 768)
    args.projection_dim = getattr(args, 'projection_dim', 128)
    args.activation = getattr(args, 'activation', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.model_name = getattr(args, 'model_name', 'distilbert-base-multilingual-cased')
    args.model_type = getattr(args, 'model_type', 'distilbert')
    args.mode = getattr(args, 'mode', 'weighted')
    args.freeze_bert_weights = getattr(args, 'freeze_bert_weights', True)
    args.only_embedding = getattr(args, 'only_embedding', False)
    args.use_projection = getattr(args, 'use_projection', False)
    args.onnx = getattr(args, 'onnx', False)


@register_model_architecture('bert_ner', 'bert_ner')
def bert_ner_base(args):
    args.hidden_layer_size = getattr(args, 'hidden_layer_size', 512)
    args.num_hidden_layers = getattr(args, 'num_hidden_layers', 1)
    args.embedding_dim = getattr(args, 'embedding_dim', 768)
    args.projection_dim = getattr(args, 'projection_dim', 256)
    args.activation = getattr(args, 'activation', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.model_name = getattr(args, 'model_name', 'distilbert-base-multilingual-cased')
    args.model_type = getattr(args, 'model_type', 'distilbert')
    args.mode = getattr(args, 'mode', 'weighted')
    args.freeze_bert_weights = getattr(args, 'freeze_bert_weights', True)
    args.only_embedding = getattr(args, 'only_embedding', False)
    args.use_projection = getattr(args, 'use_projection', False)
    args.onnx = getattr(args, 'onnx', False)


@register_model_architecture('bert_ner', 'bert_ner_medium')
def bert_ner_medium(args):
    args.hidden_layer_size = getattr(args, 'hidden_layer_size', 1024)
    args.num_hidden_layers = getattr(args, 'num_hidden_layers', 1)
    args.embedding_dim = getattr(args, 'embedding_dim', 768)
    args.projection_dim = getattr(args, 'projection_dim', 768)
    args.activation = getattr(args, 'activation', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.model_name = getattr(args, 'model_name', 'distilbert-base-multilingual-cased')
    args.model_type = getattr(args, 'model_type', 'distilbert')
    args.model_type = getattr(args, '', 'distilbert')
    args.mode = getattr(args, 'mode', 'weighted')
    args.freeze_bert_weights = getattr(args, 'freeze_bert_weights', True)
    args.only_embedding = getattr(args, 'only_embedding', False)
    args.use_projection = getattr(args, 'use_projection', False)
    args.onnx = getattr(args, 'onnx', False)


@register_model_architecture('attn_bert_ner', 'attn_bert_ner_tiny')
def attn_bert_ner(args):
    args.attn_dropout = getattr(args, 'attn_dropout', 0.3)
    args.attn_num_heads = getattr(args, 'attn_num_heads', 1)
    args.attn_dim_val = getattr(args, 'attn_dim_val', 64)
    args.attn_dim_key = getattr(args, 'attn_dim_key', 64)
    bert_ner_tiny(args)


@register_model_architecture('attn_bert_ner', 'attn_bert_ner_small')
def attn_bert_ner(args):
    args.attn_dropout = getattr(args, 'attn_dropout', 0.3)
    args.attn_num_heads = getattr(args, 'attn_num_heads', 1)
    args.attn_dim_val = getattr(args, 'attn_dim_val', 64)
    args.attn_dim_key = getattr(args, 'attn_dim_key', 64)
    bert_ner_small(args)


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
