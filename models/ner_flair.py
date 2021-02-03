import torch.nn as nn
import torch.nn.functional as F

from models import register_model_architecture, register_model, BaseModel
from models.model_utils import loss_fn


@register_model('flair_ner')
class FlairNER(BaseModel):
    def __init__(self, args, device):
        super(FlairNER, self).__init__()
        self.params = args
        self.device = device

        # the LSTM takens embedded sentence
        self.lstm = nn.LSTM(self.params.embedding_dim, self.params.hidden_layer_size // 2,
                            batch_first=True, bidirectional=True)
        # fc layer transforms the output to give the final output layer
        self.fc = nn.Linear(self.params.hidden_layer_size, self.params.number_of_tags)

    @classmethod
    def build_model(cls, args, device):
        """
        :param device:
        :param args:
        :return:
        """
        "Helper: Construct a model from hyperparameters."

        # make sure all arguments are present in older models
        ner_base(args)

        return cls(args, device)

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

    def get_logits(self, input_, attn_mask=None):

        # run the LSTM along the sentences of length batch_max_len
        tensor, _ = self.lstm(input_)  # dim: batch_size x batch_max_len x lstm_hidden_dim

        # reshape the Variable so that each row contains one token
        tensor = tensor.reshape(-1, tensor.shape[2])  # dim: batch_size*batch_max_len x lstm_hidden_dim

        # apply the fully connected layer and obtain the output for each token
        tensor = self.fc(tensor)  # dim: batch_size*batch_max_len x num_tags

        return F.log_softmax(tensor, dim=1)  # dim: batch_size*batch_max_len x num_tags

    def score(self, batch, attn_mask=None):
        input_, labels, labels_mask = batch
        input_ = input_.to(self.device)
        logits = self.get_logits(input_)
        labels = labels.view(-1).to(self.device)
        labels_mask = labels_mask.view(-1).to(self.device)
        return loss_fn(logits, labels, labels_mask)

    def forward(self, batch, attn_mask=None):
        input_, labels, labels_mask = batch
        input_ = input_.to(self.device)
        logits = self.get_logits(input_)
        return logits.argmax(dim=1)


@register_model_architecture('flair_ner', 'flair_ner')
def ner_base(args):
    args.hidden_layer_size = getattr(args, 'hidden_layer_size', 512)
    args.num_hidden_layers = getattr(args, 'num_hidden_layers', 1)
    args.embedding_dim = getattr(args, 'embedding_dim', 4096)
    args.activation = getattr(args, 'activation', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
