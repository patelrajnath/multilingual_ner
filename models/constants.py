from transformers import BertConfig, BertTokenizer, \
    DistilBertConfig, DistilBertTokenizer, XLMRobertaTokenizer, XLMRobertaConfig, \
    RobertaConfig, RobertaTokenizer, BertweetTokenizer, ElectraConfig, ElectraTokenizer

from models.modeling_bert import BertTokenEmbedder
from models.modeling_distilbert import DistilBertTokenEmbedder
from models.modeling_electra import ElectraTokenEmbedder
from models.modeling_roberta import RobertaTokenEmbedder
from models.modeling_xlm import XLMTokenEmbedder

MODEL_CLASSES = {
    "bert": (BertConfig, BertTokenEmbedder, BertTokenizer),
    "distilbert": (DistilBertConfig, DistilBertTokenEmbedder, DistilBertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMTokenEmbedder, XLMRobertaTokenizer),
    "roberta": (RobertaConfig, RobertaTokenEmbedder, RobertaTokenizer),
    "bertweet": (RobertaConfig, RobertaTokenEmbedder, BertweetTokenizer),
    "electra": (ElectraConfig, ElectraTokenEmbedder, ElectraTokenizer),
}
