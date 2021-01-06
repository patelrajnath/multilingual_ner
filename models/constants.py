from transformers import BertConfig, BertTokenizer, \
    DistilBertConfig, DistilBertTokenizer

from models.modeling_bert import BertTokenEmbedder
from models.modeling_distilbert import DistilBertTokenEmbedder

MODEL_CLASSES = {
    "bert": (BertConfig, BertTokenEmbedder, BertTokenizer),
    "distilbert": (DistilBertConfig, DistilBertTokenEmbedder, DistilBertTokenizer)
}
