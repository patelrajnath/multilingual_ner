from transformers import BertModel, DistilBertModel, BertConfig, BertTokenizer, \
    DistilBertConfig, DistilBertTokenizer

MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer),
    "distilbert": (DistilBertConfig, DistilBertModel, DistilBertTokenizer)
}
