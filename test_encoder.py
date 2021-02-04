import torch

from models.transformer_encoder import TransformerWordEmbeddings


class StackTransformerEmbeddings(object):
    def __init__(self, encoders):
        self.encoders = encoders

    def encode(self, segments):
        segments_encoded = [encoder.encode(segments) for encoder in self.encoders]
        segments_enc = []
        for emb in zip(*segments_encoded):
            emb_cat = torch.cat(emb, dim=-1)
            segments_enc.append(emb_cat)
        return segments_enc


bert_embedding1 = TransformerWordEmbeddings('distilbert-base-multilingual-cased',
                                            layers='-1',
                                            batch_size=2)

bert_embedding2 = TransformerWordEmbeddings('distilbert-base-multilingual-cased',
                                            layers='-1',
                                            batch_size=2)

sentences = [
    "Hello, how are you?",
    "Hello, how are you ?",
    "Hello , how are you ?",
]

encoder = StackTransformerEmbeddings([bert_embedding1, bert_embedding2])
sentences_encoded = encoder.encode(sentences)
for emb in sentences_encoded:
    print(emb.size())
