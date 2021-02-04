from datautils.batchers import SamplingBatcherStackedTransformers
from models.transformer_encoder import TransformerWordEmbeddings, StackTransformerEmbeddings
import numpy as np

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

labels = [range(len(s.split())) for s in sentences]

encoder = StackTransformerEmbeddings([bert_embedding1, bert_embedding2])
sentences_encoded = encoder.encode(sentences)
for emb in sentences_encoded:
    print(emb.size())
print(encoder.embedding_length)

batcher = SamplingBatcherStackedTransformers(np.asarray(sentences_encoded, dtype=object),
                                             np.asarray(labels, dtype=object),
                                             batch_size=2,
                                             pad_id=0,
                                             pad_id_labels=3,
                                             embedding_length=encoder.embedding_length
                                             )

for batch in batcher:
    input, lable, label_mask = batch
    print(input.size())
    print(lable.size())
    print(label_mask.size())
