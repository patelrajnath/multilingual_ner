from models.transformer_encoder import TransformerWordEmbeddings, StackTransformerEmbeddings


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
