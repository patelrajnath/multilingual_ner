from typing import List

import flair
import torch
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings, TransformerWordEmbeddings, StackedEmbeddings

# init Flair embeddings
flair_forward_embedding = FlairEmbeddings('multi-forward')
flair_backward_embedding = FlairEmbeddings('multi-backward')

# init multilingual BERT
bert_embedding = TransformerWordEmbeddings('bert-base-multilingual-cased')

# now create the StackedEmbedding object that combines all embeddings
embeddings = StackedEmbeddings(
    embeddings=[flair_forward_embedding, flair_backward_embedding])
# create a sentence
sentence1 = Sentence('The grass is green .')
sentence2 = Sentence('yes, The grass is green .')
sentences = [sentence1, sentence2]
# embed words in the sentence
embeddings.embed(sentences)

lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
longest_token_sequence_in_batch: int = max(lengths)

pre_allocated_zero_tensor = torch.zeros(
    embeddings.embedding_length * longest_token_sequence_in_batch,
    dtype=torch.float,
    device=flair.device,
)

all_embs: List[torch.Tensor] = list()
for sentence in sentences:
    all_embs += [
        emb for token in sentence for emb in token.get_each_embedding()
    ]
    nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)

    if nb_padding_tokens > 0:
        t = pre_allocated_zero_tensor[
            : embeddings.embedding_length * nb_padding_tokens
            ]
        all_embs.append(t)

sentence_tensor = torch.cat(all_embs).view(
    [
        len(sentences),
        longest_token_sequence_in_batch,
        embeddings.embedding_length,
    ]
)
print(sentence_tensor.size())
# embed words in sentence
# flair_embedding_forward.embed([sentence1, sentence2])
# for word in sentence1:
#     print(word.embedding)
# print()
# for word in sentence2:
#     print(word.embedding)
