import os
from typing import List

import flair
import torch
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings, TransformerWordEmbeddings, StackedEmbeddings
import numpy as np

# init Flair embeddings
from datautils.batchers import SamplingBatcherFlair
from datautils.prepare_data import prepare_text
from datautils.vocab import load_vocabs
from models import build_model
from models.model_utils import get_device
from options.args_parser import get_training_options, update_args_arch


def train(args):
    vocab_path = os.path.join(args.data_dir, args.vocab)
    tag_path = os.path.join(args.data_dir, args.tag_set)
    word_to_idx, idx_to_word, tag_to_idx, idx_to_tag = load_vocabs(vocab_path, tag_path)
    train_sentences, train_labels, test_sentences, test_labels = prepare_text(args, tag_to_idx)

    flair_forward_embedding = FlairEmbeddings('multi-forward')
    flair_backward_embedding = FlairEmbeddings('multi-backward')

    # init multilingual BERT
    bert_embedding = TransformerWordEmbeddings('bert-base-multilingual-cased')

    # now create the StackedEmbedding object that combines all embeddings
    embeddings = StackedEmbeddings(
        embeddings=[flair_forward_embedding, flair_backward_embedding])
    # embed words in the sentence
    embeddings.embed(train_sentences)

    # Update the Namespace
    args.vocab_size = len(idx_to_word)
    args.number_of_tags = len(idx_to_tag)

    device = get_device(args)
    model = build_model(args, device)
    print(model)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters())

    pad_id = word_to_idx['PAD']
    pad_id_labels = tag_to_idx['PAD']

    batcher = SamplingBatcherFlair(np.asarray(train_sentences, dtype=object),
                                   np.asarray(train_labels, dtype=object),
                                   batch_size=args.batch_size,
                                   pad_id=pad_id,
                                   pad_id_labels=pad_id_labels,
                                   embedding_length=embeddings.embedding_length
                                   )

    for epoch in range(args.epochs):
        for batch in batcher:
            input_, labels, labels_mask = batch
            print(input_, labels, labels_mask)
            exit()
            optimizer.zero_grad()
            loss = model.score(batch)
            loss.backward()
            optimizer.step()
            print(loss)


if __name__ == '__main__':
    parser = get_training_options()
    args = parser.parse_args()
    args = update_args_arch(args)
    train(args)
