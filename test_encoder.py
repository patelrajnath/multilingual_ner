import os
import time
from math import inf

import torch

from datautils.batchers import SamplingBatcherStackedTransformers
from datautils.prepare_data import prepare_text
from datautils.vocab import load_vocabs
from models import build_model
from models.model_utils import get_device, loss_fn, save_state
from models.transformer_encoder import TransformerWordEmbeddings, StackTransformerEmbeddings
import numpy as np

from options.args_parser import get_training_options, update_args_arch


def train(args):
    vocab_path = os.path.join(args.data_dir, args.vocab)
    tag_path = os.path.join(args.data_dir, args.tag_set)
    word_to_idx, idx_to_word, tag_to_idx, idx_to_tag = load_vocabs(vocab_path, tag_path)
    train_sentences, train_labels, test_sentences, test_labels = prepare_text(args, tag_to_idx)

    device = get_device(args)
    bert_embedding1 = TransformerWordEmbeddings('distilbert-base-multilingual-cased',
                                                layers='-1',
                                                batch_size=args.batch_size)

    bert_embedding2 = TransformerWordEmbeddings('distilbert-base-multilingual-cased',
                                                layers='-1',
                                                batch_size=args.batch_size)

    encoder = StackTransformerEmbeddings([bert_embedding1, bert_embedding2])

    train_sentences_encoded = encoder.encode(train_sentences)
    test_sentences_encoded = encoder.encode(train_sentences)

    # Update the Namespace
    args.vocab_size = len(idx_to_word)
    args.number_of_tags = len(idx_to_tag)

    model = build_model(args, device)
    print(model)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters())

    pad_id = word_to_idx['PAD']
    pad_id_labels = tag_to_idx['PAD']

    batcher = SamplingBatcherStackedTransformers(np.asarray(train_sentences_encoded, dtype=object),
                                                 np.asarray(train_labels, dtype=object),
                                                 batch_size=args.batch_size,
                                                 pad_id=pad_id,
                                                 pad_id_labels=pad_id_labels,
                                                 embedding_length=encoder.embedding_length
                                                 )

    updates = 1
    total_loss = 0
    best_loss = +inf
    stop_training = False
    output_dir = args.output_dir
    try:
        os.makedirs(output_dir)
    except:
        pass

    prefix = args.train_text.split('_')[0] if len(args.train_text.split('_')) > 1 \
        else args.train_text.split('.')[0]

    start_time = time.time()
    for epoch in range(args.epochs):
        for batch in batcher:
            updates += 1
            input_, labels, labels_mask = batch
            optimizer.zero_grad()
            loss = model.score(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.data
            if updates % args.patience == 0:
                print(f'Epoch: {epoch}, Updates:{updates}, Loss: {total_loss}')
                if best_loss > total_loss:
                    save_state(f'{output_dir}/{prefix}_best_model.pt', model, loss_fn, optimizer,
                               updates, args=args)
                    best_loss = total_loss
                total_loss = 0
            if updates % args.max_steps == 0:
                stop_training = True
                break

        if stop_training:
            break

    print('Training time:{}'.format(time.time() - start_time))


if __name__ == '__main__':
    parser = get_training_options()
    args = parser.parse_args()
    args = update_args_arch(args)
    train(args)
