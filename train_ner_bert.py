import os
import time
from math import inf
import torch

from models.model_utils import set_seed, save_state, load_model_state, loss_fn, get_attn_pad_mask, \
    get_device, predict
from models import bert_data, build_model
from models.bert_data import get_data_loader_for_predict
from sklearn_crfsuite.metrics import flat_classification_report
from analyze_utils.utils import bert_labels2tokens
from models.optimization import BertAdam
from options.args_parser import get_training_options_bert, update_args_arch

set_seed(seed_value=999)


def train(args):
    device = get_device(args)
    data = bert_data.LearnData.create(
        train_df_path=os.path.join(args.data_dir, args.train),
        valid_df_path=os.path.join(args.data_dir, args.test),
        idx2labels_path=os.path.join(args.data_dir, args.idx2labels),
        clear_cache=True,
        model_name=args.model_name,
        model_type=args.model_type,
        batch_size=args.batch_size,
        device=device,
        markup='BIO',
        max_sequence_length=args.max_seq_len,
        shuffle=args.shuffle,
    )

    args.number_of_tags = len(data.train_ds.idx2label)
    model = build_model(args, device)
    model = model.to(device)
    model.train()

    betas = (0.9, 0.999)
    eps = 1e-8
    optimizer = BertAdam(model, lr=args.learning_rate, b1=betas[0], b2=betas[1], e=eps)

    pad_id = 0  # This is pad_id of BERT model
    updates = 1
    total_loss = 0
    best_loss = +inf
    stop_training = False
    output_dir = args.output_dir
    try:
        os.makedirs(output_dir)
    except:
        pass

    prefix = args.train.split('_')[0] if len(args.train.split('_')) > 1 else args.train.split('.')[0]

    start = time.time()
    for epoch in range(args.epochs):
        for batch in data.train_dl:
            updates += 1
            optimizer.zero_grad()
            input_, labels_mask, input_type_ids, labels = batch
            # Create attn mask
            attn_mask = get_attn_pad_mask(input_, input_, pad_id)
            loss = model.score(batch, attn_mask=attn_mask)

            loss.backward()
            optimizer.step()
            total_loss += loss.data

            if updates % args.patience == 0:
                print(f'Epoch: {epoch}, Updates:{updates}, Loss: {total_loss}')
                if best_loss > total_loss:
                    save_state(f'{output_dir}/{prefix}_best_model_bert.pt', model, loss_fn, optimizer,
                               updates, args=args)
                    best_loss = total_loss
                total_loss = 0

            if updates % args.max_steps == 0:
                stop_training = True
                break

        if stop_training:
            break
    print(f'Training time: {time.time() - start}')

    model, model_args = load_model_state(f'{output_dir}/{prefix}_best_model_bert.pt', device)
    model = model.to(device)
    dl = get_data_loader_for_predict(data, df_path=os.path.join(args.data_dir, args.test))

    with open(f'{output_dir}/{prefix}_label_bert.txt', 'w') as t, \
            open(f'{output_dir}/{prefix}_predict_bert.txt', 'w') as p, \
            open(f'{output_dir}/{prefix}_text_bert.txt', 'w') as textf:
        with torch.no_grad():
            preds = predict(dl, model, data.train_ds.idx2label, pad_id=pad_id)
            pred_tokens, pred_labels = bert_labels2tokens(dl, preds)
            true_tokens, true_labels = bert_labels2tokens(dl, [x.bert_labels for x in dl.dataset])
            # print(true_tokens, true_labels)
            assert pred_tokens == true_tokens
            tokens_report = flat_classification_report(true_labels, pred_labels,
                                                       labels=data.train_ds.idx2label[4:], digits=4)
            print(tokens_report)
            t.write('\n'.join([' '.join([item for item in t_label]) for t_label in true_labels]) + '\n')
            p.write('\n'.join([' '.join([item for item in p_label]) for p_label in pred_labels]) + '\n')
            textf.write('\n'.join([' '.join([item for item in t_token]) for t_token in true_tokens]) + '\n')


if __name__ == '__main__':
    parser = get_training_options_bert()
    args = parser.parse_args()
    args = update_args_arch(args)
    print(args)
    train(args)
