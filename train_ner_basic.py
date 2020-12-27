import os
import time
from math import inf
import numpy as np
import torch
from datautils.batchers import SamplingBatcher
from datautils import Doc
from datautils.biluo_from_predictions import get_biluo
from datautils.iob_utils import offset_from_biluo
from datautils.vocab import load_vocabs
from models import build_model, tqdm
from models.model_utils import save_state, load_model_state, set_seed, loss_fn, get_attn_pad_mask, \
    transformed_result_cls, transformed_result

# Set seed to have consistent results
from models.ner import BasicNER, AttnNER
from models.optimization import BertAdam
from options.args_parser import get_training_options, update_args_arch
from datautils.prepare_data import prepare

set_seed(seed_value=999)
np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)


def train(args):
    vocab_path = os.path.join(args.data_dir, args.vocab)
    tag_path = os.path.join(args.data_dir, args.tag_set)
    word_to_idx, idx_to_word, tag_to_idx, idx_to_tag = load_vocabs(vocab_path, tag_path)
    train_sentences, train_labels, test_sentences, test_labels = prepare(args, word_to_idx, tag_to_idx)

    args.vocab_size = len(idx_to_word)
    args.number_of_tags = len(idx_to_tag)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda and not args.cpu else "cpu")

    model = build_model(args)
    print(model)
    model = model.to(device)

    # optimizer = torch.optim.Adam(model.parameters())
    betas = (0.9, 0.999)
    eps = 1e-8
    optimizer = BertAdam(model, lr=args.learning_rate, b1=betas[0], b2=betas[1], e=eps)
    pad_id = word_to_idx['PAD']
    pad_id_labels = tag_to_idx['PAD']
    batcher = SamplingBatcher(np.asarray(train_sentences, dtype=object),
                              np.asarray(train_labels, dtype=object),
                              batch_size=args.batch_size,
                              pad_id=pad_id,
                              pad_id_labels=pad_id_labels)

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
            input_, _, _, _, _ = batch
            optimizer.zero_grad()
            attn_mask = get_attn_pad_mask(input_, input_, pad_id)
            loss = model.score(batch, attn_mask)
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

    print('Training time:{}'.format(time.time()-start_time))

    def get_idx_to_tag(label_ids):
        return [idx_to_tag.get(idx) for idx in label_ids]

    def get_idx_to_word(words_ids):
        return [idx_to_word.get(idx) for idx in words_ids]

    def predict(dl, model, id2label, id2cls=None):
        model.eval()
        idx = 0
        preds_cpu = []
        preds_cpu_cls = []
        for batch in tqdm(dl, total=len(dl), leave=False, desc="Predicting"):
            idx += 1
            input_, _, _, _, labels_mask = batch
            # Create attn mask
            attn_mask = get_attn_pad_mask(input_, input_, pad_id)

            preds = model(batch, attn_mask=attn_mask)
            preds = preds.view(labels_mask.shape)

            if id2cls is not None:
                preds, preds_cls = preds
                preds_cpu_ = transformed_result_cls([preds_cls], [preds_cls], id2cls, False)
                preds_cpu_cls.extend(preds_cpu_)

            preds_cpu_ = transformed_result([preds], [labels_mask], id2label)
            preds_cpu.extend(preds_cpu_)
        if id2cls is not None:
            return preds_cpu, preds_cpu_cls
        return preds_cpu

    model, model_args = load_model_state(f'{output_dir}/{prefix}_best_model.pt')
    batcher_test = SamplingBatcher(np.asarray(test_sentences, dtype=object),
                                   np.asarray(test_labels, dtype=object),
                                   batch_size=args.batch_size, pad_id=pad_id,
                                   pad_id_labels=pad_id_labels)
    ne_class_list = set()
    true_labels_for_testing = []
    results_of_prediction = []
    with open(f'{output_dir}/{prefix}_label.txt', 'w', encoding='utf8') as t, \
            open(f'{output_dir}/{prefix}_predict.txt', 'w', encoding='utf8') as p, \
            open(f'{output_dir}/{prefix}_text.txt', 'w', encoding='utf8') as textf:
        with torch.no_grad():
            # predict() method returns final labels not the label_ids
            preds = predict(batcher_test, model, idx_to_tag)
            cnt = 0
            for text, labels, predict_labels in zip(test_sentences, test_labels, preds):
                cnt += 1
                tag_labels_true = get_idx_to_tag(labels)
                text_ = get_idx_to_word(text)

                tag_labels_predicted = ' '.join(predict_labels)
                tag_labels_true = ' '.join(tag_labels_true)
                text_ = ' '.join(text_)
                p.write(tag_labels_predicted + '\n')
                t.write(tag_labels_true + '\n')
                textf.write(text_ + '\n')
                tag_labels_true = tag_labels_true.strip().replace('_', '-').split()
                tag_labels_predicted = tag_labels_predicted.strip().replace('_', '-').split()
                biluo_tags_true = get_biluo(tag_labels_true)
                biluo_tags_predicted = get_biluo(tag_labels_predicted)

                doc = Doc(text_)
                offset_true_labels = offset_from_biluo(doc, biluo_tags_true)
                offset_predicted_labels = offset_from_biluo(doc, biluo_tags_predicted)

                ent_labels = dict()
                for ent in offset_true_labels:
                    start, stop, ent_type = ent
                    ent_type = ent_type.replace('_', '')
                    ne_class_list.add(ent_type)
                    if ent_type in ent_labels:
                        ent_labels[ent_type].append((start, stop))
                    else:
                        ent_labels[ent_type] = [(start, stop)]
                true_labels_for_testing.append(ent_labels)

                ent_labels = dict()
                for ent in offset_predicted_labels:
                    start, stop, ent_type = ent
                    ent_type = ent_type.replace('_', '')
                    if ent_type in ent_labels:
                        ent_labels[ent_type].append((start, stop))
                    else:
                        ent_labels[ent_type] = [(start, stop)]
                results_of_prediction.append(ent_labels)

    from eval.quality import calculate_prediction_quality
    f1, precision, recall, results = \
        calculate_prediction_quality(true_labels_for_testing,
                                     results_of_prediction,
                                     tuple(ne_class_list))
    print(f1, precision, recall, results)


if __name__ == '__main__':
    parser = get_training_options()
    args = parser.parse_args()
    args = update_args_arch(args)
    train(args)
