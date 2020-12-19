import os
import time
from math import inf
import numpy as np
import torch
from datautils.batchers import SamplingBatcher
from datautils import Doc
from datautils.biluo_from_predictions import get_biluo
from datautils.iob_utils import offset_from_biluo
from models.crf import CRFNet
from models.model_utils import save_state, load_model_state, set_seed, loss_fn

# Set seed to have consistent results
from models.ner import BasicNER, AttnNER
from models.optimization import BertAdam
from options.args_parser import get_training_options
from options.model_params import HParamSet
from datautils.prepare_data import prepare

set_seed(seed_value=999)
np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)


def train(options):
    idx_to_word, idx_to_tag, train_sentences, train_labels, test_sentences, test_labels = prepare(options)
    word_to_idx = {idx_to_word[key]: key for key in idx_to_word}
    tag_to_idx = {idx_to_tag[key]: key for key in idx_to_tag}

    model_params = HParamSet(options)
    model_params.vocab_size = len(idx_to_word)
    model_params.number_of_tags = len(idx_to_tag)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # model = BasicNER(params=params)
    model = CRFNet(model_params, tag_to_idx, device)
    model = model.to(device)

    # optimizer = torch.optim.Adam(model.parameters())
    betas = (0.9, 0.999)
    eps = 1e-8
    optimizer = BertAdam(model, lr=model_params.learning_rate, b1=betas[0], b2=betas[1], e=eps)

    batcher = SamplingBatcher(np.asarray(train_sentences, dtype=object), np.asarray(train_labels, dtype=object),
                              batch_size=model_params.batch_size, pad_id=word_to_idx['PAD'])

    updates = 1
    total_loss = 0
    best_loss = +inf
    stop_training = False

    output_dir = options.output_dir
    try:
        os.makedirs(output_dir)
    except:
        pass

    prefix = options.train_text.split('_')[0] if len(options.train_text.split('_')) > 1 \
        else options.train_text.split('.')[0]

    start_time = time.time()
    for epoch in range(model_params.epochs):
        for batch in batcher:
            updates += 1
            batch_data, batch_labels, batch_len, mask_x, mask_y = batch
            optimizer.zero_grad()
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            mask_y = mask_y.to(device)
            loss = model.neg_log_likelihood(batch_data, batch_labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.data
            if updates % model_params.patience == 0:
                print(f'Epoch: {epoch}, Updates:{updates}, Loss: {total_loss}')
                if best_loss > total_loss:
                    save_state(f'{output_dir}/{prefix}_best_model.pt', model, loss_fn, optimizer, updates)
                    best_loss = total_loss
                total_loss = 0
            if updates % model_params.max_steps == 0:
                stop_training = True
                break

        if stop_training:
            break

    print('Training time:{}'.format(time.time()-start_time))

    def get_idx_to_tag(label_ids):
        return [idx_to_tag.get(idx) for idx in label_ids]

    def get_idx_to_word(words_ids):
        return [idx_to_word.get(idx) for idx in words_ids]

    updates = load_model_state(f'{output_dir}/{prefix}_best_model.pt', model)
    ne_class_list = set()
    true_labels_for_testing = []
    results_of_prediction = []
    with open(f'{output_dir}/{prefix}_label.txt', 'w', encoding='utf8') as t, \
            open(f'{output_dir}/{prefix}_predict.txt', 'w', encoding='utf8') as p, \
            open(f'{output_dir}/{prefix}_text.txt', 'w', encoding='utf8') as textf:
        with torch.no_grad():
            model.eval()
            cnt = 0
            for text, label in zip(test_sentences, test_labels):
                cnt += 1
                text_tensor = torch.LongTensor(text).unsqueeze(0).to(device)
                labels = torch.LongTensor(label).unsqueeze(0).to(device)
                score, predict_labels = model(text_tensor)
                labels = labels.view(-1)

                true_labels = labels.cpu().data.tolist()
                tag_labels_predicted = get_idx_to_tag(predict_labels)
                tag_labels_true = get_idx_to_tag(true_labels)
                text_ = get_idx_to_word(text)

                tag_labels_predicted = ' '.join(tag_labels_predicted)
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
    train(args)
