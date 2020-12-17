import os
import time
from math import inf
import numpy as np
import torch
from batchers import SamplingBatcher
from document.doc import Doc
from eval.biluo_from_predictions import get_biluo
from eval.iob_utils import offset_from_biluo
from model_utils import save_state, load_model_state, set_seed

# Set seed to have consistent results
from models.ner import Net, loss_fn
from options.args_parser import parse_args_data, parse_args_model
from options.model_params import HParamSet
from prepare_data import prepare

set_seed(seed_value=999)
np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)

data_type = 'accounts'
# data_type = 'alliance'
# data_type = 'wallet'
# data_type = 'ubuntu'
# data_type = 'snips'
# data_type = 'nlu'
idx_to_word, idx_to_tag, train_sentences, train_labels, test_sentences, test_labels = \
    prepare(**parse_args_data())

word_to_idx = {idx_to_word[key]: key for key in idx_to_word}
tag_to_idx = {idx_to_tag[key]: key for key in idx_to_tag}

params = HParamSet(**parse_args_model())
params.vocab_size = len(idx_to_word)
params.number_of_tags = len(idx_to_tag)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
model = Net(params=params)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters())

batcher = SamplingBatcher(np.asarray(train_sentences, dtype=object), np.asarray(train_labels, dtype=object),
                          batch_size=params.batch_size, pad_id=word_to_idx['PAD'])

updates = 1
total_loss = 0
best_loss = +inf
stop_training = False

out_dir = 'outputs'
try:
    os.makedirs(out_dir)
except:
    pass

start_time = time.time()
for epoch in range(params.epochs):
    for batch in batcher:
        updates += 1
        batch_data, batch_labels, batch_len, mask_x, mask_y = batch
        optimizer.zero_grad()
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        mask_y = mask_y.to(device)
        output_batch = model(batch_data)
        loss = loss_fn(output_batch, batch_labels, mask_y)

        loss.backward()
        optimizer.step()

        total_loss += loss.data
        if updates % params.patience == 0:
            print(f'Epoch: {epoch}, Updates:{updates}, Loss: {total_loss}')
            if best_loss > total_loss:
                save_state(f'{out_dir}/{data_type}_best_model.pt', model, loss_fn, optimizer, updates)
                best_loss = total_loss
            total_loss = 0
        if updates % params.max_steps == 0:
            stop_training = True
            break

    if stop_training:
        break

print('Training time:{}'.format(time.time()-start_time))


def get_idx_to_tag(label_ids):
    return [idx_to_tag.get(idx) for idx in label_ids]


def get_idx_to_word(words_ids):
    return [idx_to_word.get(idx) for idx in words_ids]


updates = load_model_state(f'{out_dir}/{data_type}_best_model.pt', model)

ne_class_list = set()
true_labels_for_testing = []
results_of_prediction = []
with open(f'{out_dir}/{data_type}_label.txt', 'w', encoding='utf8') as t, \
        open(f'{out_dir}/{data_type}_predict.txt', 'w', encoding='utf8') as p, \
        open(f'{out_dir}/{data_type}_text.txt', 'w', encoding='utf8') as textf:
    with torch.no_grad():
        model.eval()
        prediction_label_ids = []
        true_label_ids = []
        cnt = 0
        for text, label in zip(test_sentences, test_labels):
            cnt += 1
            text_tensor = torch.LongTensor(text).unsqueeze(0).to(device)
            lable = torch.LongTensor(label).unsqueeze(0).to(device)
            predict = model(text_tensor)
            predict_labels = predict.argmax(dim=1)
            predict_labels = predict_labels.view(-1)
            lable = lable.view(-1)

            predicted_labels = predict_labels.cpu().data.tolist()
            true_labels = lable.cpu().data.tolist()
            tag_labels_predicted = get_idx_to_tag(predicted_labels)
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
