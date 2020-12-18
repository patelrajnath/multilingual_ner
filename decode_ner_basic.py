import os
import time
from math import inf
import numpy as np
import torch
from datautils.batchers import SamplingBatcher
from datautils import Doc
from datautils.biluo_from_predictions import get_biluo
from datautils.iob_utils import offset_from_biluo
from model_utils import save_state, load_model_state, set_seed

# Set seed to have consistent results
from models.ner import BasicNER, loss_fn
from options.args_parser import get_training_options
from options.model_params import HParamSet
from datautils.prepare_data import prepare

set_seed(seed_value=999)
np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)


def train(options):
    idx_to_word, idx_to_tag, _, _, test_sentences, test_labels = prepare(options)

    params = HParamSet(options)
    params.vocab_size = len(idx_to_word)
    params.number_of_tags = len(idx_to_tag)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = BasicNER(params=params)
    model = model.to(device)
    output_dir = options.output_dir
    try:
        os.makedirs(output_dir)
    except:
        pass

    prefix = options.test_text.split('_')[0] if len(options.test_text.split('_')) > 1 \
        else options.test_text.split('.')[0]

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


if __name__ == '__main__':
    parser = get_training_options()
    args = parser.parse_args()
    train(args)
