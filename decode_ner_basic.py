import os
import numpy as np
import torch

from datautils import Doc
from datautils.batchers import SamplingBatcher
from datautils.biluo_from_predictions import get_biluo
from datautils.iob_utils import offset_from_biluo
from datautils.vocab import load_vocabs
from models.model_utils import load_model_state, set_seed, get_device, predict

# Set seed to have consistent results
from options.args_parser import get_prediction_options
from datautils.prepare_data import prepare

set_seed(seed_value=999)
np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)


def decode(options):
    prefix = options.test_text.split('_')[0] if len(options.test_text.split('_')) > 1 \
        else options.test_text.split('.')[0]

    device = get_device(args)
    output_dir = options.output_dir
    try:
        os.makedirs(output_dir)
    except:
        pass
    model, model_args = load_model_state(options.model)
    model = model.to(device)

    vocab_path = os.path.join(model_args.data_dir, model_args.vocab)
    tag_path = os.path.join(model_args.data_dir, model_args.tag_set)
    word_to_idx, idx_to_word, tag_to_idx, idx_to_tag = load_vocabs(vocab_path, tag_path)

    *_, test_sentences, test_labels = prepare(options, word_to_idx, tag_to_idx)

    def get_idx_to_tag(label_ids):
        return [idx_to_tag.get(idx) for idx in label_ids]

    def get_idx_to_word(words_ids):
        return [idx_to_word.get(idx) for idx in words_ids]

    pad_id = word_to_idx['PAD']
    pad_id_labels = tag_to_idx['PAD']
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
            preds = predict(batcher_test, model, idx_to_tag, pad_id=pad_id)
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
    parser = get_prediction_options()
    args = parser.parse_args()
    decode(args)
