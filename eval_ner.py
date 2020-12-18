import argparse
import os

from datautils import Doc
from datautils.biluo_from_predictions import get_biluo
from datautils.iob_utils import offset_from_biluo


def eval_(output_dir, t_labels, p_labels, text):
    with open(os.path.join(output_dir, t_labels), 'r') as t, \
            open(os.path.join(output_dir, p_labels), 'r') as p, \
            open(os.path.join(output_dir, text), 'r') as textf:
        ne_class_list = set()
        true_labels_for_testing = []
        results_of_prediction = []
        for text, true_labels, predicted_labels in zip(textf, t, p):
            true_labels = true_labels.strip().replace('_', '-').split()
            predicted_labels = predicted_labels.strip().replace('_', '-').split()
            biluo_tags_true = get_biluo(true_labels)
            biluo_tags_predicted = get_biluo(predicted_labels)

            doc = Doc(text.strip())
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
    print(ne_class_list)
    f1, precision, recall, results = \
        calculate_prediction_quality(true_labels_for_testing,
                                     results_of_prediction,
                                     tuple(ne_class_list))
    print(f1, precision, recall, results)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--t_labels', type=str, default="ubuntu_label.txt")
    parser.add_argument('--p_labels', type=str, default="ubuntu_predict.txt")
    parser.add_argument('--text', type=str, default="ubuntu_text.txt")
    return vars(parser.parse_args())


if __name__ == '__main__':
    eval_(**parse_args())
