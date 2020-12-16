import pandas
import spacy
from spacy.gold import offsets_from_biluo_tags

from eval.biluo_from_bio import get_biluo

data_type = 'accounts'
nlp_blank = spacy.blank('en')

# with open('{}_label_bert_biluo.txt'.format(data_type), 'w') as t_write, \
#         open('{}_predict_bert_biluo.txt'.format(data_type), 'w') as p_write:

with open('{}_label_bert.txt'.format(data_type), 'r') as t, \
        open('{}_predict_bert.txt'.format(data_type), 'r') as p:
    df = pandas.read_csv('data/{0}/{0}_test_text.txt.csv'.format(data_type), encoding='utf8', sep='\t')
    ne_class_list = set()
    true_labels_for_testing = []
    results_of_prediction = []
    for text, true_labels, predicted_labels in zip(df.text.tolist(), t, p):
        doc = nlp_blank(text.strip())
        true_labels = true_labels.strip().replace('_', '-').split()
        predicted_labels = predicted_labels.strip().replace('_', '-').split()
        biluo_tags_true = get_biluo(true_labels)
        biluo_tags_predicted = get_biluo(predicted_labels)

        # t_write.write(' '.join(biluo_tags_true) + '\n')
        # p_write.write(' '.join(biluo_tags_predicted) + '\n')

        offset_true_labels = offsets_from_biluo_tags(doc, biluo_tags_true)
        offset_predicted_labels = offsets_from_biluo_tags(doc, biluo_tags_predicted)

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
