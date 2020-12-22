import argparse
import os
import logging
import pandas as pd

log = logging.getLogger(__name__)


def convert_ner_format(data_dir, train_bio_csv, text, labels):
    with open(os.path.join(data_dir, text), 'w', encoding='utf8') as text_file, \
            open(os.path.join(data_dir, labels), 'w', encoding='utf8') as label_file:
        df = pd.read_csv(os.path.join(data_dir, train_bio_csv), encoding='utf8', sep='\t')
        text_file.write('\n'.join(df.text.tolist()))
        label_file.write('\n'.join(df.labels.tolist()))


def process_bio(data_dir,
                train_bio_csv,
                train_text,
                train_label,
                test_bio_csv=None,
                test_text=None,
                test_label=None,
                dev_bio_csv=None,
                dev_text=None,
                dev_label=None
                ):
    if train_bio_csv:
        convert_ner_format(data_dir, train_bio_csv, train_text, train_label)
    if test_text and test_label:
        convert_ner_format(data_dir, test_bio_csv, test_text, test_label)
    if dev_text and dev_label:
        convert_ner_format(data_dir, dev_bio_csv, dev_text, dev_label)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/conll2003')
    parser.add_argument('--train_bio_csv', type=str, default='eng.train.train.csv')
    parser.add_argument('--train_text', type=str, default='conll2003_train_text.txt')
    parser.add_argument('--train_label', type=str, default='conll2003_train_labels.txt')
    parser.add_argument('--dev_bio_csv', type=str, default='eng.testb.dev.csv')
    parser.add_argument('--dev_text', type=str, default='conll2003.testa.text.txt')
    parser.add_argument('--dev_label', type=str, default='conll2003.testa.labels.txt')
    parser.add_argument('--test_bio_csv', type=str, default='eng.testb.dev.csv')
    parser.add_argument('--test_text', type=str, default='conll2003_testb_text.txt')
    parser.add_argument('--test_label', type=str, default='conll2003_testb_labels.txt')
    return vars(parser.parse_args())


if __name__ == '__main__':
    process_bio(**parse_args())
