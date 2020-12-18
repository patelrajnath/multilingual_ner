import argparse
import os
import logging
import pandas as pd

log = logging.getLogger(__name__)


def read_data(input_file_text, input_file_label):
    """Reads a BIO data."""
    with open(input_file_text, "r", encoding="utf-8") as f_text, \
            open(input_file_label, "r", encoding="utf-8") as f_label:
        lines = []
        for text, label in zip(f_text, f_label):
            if len(text.split()) == len(label.split()):
                lbl = ' '.join([lbl.replace('-', '_')for lbl in label.split()])
                lines.append([lbl.strip(), text.strip()])
            else:
                log.warning(f'Bad training data, token and label count is different. \n'
                            f'text: {text}label: {label}')
        return lines


def convert_bert_format(data_dir, text, labels):
    train_f = read_data(os.path.join(data_dir, text), os.path.join(data_dir, labels))
    df = pd.DataFrame({"labels": [x[0] for x in train_f], "text": [x[1] for x in train_f]})
    df["cls"] = df["labels"].apply(lambda x: all([y.split("_")[0] == "O" for y in x.split()]))
    df.to_csv(os.path.join(data_dir, "{}.csv".format(text)), index=False, sep="\t")


def process_bio(data_dir, train_text, train_label, test_text=None, test_label=None, dev_text=None,
                dev_label=None):
    if train_text and train_label:
        convert_bert_format(data_dir, train_text, train_label)
    if test_text and test_label:
        convert_bert_format(data_dir, test_text, test_label)
    if dev_text and dev_label:
        convert_bert_format(data_dir, dev_text, dev_label)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/accounts')
    parser.add_argument('--train_text', type=str, default='accounts_train_text.txt')
    parser.add_argument('--train_label', type=str, default='accounts_train_labels.txt')
    parser.add_argument('--dev_text', type=str, default=None)
    parser.add_argument('--dev_label', type=str, default=None)
    parser.add_argument('--test_text', type=str, default='accounts_test_text.txt')
    parser.add_argument('--test_label', type=str, default='accounts_test_labels.txt')
    return vars(parser.parse_args())


if __name__ == '__main__':
    process_bio(**parse_args())
