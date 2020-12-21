import argparse
from models import ARCH_MODEL_REGISTRY


def get_parser(desc, default_task='ner'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/ubuntu')
    parser.add_argument('--train_text', type=str, default='ubuntu_train_text.txt')
    parser.add_argument('--train_label', type=str, default='ubuntu_train_labels.txt')
    parser.add_argument('--dev_text', type=str, default=None)
    parser.add_argument('--dev_label', type=str, default=None)
    parser.add_argument('--test_text', type=str, default='ubuntu_test_text.txt')
    parser.add_argument('--test_label', type=str, default='ubuntu_test_labels.txt')
    parser.add_argument('--vocab', type=str, default='words.txt')
    parser.add_argument('--tag_set', type=str, default='tags.txt')
    return parser


def get_parser_bert(desc, default_task='ner'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/ubuntu')
    parser.add_argument('--train', type=str, default='ubuntu_train_text.txt.csv')
    parser.add_argument('--dev', type=str, default=None)
    parser.add_argument('--test', type=str, default='ubuntu_test_text.txt.csv')
    parser.add_argument('--idx2labels', type=str, default='idx2labels.txt')
    return parser


def add_training_args(parser):
    group = parser.add_argument_group('Model training')
    group.add_argument('--max_sts_score', type=int, default=5)
    group.add_argument('--balance_data', action='store_true')
    group.add_argument('--output_size', type=int, default=None)
    group.add_argument('--batch_size', type=int, default=32)
    group.add_argument('--optimizer', type=str, default='sgd')
    group.add_argument('--learning_rate', type=float, default=0.001)
    group.add_argument('--lr_decay_pow', type=int, default=1)
    group.add_argument('--epochs', type=int, default=100)
    group.add_argument('--seed', type=int, default=999)
    group.add_argument('--max_steps', type=int, default=1500)
    group.add_argument('--patience', type=int, default=100)
    group.add_argument('--eval_each_epoch', action='store_true')
    group.add_argument('--output_dir', type=str, default='outputs')
    group.add_argument('--cpu', action='store_true')
    group.add_argument('--arch', '-a', default='ner', metavar='ARCH', required=True,
                       choices=ARCH_MODEL_REGISTRY.keys(),
                       help='Model Architecture')

    return group


def get_training_options(default_task='NER'):
    parser = get_parser('Preprocessing', default_task)
    add_training_args(parser)
    return parser


def get_training_options_bert(default_task='NER'):
    parser = get_parser_bert('Preprocessing', default_task)
    add_training_args(parser)
    return parser


def get_preprocessing_options(default_task='NER'):
    parser = get_parser('Preprocessing', default_task)
    return parser


if __name__ == '__main__':
    parser = get_training_options()
    options = parser.parse_args()
    print(options.data_dir)
