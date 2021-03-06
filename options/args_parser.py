import argparse
from models import ARCH_MODEL_REGISTRY, MODEL_REGISTRY, ARCH_CONFIG_REGISTRY


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


def get_parser_prediction(desc, default_task='ner'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/ubuntu')
    parser.add_argument('--test_text', type=str, default='ubuntu_test_text.txt')
    parser.add_argument('--test_label', type=str, default='ubuntu_test_labels.txt')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model', type=str, default='outputs/ubuntu_best_model.pt')
    parser.add_argument('--output_dir', type=str, default='outputs')
    return parser


def get_parser_prediction_bert(desc, default_task='ner'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/ubuntu')
    parser.add_argument('--test', type=str, default='ubuntu_test_text.txt.csv')
    parser.add_argument('--idx2labels', type=str, default='data/ubuntu/idx2labels.txt')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--model', type=str, default='outputs/ubuntu_best_model_bert.pt')

    return parser


def get_parser_bert(desc, default_task='ner'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/ubuntu')
    parser.add_argument('--train', type=str, default='ubuntu_train_text.txt.csv')
    parser.add_argument('--dev', type=str, default=None)
    parser.add_argument('--test', type=str, default='ubuntu_test_text.txt.csv')
    parser.add_argument('--idx2labels', type=str, default='idx2labels.txt')
    parser.add_argument('--max_seq_len', type=int, default=100)
    return parser


def add_training_args(parser):
    group = parser.add_argument_group('Model training')
    group.add_argument('--batch_size', type=int, default=32)
    group.add_argument('--optimizer', type=str, default='sgd')
    group.add_argument('--learning_rate', type=float, default=0.001)
    group.add_argument('--epochs', type=int, default=100)
    group.add_argument('--seed', type=int, default=999)
    group.add_argument('--max_steps', type=int, default=1500)
    group.add_argument('--patience', type=int, default=100)
    group.add_argument('--eval_each_epoch', action='store_true')
    group.add_argument('--output_dir', type=str, default='outputs')
    group.add_argument('--cpu', action='store_true')
    group.add_argument('--shuffle', action='store_true')
    group.add_argument('--save_cache_features', action='store_true')
    group.add_argument('--reset_cache', action='store_true')
    group.add_argument('--in_memory_cache', action='store_true')
    group.add_argument('--cache_features', action='store_true')
    group.add_argument('--dynamic_quantize', action='store_true')
    group.add_argument('--arch', '-a', default='ner', metavar='ARCH', required=True,
                       choices=ARCH_MODEL_REGISTRY.keys(),
                       help='Model Architecture')
    return group


def add_args_arch(parser: argparse.ArgumentParser):
    """
    Args:
        parser (ArgumentParser): the parse
    """
    args, leftovers = parser.parse_known_args()
    model_specific_group = parser.add_argument_group(
        "Model-specific configuration",
        # Only include attributes which are explicitly given as command-line
        # arguments or which have default values.
        argument_default=argparse.SUPPRESS,
    )
    if args.arch in ARCH_MODEL_REGISTRY:
        ARCH_MODEL_REGISTRY[args.arch].add_args(model_specific_group)
    elif args.arch in MODEL_REGISTRY:
        MODEL_REGISTRY[args.arch].add_args(model_specific_group)
    else:
        raise RuntimeError()


def update_args_arch(args):
    # Apply architecture configuration.
    if hasattr(args, "arch") and args.arch in ARCH_CONFIG_REGISTRY:
        ARCH_CONFIG_REGISTRY[args.arch](args)
    return args


def get_training_options(default_task='NER'):
    parser = get_parser('Preprocessing', default_task)
    add_training_args(parser)
    add_args_arch(parser)
    return parser


def get_training_options_bert(default_task='NER'):
    parser = get_parser_bert('Preprocessing', default_task)
    add_training_args(parser)
    add_args_arch(parser)
    return parser


def get_preprocessing_options(default_task='NER'):
    parser = get_parser('Preprocessing', default_task)
    return parser


def get_prediction_options(default_task='NER'):
    parser = get_parser_prediction('Prediction', default_task)
    return parser


def get_prediction_options_bert(default_task='NER'):
    parser = get_parser_prediction_bert('Prediction', default_task)
    return parser


if __name__ == '__main__':
    parser = get_training_options()
    options = parser.parse_args()
    print(options)
