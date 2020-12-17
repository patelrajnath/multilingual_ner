import argparse

from options.model_params import HParamSet


def parse_args_data():
    parse_data_args = argparse.ArgumentParser()
    parse_data_args.add_argument('--data_dir', type=str, default='data/accounts')
    parse_data_args.add_argument('--train_text', type=str, default='accounts_train_text.txt')
    parse_data_args.add_argument('--train_label', type=str, default='accounts_train_labels.txt')
    parse_data_args.add_argument('--dev_text', type=str, default=None)
    parse_data_args.add_argument('--dev_label', type=str, default=None)
    parse_data_args.add_argument('--test_text', type=str, default='accounts_test_text.txt')
    parse_data_args.add_argument('--test_label', type=str, default='accounts_test_labels.txt')
    parse_data_args.add_argument('--vocab', type=str, default='words.txt')
    parse_data_args.add_argument('--tag_set', type=str, default='tags.txt')
    return vars(parse_data_args.parse_args())


def parse_args_model():
    parser_model_args = argparse.ArgumentParser()
    parser_model_args.add_argument('--max_sts_score', type=int, default=32)
    parser_model_args.add_argument('--balance_data', type=bool, default=False)
    parser_model_args.add_argument('--output_size', type=int, default=None)
    parser_model_args.add_argument('--batch_size', type=int, default=32)
    parser_model_args.add_argument('--activation', type=str, default='relu')
    parser_model_args.add_argument('--hidden_layer_size', type=int, default=512)
    parser_model_args.add_argument('--num_hidden_layers', type=int, default=1)
    parser_model_args.add_argument('--embedding_dim', type=int, default=256)
    parser_model_args.add_argument('--dropout', type=float, default=0.1)
    parser_model_args.add_argument('--optimizer', type=str, default='sgd')
    parser_model_args.add_argument('--learning_rate', type=float, default=0.7)
    parser_model_args.add_argument('--lr_decay_pow', type=int, default=1)
    parser_model_args.add_argument('--epochs', type=int, default=100)
    parser_model_args.add_argument('--seed', type=int, default=999)
    parser_model_args.add_argument('--max_steps', type=int, default=1500)
    parser_model_args.add_argument('--patience', type=int, default=100)
    parser_model_args.add_argument('--eval_each_epoch', type=bool, default=False)
    return vars(parser_model_args.parse_args())


if __name__ == '__main__':
    params = HParamSet(**parse_args_model())
    print(params.batch_size)
