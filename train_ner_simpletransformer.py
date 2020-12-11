import ast

from simpletransformers.ner import NERModel


if __name__ == '__main__':
    args = {
        "output_dir": "outputs/",
        "cache_dir": "cache/",
        "best_model_dir": "outputs/best_model/",

        "fp16": False,
        "max_seq_length": 32,
        "train_batch_size": 8,
        "eval_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "num_train_epochs": 10,
        "weight_decay": 0,
        "learning_rate": 4e-5,
        "adam_epsilon": 1e-8,
        "warmup_ratio": 0.06,
        "warmup_steps": 0,
        "max_grad_norm": 1.0,
        "do_lower_case": False,

        "logging_steps": 50,
        "evaluate_during_training": True,
        "evaluate_during_training_steps": 500,
        "evaluate_during_training_verbose": False,
        "use_cached_eval_features": False,
        "save_eval_checkpoints": False,
        "save_steps": 0,
        "no_cache": False,
        "save_model_every_epoch": False,
        "tensorboard_dir": None,

        "overwrite_output_dir": True,
        "reprocess_input_data": True,
        "process_count": 1,
        "n_gpu": 0,
        "silent": False,
        "use_multiprocessing": False,

        "wandb_project": None,
        "wandb_kwargs": {},

        "use_early_stopping": True,
        "early_stopping_patience": 3,
        "early_stopping_delta": 0,
        "early_stopping_metric": "eval_loss",
        "early_stopping_metric_minimize": True,

        "manual_seed": None,
        "encoding": None,
        "config": {},
    }

    with open('tag-set.txt', 'r') as f:
       ents_dict = set(ast.literal_eval(f.read()))

    print(ents_dict)
    # Create a NERModel
    model = NERModel('bert', 'bert-base-cased', args=args, use_cuda=False, labels=ents_dict)

    model.train_model('sample_data/train.txt', eval_data='sample_data/test.txt')

    results, model_outputs, predictions = model.eval_model('sample_data/test.txt')

    print(results)
