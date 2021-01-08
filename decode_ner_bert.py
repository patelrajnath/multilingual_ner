import os
import torch

from models.model_utils import set_seed, load_model_state, get_attn_pad_mask, predict, get_device
from models import tqdm
from models.bert_data import TextDataLoader, TextDataSet
from sklearn_crfsuite.metrics import flat_classification_report
from analyze_utils.utils import bert_labels2tokens
from options.args_parser import get_prediction_options_bert

set_seed(seed_value=999)


def decode(options):
    device = get_device(options)
    output_dir = options.output_dir
    try:
        os.makedirs(output_dir)
    except:
        pass

    prefix = options.test.split('_')[0] if len(options.test.split('_')) > 1 else options.test.split('.')[0]
    # Load the trained model
    model, model_args = load_model_state(options.model)
    model = model.to(device)
    model.eval()

    ds = TextDataSet.create(
        df_path=os.path.join(options.data_dir, options.test),
        idx2labels_path=options.idx2labels,
        model_name=model_args.model_name,
        model_type=model_args.model_type,
        markup='BIO',
        max_sequence_length=model_args.max_seq_len
    )
    dl = TextDataLoader(ds, device=device, batch_size=options.batch_size, shuffle=False)
    pad_id = 0

    with open(f'{output_dir}/{prefix}_label_bert.txt', 'w') as t, \
            open(f'{output_dir}/{prefix}_predict_bert.txt', 'w') as p, \
            open(f'{output_dir}/{prefix}_text_bert.txt', 'w') as textf:
        with torch.no_grad():
            preds = predict(dl, model, ds.idx2label, pad_id=pad_id)
            pred_tokens, pred_labels = bert_labels2tokens(dl, preds)
            true_tokens, true_labels = bert_labels2tokens(dl, [x.bert_labels for x in dl.dataset])

            assert pred_tokens == true_tokens
            tokens_report = flat_classification_report(true_labels, pred_labels,
                                                       labels=ds.idx2label[4:], digits=4)
            print(tokens_report)
            t.write('\n'.join([' '.join([item for item in t_label]) for t_label in true_labels]) + '\n')
            p.write('\n'.join([' '.join([item for item in p_label]) for p_label in pred_labels]) + '\n')
            textf.write('\n'.join([' '.join([item for item in t_token]) for t_token in true_tokens]) + '\n')


if __name__ == '__main__':
    parser = get_prediction_options_bert()
    args = parser.parse_args()
    print(args)
    decode(args)
