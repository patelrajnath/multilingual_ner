import os
import torch

from models.model_utils import set_seed, load_model_state, get_attn_pad_mask
from models import tqdm
from models.bert_data import TextDataLoader, TextDataSet
from sklearn_crfsuite.metrics import flat_classification_report
from analyze_utils.utils import bert_labels2tokens
from options.args_parser import get_prediction_options_bert

set_seed(seed_value=999)


def decode(options):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda and not options.cpu else "cpu")

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
        idx2labels_path=os.path.join(options.data_dir, options.idx2labels),
        model_name=model_args.model_name,
        markup='BIO',
        max_sequence_length=100
    )
    dl = TextDataLoader(ds, device=device, batch_size=options.batch_size, shuffle=False)

    def transformed_result(preds, mask, id2label, target_all=None, pad_idx=0):
        preds_cpu = []
        targets_cpu = []
        lc = len(id2label)
        if target_all is not None:
            for batch_p, batch_t, batch_m in zip(preds, target_all, mask):
                for pred, true_, bm in zip(batch_p, batch_t, batch_m):
                    sent = []
                    sent_t = []
                    bm = bm.sum().cpu().data.tolist()
                    for p, t in zip(pred[:bm], true_[:bm]):
                        p = p.cpu().data.tolist()
                        p = p if p < lc else pad_idx
                        sent.append(p)
                        sent_t.append(t.cpu().data.tolist())
                    preds_cpu.append([id2label[w] for w in sent])
                    targets_cpu.append([id2label[w] for w in sent_t])
        else:
            for batch_p, batch_m in zip(preds, mask):

                for pred, bm in zip(batch_p, batch_m):
                    assert len(pred) == len(bm)
                    bm = bm.sum().cpu().data.tolist()
                    sent = pred[:bm].cpu().data.tolist()
                    preds_cpu.append([id2label[w] for w in sent])
        if target_all is not None:
            return preds_cpu, targets_cpu
        else:
            return preds_cpu

    def transformed_result_cls(preds, target_all, cls2label, return_target=True):
        preds_cpu = []
        targets_cpu = []
        for batch_p, batch_t in zip(preds, target_all):
            for pred, true_ in zip(batch_p, batch_t):
                preds_cpu.append(cls2label[pred.cpu().data.tolist()])
                if return_target:
                    targets_cpu.append(cls2label[true_.cpu().data.tolist()])
        if return_target:
            return preds_cpu, targets_cpu
        return preds_cpu

    def predict(dl, model, id2label, id2cls=None):
        model.eval()
        idx = 0
        pad_id = 0
        preds_cpu = []
        preds_cpu_cls = []
        for batch in tqdm(dl, total=len(dl), leave=False, desc="Predicting"):
            idx += 1
            input_, labels_mask, input_type_ids, labels_ids = batch
            # Create attn mask
            attn_mask = get_attn_pad_mask(input_, input_, pad_id)
            preds = model(input_, attn_mask)
            preds = preds.argmax(dim=1)
            preds = preds.view(labels_mask.shape)
            if id2cls is not None:
                preds, preds_cls = preds
                preds_cpu_ = transformed_result_cls([preds_cls], [preds_cls], id2cls, False)
                preds_cpu_cls.extend(preds_cpu_)

            preds_cpu_ = transformed_result([preds], [labels_mask], id2label)
            preds_cpu.extend(preds_cpu_)
        if id2cls is not None:
            return preds_cpu, preds_cpu_cls
        return preds_cpu

    with open(f'{output_dir}/{prefix}_label_bert.txt', 'w') as t, \
            open(f'{output_dir}/{prefix}_predict_bert.txt', 'w') as p, \
            open(f'{output_dir}/{prefix}_text_bert.txt', 'w') as textf:
        with torch.no_grad():
            preds = predict(dl, model, ds.idx2label)
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
