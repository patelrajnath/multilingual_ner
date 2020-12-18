import os
import time
from math import inf
import torch

from model_utils import set_seed, save_state, load_model_state
from models import bert_data, tqdm
from models.bert_data import get_data_loader_for_predict
from sklearn_crfsuite.metrics import flat_classification_report
from analyze_utils.utils import bert_labels2tokens, voting_choicer
from analyze_utils.plot_metrics import get_bert_span_report
from models.ner_bert import loss_fn, BertNER
from options.args_parser import get_training_options, get_training_options_bert
from options.model_params import HParamSet

set_seed(seed_value=999)


def train(options):
    model_params = HParamSet(options)
    data_type = options.train.split('_')[0] if len(options.train.split('_')) > 1 else options.train.split('.')[0]
    data = bert_data.LearnData.create(
        train_df_path=os.path.join(options.data_dir, options.train),
        valid_df_path=os.path.join(options.data_dir, options.test),
        idx2labels_path=os.path.join(options.data_dir, options.idx2labels),
        clear_cache=True,
        model_name=options.model_name,
        batch_size=model_params.batch_size,
        markup='BIO'
    )

    model_params.number_of_tags = len(data.train_ds.idx2label)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = BertNER(model_params, options, device=device)
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters())

    updates = 1
    total_loss = 0
    best_loss = +inf
    stop_training = False
    out_dir = 'outputs'
    try:
        os.makedirs(out_dir)
    except:
        pass

    start = time.time()
    for epoch in range(model_params.epochs):
        for batch in data.train_dl:
            updates += 1
            optimizer.zero_grad()
            input_, labels_mask, input_type_ids, labels = batch
            labels = labels.view(-1).to(device)
            labels_mask = labels_mask.view(-1).to(device)

            output = model(batch)
            loss = loss_fn(output, labels, labels_mask)

            loss.backward()
            optimizer.step()
            total_loss += loss.data

            if updates % model_params.patience == 0:
                print(f'Epoch: {epoch}, Updates:{updates}, Loss: {total_loss}')
                if best_loss > total_loss:
                    save_state(f'{out_dir}/{data_type}_best_model_bert.pt', model, loss_fn, optimizer, updates)
                    best_loss = total_loss
                total_loss = 0

            if updates % model_params.max_steps == 0:
                stop_training = True
                break

        if stop_training:
            break
    print(f'Training time: {time.time() - start}')

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
        preds_cpu = []
        preds_cpu_cls = []
        for batch in tqdm(dl, total=len(dl), leave=False, desc="Predicting"):
            idx += 1
            labels_mask, labels_ids = batch[1], batch[3]
            preds = model(batch)
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

    updates = load_model_state(f'{out_dir}/{data_type}_best_model_bert.pt', model)
    # dl = get_data_loader_for_predict(data, df_path='multilingual.test.csv')
    # dl = get_data_loader_for_predict(data, df_path='data/conll2003/eng.testb.dev.csv')
    dl = get_data_loader_for_predict(data, df_path='data/conll2003/eng.testa.dev.csv')
    # dl = get_data_loader_for_predict(data, df_path='data/conll2003-de/deuutf.testa.dev.csv')
    # dl = get_data_loader_for_predict(data, df_path='data/accounts/accounts_test_text.txt.csv')
    dl = get_data_loader_for_predict(data, df_path=os.path.join(options.data_dir, options.test))

    with open(f'{out_dir}/{data_type}_label_bert.txt', 'w') as t, \
            open(f'{out_dir}/{data_type}_predict_bert.txt', 'w') as p:
        with torch.no_grad():
            preds = predict(dl, model, data.train_ds.idx2label)
            pred_tokens, pred_labels = bert_labels2tokens(dl, preds)
            true_tokens, true_labels = bert_labels2tokens(dl, [x.bert_labels for x in dl.dataset])
            # print(true_tokens, true_labels)
            assert pred_tokens == true_tokens
            tokens_report = flat_classification_report(true_labels, pred_labels,
                                                       labels=data.train_ds.idx2label[4:], digits=4)
            print(tokens_report)
            t.write('\n'.join([' '.join([item for item in t_label]) for t_label in true_labels]) + '\n')
            p.write('\n'.join([' '.join([item for item in p_label]) for p_label in pred_labels]) + '\n')


if __name__ == '__main__':
    parser = get_training_options_bert()
    args = parser.parse_args()
    train(args)
